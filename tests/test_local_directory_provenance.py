"""Mixed Writer provenance and recovery boundaries, without model inference."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from agmem.build_local_directories import (
    ProvenanceStore, _entry_metadata, _version, import_history, load_recipe,
)
from agmem.core import Example, atomic_json, digest
from agmem.data import clean_history
from agmem.pipeline import validate_directories
from agmem.services import DirectoryBuilder
from agmem.store import Store


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


class LocalDirectoryProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.writer = {"lookback_sessions": 3, "content_max_tokens": 300}
        self.local = {"base_url": "http://127.0.0.1:50543/v1", "model": "agmem-27b",
                      "model_revision": "fixture-qwen"}
        self.remote = {"base_url": "https://writer.invalid/v1", "model": "gpt-fixture",
                       "model_revision": "fixture-gpt", "request_profile": "openai_chat"}
        self.mixed = {"base_url": "http://mixed-session-directory.invalid/v1",
                      "model": "mixed-session-directory", "model_revision": "fixture-contract"}
        self.version = _version(self.mixed, self.writer)
        self.records = clean_history({"haystack_sessions": [
            [{"role": "user", "content": f"Original session {i}"}] for i in range(4)]})
        self.source = Store(self.root / "gpt" / "split_longmemeval_s")
        self.memory_id = self.source.add_memory(self.records)
        self.spec = {"root": str(self.source.root), "label": "GPT", "available": True,
                     "writer_identity": self.remote, "directory_version": _version(self.remote, self.writer)}
        self.target = ProvenanceStore(self.root / "target", self.local, [self.spec])
        self.assertEqual(self.target.add_memory(self.records), self.memory_id)
        self.addCleanup(lambda: self.source.close())
        self.addCleanup(lambda: self.target.close())

    def entry(self, index, content=None):
        return {**_entry_metadata(self.records[index]),
                "content": content if content is not None else f"Summary {index}"}

    def key(self, index=0):
        return self.memory_id, self.version, self.records[index]["id"]

    def save_remote(self, index):
        entry = self.entry(index, f"GPT summary {index}")
        self.source.put_directory_entry(self.memory_id, self.spec["directory_version"], entry)
        return entry

    def test_unpublished_partial_source_keeps_exact_summaries_and_writer_identity(self):
        expected = self.save_remote(1)
        before = {path: path.read_bytes() for path in self.source.root.iterdir() if path.is_file()}
        imported = import_history(self.target, self.memory_id, self.version, [(self.source, self.spec)])
        self.assertEqual(dict(imported), {"GPT": 1})
        self.assertIsNone(self.source.get_metadata("directory_version"))
        self.assertEqual(self.target.directory_entry(*self.key(1)), expected)
        journal = self.target.provenance.get(self.key(1))
        self.assertEqual(journal["origin"], ProvenanceStore.import_origin(self.spec))
        self.assertEqual(journal["entry_sha256"], digest(expected))
        self.assertEqual(before, {path: path.read_bytes() for path in self.source.root.iterdir() if path.is_file()})

    def test_same_history_id_with_changed_raw_messages_is_rejected(self):
        self.save_remote(0)
        changed = deepcopy(self.source.histories.get((self.memory_id,)))
        changed["records"][3]["messages"][0]["content"] = "Changed unseen context"
        self.source.histories.put(changed)
        with self.assertRaisesRegex(ValueError, "different complete records"):
            import_history(self.target, self.memory_id, self.version, [(self.source, self.spec)])
        self.assertFalse(list(self.target.directory_entries.keys()))

    def test_source_public_metadata_must_match_the_exact_session(self):
        changed = self.entry(0)
        changed["time_order"]["order"] = 2
        self.source.put_directory_entry(self.memory_id, self.spec["directory_version"], changed)
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            import_history(self.target, self.memory_id, self.version, [(self.source, self.spec)])

    def test_journal_replays_a_crash_before_the_directory_append(self):
        entry = self.entry(0)
        origin = ProvenanceStore.import_origin(self.spec)
        with patch.object(Store, "put_directory_entry", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.target.put_with_origin(self.memory_id, self.version, entry, origin)
        self.assertIsNone(self.target.directory_entry(*self.key()))
        self.target.close()
        self.target = ProvenanceStore(self.target.root, self.local, [self.spec])
        self.target.recover_history(self.memory_id, self.version, self.records)
        self.assertEqual(self.target.directory_entry(*self.key()), entry)
        self.assertEqual(self.target.provenance.get(self.key())["origin"], origin)

    def test_entry_without_provenance_cannot_be_assigned_a_guessed_writer(self):
        Store.put_directory_entry(self.target, self.memory_id, self.version, self.entry(0))
        with self.assertRaisesRegex(ValueError, "no provenance"):
            self.target.recover_history(self.memory_id, self.version, self.records)

    def test_changed_entry_digest_is_rejected_before_replay(self):
        self.target.put_directory_entry(self.memory_id, self.version, self.entry(0))
        journal = self.target.provenance.get(self.key())
        journal["entry"]["content"] = "Different summary"
        self.target.provenance.put(journal)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.target.recover_history(self.memory_id, self.version, self.records)

    def test_origin_changes_cannot_pass_with_an_unchanged_entry_digest(self):
        self.target.put_with_origin(self.memory_id, self.version, self.entry(0),
                                    ProvenanceStore.import_origin(self.spec))
        original = self.target.provenance.get(self.key())
        for field, value in (("source_root", "/another/source"), ("source_version", "0" * 64),
                             ("writer_label", "Qwen"), ("writer_identity", self.local)):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed["origin"][field] = value
                self.target.provenance.put(changed)
                with self.assertRaisesRegex(ValueError, "provenance contract"):
                    self.target.recover_history(self.memory_id, self.version, self.records)
        self.target.provenance.put(original)
        self.target.recover_history(self.memory_id, self.version, self.records)

    def test_mixed_snapshot_passes_existing_frozen_reader_validation(self):
        remote = self.save_remote(1)
        import_history(self.target, self.memory_id, self.version, [(self.source, self.spec)])
        generated = []

        def complete(messages, *args, **kwargs):
            index = json.loads(messages[1]["content"])["TARGET_SESSION"]["order"]
            generated.append(index)
            return f"Local summary {index}"

        builder = DirectoryBuilder(self.target, SimpleNamespace(identity=self.mixed, complete=complete),
                                   self.writer, WordTokenizer())
        entries = builder.build(self.memory_id, workers=2)
        self.assertEqual(set(generated), {0, 2, 3})
        self.assertEqual(entries[1], remote)
        self.target.add_example(Example("fixture/test/0", "fixture", "test", self.memory_id, "Question"))
        self.target.set_metadata("directory_version", self.version)
        self.target.set_metadata("directory_service_identity", self.mixed)
        config = {"data": {"frozen_directory_version": self.version}, "writer": self.writer}
        validate_directories(config, self.target, ["test"])
        for index, entry in enumerate(self.target.directory(self.memory_id, self.version)):
            self.assertEqual(set(entry), set(Store.ENTRY_FIELDS))
            origin = self.target.provenance.get(self.key(index))["origin"]
            self.assertEqual(origin["writer_identity"], self.remote if index == 1 else self.local)

    def test_deployment_ports_and_concurrency_do_not_change_mixed_identity(self):
        saved = {"config": {"data": {}, "models": {"task_path": str(self.root / "model")},
                            "writer": self.writer, "service": self.local}}
        atomic_json(self.root / "training" / "run.json", saved)
        atomic_json(self.root / "input" / "manifest.json", {"fixture": True})
        self.source.set_metadata("prepare_complete", True)
        self.source.set_metadata("prepare_identity", {"sources": ["split_longmemeval_s"]})
        self.source.set_metadata("directory_service_identity", self.remote)
        recipe = {"training_run": str(self.root / "training"), "input_root": str(self.root / "input"),
                  "prepared_root": str(self.root / "completed"), "reuse_train_root": str(self.root / "old"),
                  "reuse_test_roots": [{"root": str(self.root / "gpt"), "label": "GPT"}]}
        path = self.root / "recipe.yaml"
        path.write_text(yaml.safe_dump(recipe))
        with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture-qwen"}):
            first = load_recipe(path, 16, ["http://localhost:1111/v1"])
            second = load_recipe(path, 48, ["http://localhost:2222/v1", "http://localhost:3333/v1"])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])
        self.assertEqual(_version(first[3], self.writer), _version(second[3], self.writer))
        self.assertNotEqual(first[1]["execution"], second[1]["execution"])
        self.assertNotEqual(first[4], second[4])


if __name__ == "__main__":
    unittest.main()
