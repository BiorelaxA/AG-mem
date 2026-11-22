"""Frozen DeepSeek work survives an explicit, independently journaled import."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agmem import import_deepseek_directories as importer
from agmem.build_adjacent_directories import AdjacentBuildStore
from agmem.core import canonical, file_digest
from agmem.hierarchical.binding import bind_directory, directory_version
from agmem.hierarchical.records import make_session, time_metadata
from agmem.jobs import file_lock


class ImportDeepSeekDirectoriesTests(unittest.TestCase):
    def setUp(self):
        progress = patch.object(importer, "announce")
        self.progress = progress.start()
        self.addCleanup(progress.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.source, self.target = self.home / "deepseek", self.home / "qwen"
        self.data = self.home / "data.json"
        self.data.write_text('{"fixture": "immutable dataset"}')
        self.count = lambda text: len(text.split())
        self.old_identity = {
            "protocol": importer.SOURCE_PROTOCOL,
            "datasets": [{"spec": {"name": "split_longmemeval_m", "path": str(self.data), "split": "test",
                                    "format": "longmemeval_json", "modality_policy": "existing_captions"},
                          "files": [{"split": "test", "path": str(self.data), "sha256": file_digest(self.data)}]}],
            "writer": {"model": "deepseek-flash", "model_revision": "provider-alias:deepseek-flash"},
            "tokenizer_id": "fixture-qwen-tokenizer",
            "writer_settings": {"context_tokens": 65536, "parent_content_tokens": 400, "child_content_tokens": 300},
            "embedding": {"ready": True, "model_revision": "fixed", "tokenizer_id": "embedding-tokenizer",
                          "settings": {"model_id": "Qwen/Qwen3-Embedding-4B", "instruction": "topic/event", "expected_dimensions": 2560}},
            "segmentation": {"high_threshold": .72, "low_threshold": .5},
            "prompt_hashes": {"child.txt": "child", "parent.txt": "parent", "boundary.txt": "topic-event"},
            "prompt_version": "directory-writer.adjacent.v1", "block_policy": "block.v1",
            "boundary_prompt_version": "topic_event_v2", "base_resource_hashes": {"schema": "fixed"},
            "implementation_hashes": {"source.py": "old"},
            "request_options": {"response_format": "json_object"},
        }
        self.new_identity = deepcopy(self.old_identity)
        self.new_identity.update(protocol=importer.TARGET_PROTOCOL,
                                 writer={"model": "qwen3.5-27b", "model_revision": "local-qwen"},
                                 implementation_hashes={"target.py": "new"},
                                 request_options={"structured_outputs": True})
        self.sessions = []
        for i in range(1, 7):
            records = [] if i == 3 else [{"record_id": f"p{i:04d}:r0001", "seq": 1, "kind": "message",
                                         "speaker": "user", "call_id": None, "text": f"source fact {i}", "attachments": []}]
            self.sessions.append(make_session(bank_id="bank_fixture", parent_id=f"p{i:04d}", order=i,
                records=records, time=time_metadata("2024-01-01"), participants=["user"] if records else [],
                count_tokens=self.count))
        self.bank = {"bank_id": "bank_fixture", "dataset": "split_longmemeval_m", "sessions": self.sessions,
                     "source_map": [{"parent_id": s["parent_id"], "source_hash": s["source_hash"]} for s in self.sessions],
                     "query_memberships": [{"original_question_id": "q1"}], "issues": []}
        store = AdjacentBuildStore(self.source, self.old_identity)
        store.add_bank(self.bank)
        for index, session in enumerate(self.sessions[:5]):
            store.add_session(session)
            if index < 3:
                status = ["complete", "fallback", "empty"][index]
                directory = self.directory(session, self.old_identity, status)
                store.commit(session, directory, {"calls": index + 1, "provider_usage": {"output_tokens": 7}})
            elif index == 4:
                store.failed(session, "InfrastructureError", "credit exhausted")
        store.save_segments("old-segmentation", {"spans": [], "stats": {"review_calls": 8}})
        store.cache("old-writer", {"status": "complete", "model": "deepseek-flash"})
        store.trace({"large_remote_request": "source trace must stay in source"})
        store.close()
        (self.source / ".build.lock").touch()
        (self.source / "embedding_cache.sqlite3").write_bytes(b"not imported")

    def directory(self, session, identity, status="complete"):
        result = {"parent": {"title": "Topic", "content": "Source facts"}, "status": status, "mode": "single_session",
                  "children": [] if status == "empty" else [
                      {"end_block": 1, "title": "Topic", "content": "Source fact", "entities": [], "start_reason": "session_start"}]}
        return bind_directory(session, result, identity["writer"], identity["tokenizer_id"],
                              count_tokens=self.count, prompt_version=identity["prompt_version"])

    def run_import(self, **kwargs):
        return importer.import_committed(self.source, self.target, self.new_identity,
                                         count_tokens=self.count, **kwargs)

    def files(self, root):
        return {str(path.relative_to(root)): file_digest(path) for path in root.rglob("*") if path.is_file()}

    def connect(self, root):
        connection = sqlite3.connect(root / "build_journal.sqlite3")
        self.addCleanup(connection.close)
        return connection

    def test_import_preserves_source_and_exact_directory_provenance_without_caches(self):
        before = self.files(self.source)
        report = self.run_import()
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["counters"]["imported_complete"], 1)
        self.assertEqual(report["counters"]["imported_fallback"], 1)
        self.assertEqual(report["counters"]["imported_empty"], 1)
        self.assertEqual(report["counters"]["imported_pending"], 2)
        self.assertEqual(self.files(self.source), before)
        source, target = self.connect(self.source), self.connect(self.target)
        for pid in ("p0001", "p0002", "p0003"):
            query = "SELECT snapshot,status,directory,result_stats FROM sessions WHERE parent_id=?"
            self.assertEqual(source.execute(query, (pid,)).fetchone(), target.execute(query, (pid,)).fetchone())
        self.assertEqual(target.execute("SELECT status,error,directory,result_stats FROM sessions WHERE parent_id='p0005'").fetchone(),
                         ("pending", None, None, None))
        self.assertEqual(target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 5)
        for table in ("writer_cache", "segment_cache", "writer_traces"):
            self.assertEqual(target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        self.assertFalse((self.target / "embedding_cache.sqlite3").exists())
        self.assertEqual(target.execute("SELECT COUNT(*) FROM reuse_provenance").fetchone()[0], 5)
        provenance = json.loads(target.execute("SELECT payload FROM reuse_provenance WHERE parent_id='p0005'").fetchone()[0])
        self.assertEqual(provenance["source_error"]["reason"], "credit exhausted")
        self.assertEqual(provenance["source_status"], "failed")
        self.assertIsNotNone(target.execute("SELECT value FROM metadata WHERE key=?", (importer.COMPLETE_MARKER,)).fetchone())
        manifest = json.loads((self.target / "deepseek_import_manifest.json").read_text())
        self.assertEqual(manifest["source_identity"], self.old_identity)
        self.assertEqual(manifest["target_identity"], self.new_identity)
        self.assertEqual(len((self.target / "reuse_provenance.jsonl").read_text().splitlines()), 5)
        self.assertEqual(self.progress.call_args.kwargs["processed"], 5)
        self.assertEqual(self.progress.call_args.kwargs["total"], 5)

    def test_dry_run_is_read_only_and_never_tokenizes(self):
        before = self.files(self.source)
        with patch("agmem.policy.load_tokenizer", side_effect=AssertionError("must not load a model")):
            report = importer.import_committed(self.source, self.target, self.new_identity, dry_run=True,
                                               count_tokens=lambda text: self.fail("dry run must not tokenize"))
        self.assertEqual(report["source_states"], {"complete": 1, "fallback": 1, "empty": 1, "pending": 1, "failed": 1})
        self.assertFalse(self.target.exists())
        self.assertEqual(self.files(self.source), before)

    def test_changed_semantics_are_rejected_before_target_creation(self):
        changes = [
            ("tokenizer_id", "different"), ("writer_settings", {"parent_content_tokens": 999}),
            ("segmentation", {"high_threshold": .99}), ("embedding", {"model_revision": "different"}),
            ("prompt_hashes", {"child.txt": "changed"}), ("boundary_prompt_version", "old-prompt"),
            ("protocol", importer.SOURCE_PROTOCOL),
        ]
        for field, value in changes:
            identity = deepcopy(self.new_identity)
            identity[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                importer.import_committed(self.source, self.target, identity)
            self.assertFalse(self.target.exists())
        for field, value in [("format", "other_adapter"), ("modality_policy", "references_only")]:
            identity = deepcopy(self.new_identity)
            identity["datasets"][0]["spec"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "datasets/spec"):
                importer.import_committed(self.source, self.target, identity)
        identity = deepcopy(self.new_identity)
        identity["datasets"][0]["files"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            importer.import_committed(self.source, self.target, identity)
        self.assertFalse(self.target.exists())

    def test_writer_transport_and_embedding_execution_settings_can_change(self):
        self.new_identity["embedding"]["settings"].update(workers=256, batch_size=1, timeout_seconds=45)
        self.new_identity["directory_reuse_policy"] = "explicit-frozen-deepseek"
        report = self.run_import()
        self.assertEqual(report["target"]["states"]["complete"], 1)

    def test_changed_dataset_bytes_are_rejected(self):
        self.data.write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256 changed"):
            self.run_import()
        self.assertFalse(self.target.exists())

    def test_active_source_and_target_builds_are_rejected(self):
        with file_lock(self.source / ".build.lock"), self.assertRaisesRegex(RuntimeError, "Source directory builder is active"):
            self.run_import()
        self.assertFalse(self.target.exists())
        with file_lock(self.target / ".build.lock"), self.assertRaisesRegex(RuntimeError, "Another job holds"):
            self.run_import()
        self.assertEqual({path.name for path in self.target.iterdir()}, {".build.lock"})

    def test_source_and_nested_targets_are_rejected(self):
        for target in (self.source, self.source / "child", self.source.parent):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "separate"):
                importer.import_committed(self.source, target, self.new_identity)

    def test_resume_after_partial_import_is_idempotent_and_atomic(self):
        original = importer._import_session
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted fixture")
            return original(*args, **kwargs)
        with patch.object(importer, "_import_session", side_effect=interrupt), self.assertRaisesRegex(RuntimeError, "interrupted fixture"):
            self.run_import()
        target = self.connect(self.target)
        self.assertEqual(target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 2)
        self.assertEqual(target.execute("SELECT COUNT(*) FROM reuse_provenance").fetchone()[0], 2)
        self.assertIsNone(target.execute("SELECT value FROM metadata WHERE key=?", (importer.COMPLETE_MARKER,)).fetchone())
        self.assertFalse((self.target / "deepseek_import_manifest.json").exists())
        result = self.run_import()
        self.assertEqual(result["counters"]["target_sessions_preserved"], 2)
        self.assertEqual(target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 5)
        again = self.run_import()
        self.assertEqual(again["counters"]["target_sessions_preserved"], 5)
        self.assertEqual(target.execute("SELECT COUNT(*) FROM reuse_provenance").fetchone()[0], 5)

    def test_resume_preserves_new_qwen_completion_and_its_stats(self):
        self.run_import()
        store = AdjacentBuildStore(self.target, self.new_identity)
        session = self.sessions[4]
        qwen = self.directory(session, self.new_identity)
        store.commit(session, qwen, {"qwen_calls": 3})
        store.close()
        self.run_import()
        target = self.connect(self.target)
        row = target.execute("SELECT directory,result_stats FROM sessions WHERE parent_id='p0005'").fetchone()
        self.assertEqual(json.loads(row[0]), qwen)
        self.assertEqual(json.loads(row[1]), {"qwen_calls": 3})

    def test_completed_import_fast_path_does_not_scan_or_reexport_directories(self):
        self.run_import()
        provenance = (self.target / "reuse_provenance.jsonl").read_bytes()
        manifest = (self.target / "deepseek_import_manifest.json").read_bytes()
        with patch.object(importer, "_import_session", side_effect=AssertionError("must not rescan")), \
                patch.object(AdjacentBuildStore, "_jsonl", side_effect=AssertionError("must not reexport")):
            result = self.run_import()
        self.assertTrue(result["import_already_complete"])
        self.assertFalse(result["token_counts_rechecked"])
        self.assertEqual((self.target / "reuse_provenance.jsonl").read_bytes(), provenance)
        self.assertEqual((self.target / "deepseek_import_manifest.json").read_bytes(), manifest)

    def test_frozen_source_cannot_change_after_a_completed_import(self):
        self.run_import()
        source = self.connect(self.source)
        with source:
            source.execute("UPDATE sessions SET result_stats=? WHERE parent_id='p0001'", ('{"changed":true}',))
        with self.assertRaisesRegex(ValueError, "Frozen DeepSeek import"):
            self.run_import()

    def test_preexisting_target_completion_is_not_relabelled_or_given_import_provenance(self):
        store = AdjacentBuildStore(self.target, self.new_identity)
        store.add_bank(self.bank)
        store.add_session(self.sessions[0])
        qwen = self.directory(self.sessions[0], self.new_identity)
        store.commit(self.sessions[0], qwen, {"qwen_calls": 2})
        store.add_session(self.sessions[1])  # Unfinished target work can use a committed source directory.
        store.close()
        report = self.run_import()
        self.assertEqual(report["counters"]["target_sessions_preserved"], 1)
        target = self.connect(self.target)
        self.assertEqual(json.loads(target.execute("SELECT directory FROM sessions WHERE parent_id='p0001'").fetchone()[0]), qwen)
        self.assertIsNone(target.execute("SELECT payload FROM reuse_provenance WHERE parent_id='p0001'").fetchone())
        self.assertEqual(target.execute("SELECT status FROM sessions WHERE parent_id='p0002'").fetchone()[0], "fallback")

    def test_corrupt_directory_is_not_imported_or_marked_complete(self):
        source = self.connect(self.source)
        with source:
            value = json.loads(source.execute("SELECT directory FROM sessions WHERE parent_id='p0001'").fetchone()[0])
            value["content"] = "tampered without updating immutable version"
            source.execute("UPDATE sessions SET directory=? WHERE parent_id='p0001'", (canonical(value),))
        with self.assertRaisesRegex(ValueError, "Directory version"):
            self.run_import()
        target = self.connect(self.target)
        self.assertEqual(target.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)
        self.assertIsNone(target.execute("SELECT value FROM metadata WHERE key=?", (importer.COMPLETE_MARKER,)).fetchone())

    def test_token_counts_and_original_writer_are_verified(self):
        source = self.connect(self.source)
        original = json.loads(source.execute("SELECT directory FROM sessions WHERE parent_id='p0001'").fetchone()[0])
        for mutation in ("tokens", "writer"):
            changed = deepcopy(original)
            if mutation == "tokens":
                changed["source"]["body_token_count"] += 1
            else:
                changed["build"]["writer_model"] = "silently-relabelled-qwen"
            changed["directory_version"] = directory_version(changed)
            with source:
                source.execute("UPDATE sessions SET directory=? WHERE parent_id='p0001'", (canonical(changed),))
            target = self.home / mutation
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                importer.import_committed(self.source, target, self.new_identity, count_tokens=self.count)

    def test_existing_target_identity_is_checked_even_on_dry_run(self):
        wrong = deepcopy(self.new_identity)
        wrong["writer"]["model_revision"] = "other"
        store = AdjacentBuildStore(self.target, wrong)
        store.close()
        before = self.files(self.target)
        with self.assertRaisesRegex(ValueError, "Target construction identity changed"):
            self.run_import(dry_run=True)
        self.assertEqual(self.files(self.target), before)

    def test_source_journal_symlink_cannot_be_used_as_target(self):
        self.target.mkdir()
        (self.target / "build_journal.sqlite3").symlink_to(self.source / "build_journal.sqlite3")
        before = self.files(self.source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_import()
        self.assertEqual(self.files(self.source), before)


if __name__ == "__main__":
    unittest.main()
