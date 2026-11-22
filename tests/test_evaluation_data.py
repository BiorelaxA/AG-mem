from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import Example, atomic_json, canonical, file_digest
from agmem.data import ADAPTER_VERSION, clean_history
from agmem.evaluation_data import _current_writer_version, prepare_source, build_source_directories
from agmem.pipeline import build_directories, open_prepared, validate_directory_version
from agmem.services import DirectoryBuilder
from agmem.store import Store


class EvaluationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"base_url": "http://original-writer/v1", "model": "qwen-local",
                         "model_revision": "old-file-fingerprint", "api_key_env": "AGMEM_API_KEY"}
        self.config = {
            "data": {"input_root": str(self.root / "input"), "sources": ["locomo"],
                     "prepared_root": str(self.root / "train"), "layout": "by_source",
                     "splits": ["train", "validation"], "frozen_directory_version": "a" * 64},
            "models": {"task_path": "unused-local-model"},
            "service": {"base_url": "http://current-writer/v1", "model": "qwen-local"},
            "writer": {"lookback_sessions": 3, "content_max_tokens": 300},
        }
        identity = {key: self.identity[key] for key in ("base_url", "model", "model_revision")}
        self.version = DirectoryBuilder(SimpleNamespace(FORMAT=Store.FORMAT),
                                       SimpleNamespace(identity=identity), self.config["writer"], None).version

    @staticmethod
    def row(fact, question="unseen test question"):
        return {"question": question, "answer": "reference never supplied to the Writer",
                "haystack_sessions": canonical([[{"role": "user", "content": fact}]]),
                "haystack_session_ids": ["session0"], "haystack_dates": ["2026-01-01"]}

    def input(self, rows):
        path = self.root / "input/locomo/test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        atomic_json(path.parent.parent / "manifest.json", {"sources": [
            {"source": "locomo", "output_files": {"test": {"sha256": file_digest(path)}}}]})

    def origin(self, rows, *, pending=False):
        store = Store(self.root / "train/locomo")
        store.set_metadata("prepare_complete", True)
        store.set_metadata("prepare_identity", {"adapter_version": ADAPTER_VERSION,
            "storage_format": Store.FORMAT, "sources": ["locomo"], "splits": ["train", "validation"]})
        for i, row in enumerate(rows):
            mid = store.add_memory(clean_history(row))
            store.add_example(Example(f"locomo/train/{i:06d}", "locomo", "train", mid, "training query"))
            entries = []
            for record in store.memory(mid):
                entry = {"id": record["id"], "task_session": {"task_id": None, "session_id": record["source_session_id"]},
                         "actors_source": {"actors": ["user"], "sources": []},
                         "time_order": {"time": record["time"], "order": record["order"]},
                         "content": "" if pending else "Original history summary."}
                if pending:
                    entry["writer_status"] = "empty_pending"
                entries.append(entry)
            store.put_directory(mid, self.version, entries)
        store.set_metadata("directory_version", self.version)
        store.set_metadata("directory_service_identity", deepcopy(self.identity))
        store.set_metadata("directory_empty_pending", int(pending))
        store.close()
        return self.root / "train/locomo"

    @staticmethod
    def bytes_at(root):
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def test_full_exact_history_reuse_preserves_original_provenance_and_training_store(self):
        self.input([self.row("same history"), self.row("same history", "another held-out query")])
        origin = self.origin([self.row("same history", "different training question")])
        before = self.bytes_at(origin)
        with patch("agmem.evaluation_data._current_writer_version", side_effect=AssertionError("no model needed")):
            result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(result["status"], "ready")
        self.assertEqual((result["examples"], result["histories"], result["reused_entries"]), (2, 1, 1))
        self.assertEqual(result["config"]["data"]["frozen_directory_version"], self.version)
        target = open_prepared(result["config"])
        try:
            self.assertEqual(target.get_metadata("directory_service_identity"), self.identity)
            self.assertEqual(target.examples("train"), [])
            self.assertEqual(target.directory(target.memory_ids()[0], self.version)[0]["content"], "Original history summary.")
        finally:
            target.close()
        self.assertEqual(self.bytes_at(origin), before)
        self.assertEqual(self.config["data"]["splits"], ["train", "validation"])
        with patch("agmem.evaluation_data.build_directories") as build:
            self.assertEqual(build_source_directories(result)["status"], "ready")
            build.assert_not_called()
        repeated = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(repeated["reused_histories"], 1)

    def test_different_complete_history_is_not_reused(self):
        self.input([self.row("new fact")])
        self.origin([self.row("old fact")])
        result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(result["status"], "needs_build")
        self.assertEqual(result["reused_histories"], 0)
        self.assertEqual(result["missing_histories"], 1)
        self.assertNotIn("frozen_directory_version", result["config"]["data"])

    def test_public_metadata_is_part_of_history_identity(self):
        changed = self.row("same messages")
        changed["haystack_dates"] = ["2026-02-01"]
        self.input([changed])
        self.origin([self.row("same messages")])
        result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(result["reused_histories"], 0)

    def test_pending_source_directory_does_not_qualify_as_complete_reuse(self):
        self.input([self.row("same history")])
        self.origin([self.row("same history")], pending=True)
        result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(result["status"], "needs_build")
        self.assertEqual(result["reused_entries"], 0)
        self.assertEqual(len(result["reuse"]["incomplete_source_histories"]), 1)

    def test_partial_reuse_requires_matching_writer_version(self):
        self.input([self.row("old history"), self.row("new history")])
        self.origin([self.row("old history")])
        with patch("agmem.evaluation_data._current_writer_version", return_value="b" * 64):
            with self.assertRaisesRegex(ValueError, "Writer version differs"):
                prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertFalse((self.root / "test/locomo/directories.jsonl").exists())
        with patch("agmem.evaluation_data._current_writer_version", return_value=self.version):
            result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(result["status"], "needs_build")
        self.assertEqual((result["reused_histories"], result["missing_histories"]), (1, 1))

    def test_build_publishes_and_freezes_the_actual_writer_version(self):
        self.input([self.row("new history")])
        result = prepare_source(self.config, "locomo", self.root / "test")

        def build(config):
            self.assertNotIn("frozen_directory_version", config["data"])
            store = open_prepared(config)
            for mid in store.memory_ids():
                ref = store.memory(mid)[0]
                store.put_directory(mid, self.version, [{"id": ref["id"],
                    "task_session": {"task_id": None, "session_id": ref["source_session_id"]},
                    "actors_source": {"actors": ["user"], "sources": []},
                    "time_order": {"time": ref["time"], "order": ref["order"]}, "content": "New summary."}])
            store.set_metadata("directory_version", self.version)
            store.set_metadata("directory_service_identity", deepcopy(self.identity))
            store.close()
            return self.version

        with patch("agmem.evaluation_data.build_directories", side_effect=build):
            done = build_source_directories(result)
        self.assertEqual(done["status"], "ready")
        self.assertEqual(done["missing_histories"], 0)
        self.assertEqual(done["config"]["data"]["frozen_directory_version"], self.version)

    def test_training_source_cannot_be_used_as_test_destination(self):
        self.input([self.row("history")])
        origin = self.origin([self.row("history")])
        before = self.bytes_at(origin)
        with self.assertRaisesRegex(ValueError, "separate store"):
            prepare_source(self.config, "locomo", self.root / "train", self.root / "train")
        self.assertEqual(self.bytes_at(origin), before)

    def remote_identity(self):
        return {"base_url": "https://remote-writer.example/v1", "model": "gpt-5-mini",
                "model_revision": "gpt-5-mini", "request_profile": "openai_chat",
                "reasoning_effort": "low", "max_completion_tokens": 2048}

    def use_remote_origin(self):
        self.identity = self.remote_identity()
        self.version = DirectoryBuilder(SimpleNamespace(FORMAT=Store.FORMAT),
            SimpleNamespace(identity=self.identity), self.config["writer"], None).version

    def test_current_writer_version_uses_full_remote_identity_without_local_fingerprint(self):
        identity = self.remote_identity()
        expected = DirectoryBuilder(SimpleNamespace(FORMAT=Store.FORMAT),
            SimpleNamespace(identity=identity), self.config["writer"], None).version
        with patch("agmem.remote_writer.writer_identity", return_value=identity) as identify, \
                patch("agmem.policy.model_fingerprint", side_effect=AssertionError("remote Writer")):
            self.assertEqual(_current_writer_version(self.config), expected)
            identify.assert_called_once_with(self.config)
        changed = {**identity, "reasoning_effort": "medium"}
        with patch("agmem.remote_writer.writer_identity", return_value=changed):
            self.assertNotEqual(_current_writer_version(self.config), expected)

    def test_frozen_remote_reuse_preserves_request_settings_and_rejects_changed_provenance(self):
        self.use_remote_origin()
        self.input([self.row("same history")])
        origin = self.origin([self.row("same history")])
        before = self.bytes_at(origin)
        with patch("agmem.remote_writer.writer_identity", side_effect=AssertionError("frozen Writer")), \
                patch("agmem.remote_writer.create_writer_service", side_effect=AssertionError("no API needed")), \
                patch("agmem.policy.model_fingerprint", side_effect=AssertionError("no model needed")):
            result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
            self.assertEqual(result["status"], "ready")
            self.assertEqual(build_source_directories(result)["status"], "ready")
        self.assertEqual(self.bytes_at(origin), before)
        store = open_prepared(result["config"])
        try:
            self.assertEqual(store.get_metadata("directory_service_identity"), self.identity)
            changed = {**self.identity, "reasoning_effort": "medium"}
            store.set_metadata("directory_service_identity", changed)
            with self.assertRaisesRegex(ValueError, "recorded Writer"):
                validate_directory_version(result["config"], store)
        finally:
            store.close()

    def test_remote_partial_reuse_requires_the_same_request_settings(self):
        self.use_remote_origin()
        self.input([self.row("old history"), self.row("new history")])
        self.origin([self.row("old history")])
        changed = {**self.identity, "reasoning_effort": "medium"}
        with patch("agmem.remote_writer.writer_identity", return_value=changed):
            with self.assertRaisesRegex(ValueError, "Writer version differs"):
                prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        with patch("agmem.remote_writer.writer_identity", return_value=self.identity):
            result = prepare_source(self.config, "locomo", self.root / "test", self.root / "train")
        self.assertEqual(result["status"], "needs_build")
        self.assertEqual((result["reused_histories"], result["missing_histories"]), (1, 1))

    def test_pipeline_remote_build_uses_task_tokenizer_and_publishes_service_identity(self):
        self.input([self.row("new history")])
        result = prepare_source(self.config, "locomo", self.root / "test")
        config = result["config"]
        identity = self.remote_identity()
        service = SimpleNamespace(identity=identity, complete=Mock(return_value="A remote summary."))
        tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda text, **kw: text.split()))
        with patch("agmem.remote_writer.create_writer_service", return_value=service) as create, \
                patch("agmem.policy.load_tokenizer", return_value=tokenizer) as load, \
                patch("agmem.policy.model_fingerprint", side_effect=AssertionError("remote Writer")):
            version = build_directories(config)
            self.assertIs(create.call_args.args[0], config)
            load.assert_called_once_with(self.config["models"]["task_path"])
        tokenizer.encode.assert_called_once_with("A remote summary.", add_special_tokens=False)
        store = open_prepared(config)
        try:
            self.assertEqual(store.get_metadata("directory_service_identity"), identity)
            self.assertEqual(store.get_metadata("directory_version"), version)
            self.assertEqual(store.get_metadata("directory_empty_pending"), 0)
            with patch("agmem.remote_writer.writer_identity", return_value=identity):
                self.assertEqual(validate_directory_version(config, store), version)
            frozen = deepcopy(config)
            frozen["data"]["frozen_directory_version"] = version
            with patch("agmem.remote_writer.writer_identity", side_effect=AssertionError("frozen Writer")):
                self.assertEqual(validate_directory_version(frozen, store), version)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
