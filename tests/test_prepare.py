from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import atomic_json, canonical, file_digest
from agmem.data import ADAPTER_VERSION, clean_history
from agmem.pipeline import build_directories, make_environment, open_prepared, prepare
from agmem.services import IncompleteServiceCompletion
from agmem.store import Store


class PreparationTests(unittest.TestCase):
    def test_pending_summary_does_not_stop_build_and_next_pass_backfills_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"data": {"prepared_root": temporary, "sources": ["fixture"], "splits": ["train"]},
                      "models": {"task_path": "fixture"},
                      "service": {"model": "fixture", "base_url": "http://fixture/v1"},
                      "writer": {"lookback_sessions": 3, "content_max_tokens": 300}}
            store = Store(temporary)
            memory_id = store.add_memory(clean_history({
                "haystack_sessions": [[{"role": "user", "content": "A real session."}]],
                "haystack_session_ids": ["session0"]}))
            store.set_metadata("prepare_complete", True)
            store.set_metadata("prepare_identity", {"adapter_version": ADAPTER_VERSION,
                "storage_format": Store.FORMAT, "sources": ["fixture"], "splits": ["train"]})
            store.close()
            writer = SimpleNamespace(identity={**config["service"], "model_revision": "fixture-v1"},
                complete=Mock(side_effect=[IncompleteServiceCompletion(None, "stop", None), "Recovered summary."]))
            tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
            with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture-v1"}), \
                    patch("agmem.policy.load_tokenizer", return_value=tokenizer), \
                    patch("agmem.services.ChatService", return_value=writer):
                version = build_directories(config)
                store = open_prepared(config)
                self.assertEqual(store.get_metadata("directory_empty_pending"), 1)
                self.assertEqual(store.get_metadata("directory_version"), version)
                self.assertEqual(store.directory(memory_id, version)[0]["writer_status"], "empty_pending")
                store.close()
                self.assertEqual(build_directories(config), version)
                store = open_prepared(config)
                self.assertEqual(store.get_metadata("directory_empty_pending"), 0)
                self.assertEqual(store.directory(memory_id, version)[0]["content"], "Recovered summary.")
                self.assertNotIn("writer_status", store.directory(memory_id, version)[0])
                self.assertEqual(writer.complete.call_count, 2)
                store.close()

    def test_fresh_preparation_preserves_empty_sessions_and_isolates_invalid_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            sources = []
            for source in ["perltqa_en", "zh4o"]:
                directory = root / source
                directory.mkdir(parents=True)
                entry = {"source": source, "output_files": {}}
                for split in ["train", "validation", "test"]:
                    row = {"question_id": "q", "question": "q", "answer": "reference",
                           "haystack_sessions": canonical([[], [{"role": "user", "content": "original", "has_answer": True}]]),
                           "haystack_session_ids": ["empty", "answer_actual"]}
                    if source == "zh4o":
                        row["question"] = "问题 A 一 B 二 C 三 D 四"
                        row["answer"] = "invalid" if split == "train" else "A"
                    path = directory / f"{split}.parquet"
                    pq.write_table(pa.Table.from_pylist([row]), path)
                    entry["output_files"][split] = {"sha256": file_digest(path)}
                sources.append(entry)
            atomic_json(root / "manifest.json", {"sources": sources})
            config = {"data": {"input_root": str(root), "prepared_root": str(Path(temporary) / "prepared"),
                               "sources": ["perltqa_en", "zh4o"], "splits": ["train", "validation", "test"]}}
            report = prepare(config)
            self.assertEqual(report["counts"]["perltqa_en/train/valid"], 1)
            self.assertEqual(report["counts"]["zh4o/train/quarantined"], 1)
            store = open_prepared(config)
            try:
                self.assertEqual(len(store.examples("train")), 1)
                self.assertEqual(len(store.examples("validation")), 2)
                example = store.examples("train")[0]
                records = store.memory(example.memory_id)
                self.assertEqual(len(records), 2)
                self.assertEqual(store.record(records[0])["messages"], [])
                self.assertNotIn("has_answer", store.record(records[1])["messages"][0])
            finally:
                store.close()

            config.update({"models": {"task_path": "fixture"},
                           "service": {"model": "fixture", "base_url": "http://fixture/v1"},
                           "writer": {"lookback_sessions": 3, "content_max_tokens": 300},
                           "retrieval": {"max_ids_per_call": 10, "max_read_calls": 8},
                           "task_agent": {"max_context_tokens": None}})
            writer = SimpleNamespace(identity={**config["service"], "model_revision": "fixture-v1"},
                                     complete=Mock(return_value="Original message."))
            tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
            with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture-v1"}), \
                    patch("agmem.policy.load_tokenizer", return_value=tokenizer), \
                    patch("agmem.services.ChatService", return_value=writer):
                version = build_directories(config)
                store = open_prepared(config)
                try:
                    env = make_environment(config, store, SimpleNamespace(check_capacity=lambda messages: None))
                    self.assertEqual(env.version, version)
                    self.assertEqual(len(store.directory(store.examples("train")[0].memory_id, version)), 2)
                    self.assertEqual(writer.complete.call_count, 1)  # The other session is empty.
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
