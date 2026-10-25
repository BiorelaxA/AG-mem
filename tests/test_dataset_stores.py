from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import Example, atomic_json, canonical, file_digest
from agmem.data import ADAPTER_VERSION, clean_history
from agmem.pipeline import (build_directories, dataset_configs, import_prepared,
                            open_prepared, prepare, validate_directories)
from agmem.services import DirectoryBuilder
from agmem.store import Store
from agmem.workflow import make_workflow_configs


class DatasetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {
            "data": {"layout": "by_source", "input_root": str(self.root / "input"),
                     "prepared_root": str(self.root / "datasets"),
                     "sources": ["locomo", "split_longmemeval_s"], "splits": ["train", "validation"]},
            "models": {"task_path": "fixture"},
            "service": {"model": "fixture", "base_url": "http://fixture/v1"},
            "writer": {"lookback_sessions": 3, "content_max_tokens": 300}}
        self.identity = {**self.config["service"], "model_revision": "fixture-v1"}
        self.writer = SimpleNamespace(identity=self.identity, complete=Mock(return_value="A useful summary."))
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        for target, value in (("agmem.policy.model_fingerprint", {"fingerprint": "fixture-v1"}),
                              ("agmem.policy.load_tokenizer", self.tokenizer),
                              ("agmem.services.ChatService", self.writer)):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp.cleanup()

    def make_store(self, root, sources, partial=False):
        store = Store(root)
        store.set_metadata("prepare_identity", {
            "input_manifest_sha256": file_digest(Path(self.config["data"]["input_root"]) / "manifest.json"),
            "adapter_version": ADAPTER_VERSION, "storage_format": Store.FORMAT,
            "sources": sources, "splits": ["train", "validation"], "limit_per_source_split": None})
        for source in sources:
            records = clean_history({"haystack_sessions": [
                [{"role": "user", "content": source + " first fact"}],
                [{"role": "user", "content": source + " second fact"}]], "haystack_session_ids": ["s0", "s1"]})
            mid = store.add_memory(records)
            for split, n in (("train", 2), ("validation", 1)):
                for i in range(n):
                    store.add_example(Example(f"{source}/{split}/{i}", source, split, mid, "question"))
            if partial:
                builder = DirectoryBuilder(store, self.writer, self.config["writer"], self.tokenizer)
                r = records[0]
                store.put_directory_entry(mid, builder.version, {
                    "id": r["id"], "task_session": {"task_id": None, "session_id": r["source_session_id"]},
                    "time_order": {"time": r["time"], "order": 0},
                    "actors_source": {"actors": ["user"], "sources": []}, "content": "Already generated."})
        store.set_metadata("prepare_complete", True)
        store.close()

    def fixture_stores(self):
        atomic_json(Path(self.config["data"]["input_root"]) / "manifest.json", {})
        for source, child in dataset_configs(self.config):
            self.make_store(child["data"]["prepared_root"], [source])

    def test_separate_build_then_combined_training_view_preserves_splits_and_reuses_history(self):
        self.fixture_stores()
        build_directories(self.config, sources=["locomo"])
        self.assertEqual(self.writer.complete.call_count, 2)  # 3 queries share two sessions.
        other = Path(self.config["data"]["prepared_root"]) / "split_longmemeval_s" / "directories.jsonl"
        self.assertFalse(other.exists())
        store = open_prepared(self.config)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_directories(self.config, store, ["train", "validation"])
        store.close()
        build_directories(self.config, sources=["split_longmemeval_s"])
        build_directories(self.config)
        self.assertEqual(self.writer.complete.call_count, 4)
        store = open_prepared(self.config)
        try:
            validate_directories(self.config, store, ["train", "validation"])
            self.assertEqual(len(store.examples("train")), 4)
            self.assertEqual(len(store.examples("validation")), 2)
            for example in store.examples("train"):
                refs = store.memory(example.memory_id)
                self.assertIn(example.source, store.record(refs[0])["raw_text"])
                self.assertEqual(len(store.directory(example.memory_id, store.get_metadata("directory_version"))), 2)
            store.put_completion("task-key", "answer")
            self.assertEqual(store.completion("task-key"), "answer")
        finally:
            store.close()

    def test_subset_training_does_not_require_unselected_dataset(self):
        self.fixture_stores()
        build_directories(self.config, sources=["locomo"])
        subset = deepcopy(self.config)
        subset["data"]["sources"] = ["locomo"]
        store = open_prepared(subset)
        try:
            validate_directories(subset, store, ["train", "validation"])
            self.assertEqual({e.source for e in store.examples("train")}, {"locomo"})
        finally:
            store.close()

    def test_workflow_s_training_does_not_require_m_and_reuses_directories(self):
        self.fixture_stores()
        config = deepcopy(self.config)
        config["data"]["sources"].append("split_longmemeval_m")
        config["training"] = {"output_dir": str(self.root / "train")}
        before = deepcopy(config)
        plan = {"directory_sources": ["locomo", "split_longmemeval_s"],
                "training_sources": ["locomo", "split_longmemeval_s"], "experiment": "locomo_s"}
        build, train = make_workflow_configs(config, plan, self.root / "job")
        self.assertEqual(config, before)
        self.assertEqual(build["data"]["prepared_root"], train["data"]["prepared_root"])
        self.assertEqual(train["training"]["output_dir"], str(self.root / "experiments/locomo_s/train"))
        build_directories(build)
        store = open_prepared(train)
        try:
            validate_directories(train, store, ["train", "validation"])
            self.assertEqual({e.source for e in store.examples("train")}, set(plan["training_sources"]))
            self.assertEqual(len(store.examples("validation")), 2)
        finally:
            store.close()
        self.assertFalse((self.root / "datasets/split_longmemeval_m").exists())
        build_directories(build)
        self.assertEqual(self.writer.complete.call_count, 4)
        # Later M construction and all-source training have independent selections.
        later = {"directory_sources": ["split_longmemeval_m"],
                 "training_sources": config["data"]["sources"], "experiment": "locomo_s_m"}
        m_build, all_train = make_workflow_configs(config, later, self.root / "later")
        self.assertEqual(m_build["data"]["sources"], ["split_longmemeval_m"])
        self.assertEqual(all_train["data"]["sources"], config["data"]["sources"])
        self.assertNotEqual(train["training"]["output_dir"], all_train["training"]["output_dir"])

    def test_workflow_rejects_unknown_sources_and_reusing_experiment_with_changed_data(self):
        config = deepcopy(self.config)
        config["training"] = {"output_dir": str(self.root / "train")}
        plan = {"directory_sources": ["locomo"], "training_sources": ["locomo"], "experiment": "first"}
        _, train = make_workflow_configs(config, plan, self.root / "job")
        atomic_json(Path(train["training"]["output_dir"]) / "run.json", {"config": train})
        changed = {**plan, "training_sources": ["split_longmemeval_s"]}
        with self.assertRaisesRegex(ValueError, "differs in data"):
            make_workflow_configs(config, changed, self.root / "job2")
        changed = {**plan, "directory_sources": ["unknown"]}
        with self.assertRaisesRegex(ValueError, "subset"):
            make_workflow_configs(config, changed, self.root / "job3")
        changed = {**plan, "experiment": "../bad"}
        with self.assertRaisesRegex(ValueError, "single directory"):
            make_workflow_configs(config, changed, self.root / "job4")

    def test_limit_does_not_make_incomplete_directories_trainable(self):
        self.fixture_stores()
        build_directories(self.config, limit=0)
        store = open_prepared(self.config)
        try:
            with self.assertRaisesRegex(ValueError, "Directory missing"):
                validate_directories(self.config, store, ["train"])
        finally:
            store.close()

    def test_different_writer_versions_are_rejected(self):
        self.fixture_stores()
        build_directories(self.config)
        changed = deepcopy(self.config)
        changed["writer"]["content_max_tokens"] = 200
        build_directories(changed, sources=["locomo"])
        store = open_prepared(changed)
        try:
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                validate_directories(changed, store, ["train"])
        finally:
            store.close()

    def test_import_preserves_partial_summaries_and_resume_only_builds_missing_entries(self):
        atomic_json(Path(self.config["data"]["input_root"]) / "manifest.json", {})
        origin = self.root / "legacy"
        self.make_store(origin, ["locomo", "split_longmemeval_s", "unused"], partial=True)
        original = (origin / "directories.jsonl").read_bytes()
        report = import_prepared(self.config, origin)
        self.assertEqual(set(report), {"locomo", "split_longmemeval_s"})
        self.assertFalse((Path(self.config["data"]["prepared_root"]) / "unused").exists())
        build_directories(self.config)
        self.assertEqual(self.writer.complete.call_count, 2)
        import_prepared(self.config, origin)  # Never overwrite newly completed destination entries.
        build_directories(self.config)
        self.assertEqual(self.writer.complete.call_count, 2)
        self.assertEqual((origin / "directories.jsonl").read_bytes(), original)

    def test_source_filter_and_per_dataset_preparation_share_train_validation_history(self):
        root = Path(self.config["data"]["input_root"])
        sources = []
        for source in self.config["data"]["sources"]:
            entry = {"source": source, "output_files": {}}
            for split in ("train", "validation"):
                path = root / source / f"{split}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                row = {"question": "q", "answer": "a", "haystack_sessions": canonical([
                    [{"role": "user", "content": source}]]), "haystack_session_ids": ["s0"]}
                if source == "split_longmemeval_s":
                    row["question_date"] = "2023/04/20 (Thu) 10:12"
                pq.write_table(pa.Table.from_pylist([row]), path)
                entry["output_files"][split] = {"sha256": file_digest(path)}
            sources.append(entry)
        atomic_json(root / "manifest.json", {"sources": sources})
        prepare(self.config, sources=["locomo"])
        self.assertFalse((Path(self.config["data"]["prepared_root"]) / "split_longmemeval_s").exists())
        prepare(self.config, sources=["split_longmemeval_s"])
        store = open_prepared(self.config)
        try:
            self.assertEqual(len(store.memory_ids()), 2)
            self.assertEqual(len(store.examples("train")), 2)
            self.assertEqual(len(store.examples("validation")), 2)
        finally:
            store.close()
        with self.assertRaisesRegex(ValueError, "subset"):
            build_directories(self.config, sources=["unknown"])
