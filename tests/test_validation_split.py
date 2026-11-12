"""Query-held-out validation expansion must preserve evidence and old artifacts."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import atomic_json, file_digest
from agmem.data import ADAPTER_VERSION, adapt_row, clean_history, task_examples
from agmem.pipeline import dataset_configs, validate_directories
from agmem.services import DirectoryBuilder
from agmem.store import Store
from agmem.validation_split import expand_validation, select_additions, support_count


def tree_hashes(root):
    return {str(path.relative_to(root)): file_digest(path)
            for path in root.rglob("*") if path.is_file()}


class ValidationSelectionTests(unittest.TestCase):
    def test_distinct_query_denominator_exclusions_and_balanced_reproducible_selection(self):
        rows = [{"question": f"query-{index:02d}",
                 "answer_session_ids": ["s0"] if index < 6 else [f"s{i}" for i in range(80)]}
                for index in range(12)]
        rows.append(deepcopy(rows[0]))
        selected, report = select_additions(rows, excluded_queries={"query-06"},
                                            fraction="0.25", seed=42)
        # ceil(12 distinct queries * .25) = 3; ceil(13 rows * .25) would be 4.
        self.assertEqual(report["train_rows_before"], 13)
        self.assertEqual(report["train_queries_before"], 12)
        self.assertEqual(report["added_queries"], 3)
        self.assertEqual(report["added_strata"], {"single": 1, "multi": 2})
        self.assertEqual(report["eligible_strata"], {"single": 5, "multi": 5})
        self.assertEqual(selected, sorted(selected))
        self.assertTrue({0, 6, 12}.isdisjoint(selected))
        self.assertEqual(Counter("single" if support_count(rows[i]) == 1 else "multi"
                                 for i in selected), {"single": 1, "multi": 2})
        self.assertEqual((selected, report), select_additions(
            rows, excluded_queries={"query-06"}, fraction="0.25", seed=42))

    def test_support_ids_are_normalized_and_deduplicated(self):
        self.assertEqual(support_count({"answer_session_ids": '["answer_answer_s0", "s0"]'}), 1)

    def test_target_counts_existing_validation_and_allows_per_source_override(self):
        rows = [{"question": str(i), "answer_session_ids": ["s0"] if i % 2 else ["s0", "s1"]}
                for i in range(20)]
        selected, report = select_additions(rows, mode="target", validation_queries=4, fraction="0.5")
        self.assertEqual(len(selected), 6)  # total 10, not an additional 10
        self.assertEqual(report["requested_validation_queries"], 10)
        selected, report = select_additions(rows, mode="target", validation_queries=4,
                                            fraction="0.5", target_queries=12)
        self.assertEqual(len(selected), 8)
        self.assertEqual(report["added_strata"], {"single": 4, "multi": 4})
        self.assertEqual(report["explicit_target_queries"], 12)
        selected, report = select_additions(rows, mode="target", validation_queries=12, fraction="0.5")
        self.assertEqual(selected, [])  # Never remove existing validation questions.
        with self.assertRaisesRegex(ValueError, "requires target mode"):
            select_additions(rows, target_queries=12)

    def test_insufficient_stratum_is_rejected(self):
        rows = [{"question": str(i), "answer_session_ids": ["s0"]} for i in range(8)]
        with self.assertRaisesRegex(ValueError, "Not enough eligible multi"):
            select_additions(rows, fraction="0.25")
        with self.assertRaisesRegex(ValueError, "at least two queries"):
            select_additions(rows, fraction="0.01")


class ValidationExpansionTests(unittest.TestCase):
    SOURCES = ("locomo", "split_longmemeval_s")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_data = self.root / "input"
        self.old_prepared = self.root / "prepared"
        self.new_data = self.root / "expanded_input"
        self.new_prepared = self.root / "expanded_prepared"
        self.config = {
            "data": {"layout": "by_source", "input_root": str(self.old_data),
                     "prepared_root": str(self.old_prepared), "sources": list(self.SOURCES),
                     "splits": ["train", "validation"]},
            "writer": {"lookback_sessions": 3, "content_max_tokens": 300},
            "reward": {"formula": "judge_cost_v2"},
        }
        self.writer_identity = {"base_url": "http://fixture/v1", "model": "fixture",
                                "model_revision": "fixture-version"}

    def make_fixture(self, *, single_only=False):
        self.old_data.mkdir()
        self.tables = {}
        manifest = {"sources": []}
        for source in self.SOURCES:
            directory = self.old_data / source
            directory.mkdir()
            files = {}
            for split, size in (("train", 8), ("validation", 2), ("test", 2)):
                rows = []
                for i in range(size):
                    single = single_only or i % 2 == 0
                    rows.append({
                        "question": f"{source} {split} query {i}",
                        "question_id": f"{source}-{split}-{i}",
                        "question_type": "single-session-user" if single else "multi-session",
                        "question_date": f"2023/06/{i + 1:02d} (Thu) 09:00",
                        "answer": f"answer {i}",
                        "answer_session_ids": ["answer_s0"] if single else ["s0", "s1"],
                        "haystack_session_ids": ["s0", "s1"],
                        "haystack_dates": ["2023/01/01", "2023/02/01"],
                        "haystack_sessions": [
                            [{"role": "user", "content": f"{source} history {i % 2} first 中文"}],
                            [{"role": "assistant", "content": "second fact"}]],
                        "untouched_metadata": {"nullable": None, "tags": ["keep", str(i)]},
                    })
                table = pa.Table.from_pylist(rows).replace_schema_metadata({b"fixture": b"preserve-schema"})
                path = directory / f"{split}.parquet"
                pq.write_table(table, path)
                # Compare against the materialized parent schema (Parquet may
                # canonicalize nested list child names on the first write).
                self.tables[source, split] = pq.read_table(path)
                files[split] = {"rows": size, "sha256": file_digest(path)}
            manifest["sources"].append({"source": source, "output_files": files})
        atomic_json(self.old_data / "manifest.json", manifest)
        for source in self.SOURCES:
            store = Store(self.old_prepared / source)
            builder = DirectoryBuilder(store, SimpleNamespace(identity=self.writer_identity),
                                       self.config["writer"], None)
            self.config["data"]["frozen_directory_version"] = builder.version
            for split in ("train", "validation"):
                for i, row in enumerate(self.tables[source, split].to_pylist()):
                    records = clean_history(row)
                    memory_id = store.add_memory(records)
                    store.add_example(adapt_row(row, source, split, i, memory_id=memory_id))
                    entries = [{"id": r["id"],
                                "task_session": {"task_id": None, "session_id": r["source_session_id"]},
                                "actors_source": {"actors": ["user"], "sources": []},
                                "time_order": {"time": r["time"], "order": r["order"]},
                                "content": f"Frozen directory summary {r['order']}"}
                               for r in records]
                    store.put_directory(memory_id, builder.version, entries)
            store.set_metadata("directory_version", builder.version)
            store.set_metadata("directory_service_identity", self.writer_identity)
            store.set_metadata("directory_empty_pending", 0)
            store.set_metadata("prepare_identity", {
                "input_manifest_sha256": file_digest(self.old_data / "manifest.json"),
                "sources": [source], "splits": ["train", "validation"],
                "limit_per_source_split": None, "adapter_version": ADAPTER_VERSION,
                "storage_format": Store.FORMAT})
            store.set_metadata("prepare_complete", True)
            store.close()

    def test_expansion_preserves_rows_and_frozen_memories_without_touching_source(self):
        self.make_fixture()
        before_data, before_prepared = tree_hashes(self.old_data), tree_hashes(self.old_prepared)
        report = expand_validation(self.config, self.new_data, self.new_prepared, fraction="0.25", seed=42)
        current = deepcopy(self.config)
        current["data"].update(input_root=str(self.new_data), prepared_root=str(self.new_prepared))
        for entry in report["sources"]:
            source = entry["source"]
            added = entry["added_parent_train_rows"]
            self.assertEqual(entry["added_queries"], 2)
            self.assertEqual(entry["added_strata"], {"single": 1, "multi": 1})
            self.assertEqual((entry["train_rows"], entry["validation_rows"]), (6, 4))
            old_train = self.tables[source, "train"]
            old_val = self.tables[source, "validation"]
            new_train = pq.read_table(self.new_data / source / "train.parquet")
            new_val = pq.read_table(self.new_data / source / "validation.parquet")
            keep = [i for i in range(8) if i not in added]
            expected_train = old_train.take(pa.array(keep, type=pa.int64()))
            expected_val = pa.concat_tables([old_val, old_train.take(pa.array(added, type=pa.int64()))])
            self.assertTrue(new_train.equals(expected_train, check_metadata=True))
            self.assertTrue(new_val.equals(expected_val, check_metadata=True))
            self.assertTrue(new_val.slice(0, len(old_val)).equals(old_val, check_metadata=True))
            self.assertTrue(set(new_train.column("question").to_pylist()).isdisjoint(
                new_val.column("question").to_pylist()))
            self.assertEqual((self.new_data / source / "test.parquet").read_bytes(),
                             (self.old_data / source / "test.parquet").read_bytes())
            for name in ("memory_records.jsonl", "directories.jsonl", "directory_manifest.json"):
                old, new = self.old_prepared / source / name, self.new_prepared / source / name
                self.assertEqual(file_digest(old), file_digest(new))
                self.assertNotEqual(old.stat().st_ino, new.stat().st_ino)

            old_examples = {split: [json.loads(line) for line in
                (self.old_prepared / source / f"{split}_examples.jsonl").read_text().splitlines()]
                for split in ("train", "validation")}
            child = dict(dataset_configs(current))[source]
            store = Store(self.new_prepared / source)
            try:
                validate_directories(child, store, ["train", "validation"])
                self.assertEqual(store.get_metadata("prepare_identity")["input_manifest_sha256"],
                                 file_digest(self.new_data / "manifest.json"))
                self.assertEqual(store.get_metadata("directory_service_identity"), self.writer_identity)
                for split, table, origins in (
                    ("train", new_train, [("train", i) for i in keep]),
                    ("validation", new_val, [("validation", 0), ("validation", 1)]
                     + [("train", i) for i in added]),
                ):
                    examples = task_examples(child, store, split)
                    self.assertEqual(len(examples), table.num_rows)
                    for i, (example, raw, (old_split, old_index)) in enumerate(
                            zip(examples, table.to_pylist(), origins)):
                        memory_id = old_examples[old_split][old_index]["memory_id"]
                        self.assertEqual(asdict(example), asdict(adapt_row(
                            raw, source, split, i, memory_id=memory_id)))
            finally:
                store.close()
        self.assertEqual(tree_hashes(self.old_data), before_data)
        self.assertEqual(tree_hashes(self.old_prepared), before_prepared)

    def test_existing_destination_is_rejected_without_modifying_anything(self):
        self.make_fixture()
        self.new_data.mkdir()
        sentinel = self.new_data / "keep.txt"
        sentinel.write_text("existing output")
        before = tree_hashes(self.root)
        with self.assertRaisesRegex(ValueError, "fresh output roots"):
            expand_validation(self.config, self.new_data, self.new_prepared, fraction="0.25")
        self.assertEqual(tree_hashes(self.root), before)
        self.assertFalse(self.new_prepared.exists())

    def test_target_expansion_preserves_existing_validation_and_overrides_one_source(self):
        self.make_fixture()
        report = expand_validation(self.config, self.new_data, self.new_prepared,
                                   fraction="0.5", mode="target", target_counts={"locomo": 6})
        counts = {entry["source"]: entry for entry in report["sources"]}
        self.assertEqual((counts["locomo"]["train_rows"], counts["locomo"]["validation_rows"]), (4, 6))
        self.assertEqual((counts["split_longmemeval_s"]["train_rows"],
                          counts["split_longmemeval_s"]["validation_rows"]), (6, 4))
        for source in self.SOURCES:
            original = self.tables[source, "validation"]
            new = pq.read_table(self.new_data / source / "validation.parquet")
            self.assertTrue(new.slice(0, original.num_rows).equals(original, check_metadata=True))
            train = pq.read_table(self.new_data / source / "train.parquet", columns=["question"])
            self.assertTrue(set(train.column("question").to_pylist()).isdisjoint(new.column("question").to_pylist()))

    def test_insufficient_stratum_fails_before_publishing_outputs(self):
        self.make_fixture(single_only=True)
        before = tree_hashes(self.root)
        with self.assertRaisesRegex(ValueError, "Not enough eligible multi"):
            expand_validation(self.config, self.new_data, self.new_prepared, fraction="0.25")
        self.assertFalse(self.new_data.exists())
        self.assertFalse(self.new_prepared.exists())
        self.assertEqual(tree_hashes(self.root), before)


if __name__ == "__main__":
    unittest.main()
