from copy import deepcopy
from dataclasses import asdict
import json
import unittest

import pyarrow.parquet as pq

from agmem.core import file_digest
from agmem.data import adapt_row, task_examples
from agmem.pipeline import dataset_configs, validate_directories
from agmem.query_resplit import resplit_locomo, split_queries
from agmem.store import Store
from tests import test_validation_split as fixtures


class QueryResplitTests(unittest.TestCase):
    setUp = fixtures.ValidationExpansionTests.setUp
    make_fixture = fixtures.ValidationExpansionTests.make_fixture
    SOURCES = fixtures.ValidationExpansionTests.SOURCES

    def test_query_sampling_ignores_duplicates_and_old_partition_order(self):
        rows = [{"question": f"query-{i}"} for i in range(11)]
        chosen = split_queries(rows)
        self.assertEqual(len(chosen), 2)  # ceil(11 * 0.1), not row count.
        repeated = rows + [rows[0]] * 100
        self.assertEqual(chosen, split_queries(list(reversed(repeated))))
        for fraction in (0, 1, "NaN", "Infinity"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_queries(rows, fraction=fraction)
        with self.assertRaises(ValueError):
            split_queries([{"question": "only query"}] * 10)

    def test_resplit_preserves_every_row_prepared_memory_and_other_sources(self):
        self.make_fixture()
        before_data = fixtures.tree_hashes(self.old_data)
        before_prepared = fixtures.tree_hashes(self.old_prepared)
        report = resplit_locomo(self.config, self.new_data, self.new_prepared)
        counts = {entry["source"]: entry for entry in report["sources"]}
        self.assertEqual((counts["locomo"]["train_queries"], counts["locomo"]["validation_queries"]), (9, 1))
        mapping = [json.loads(line) for line in (self.new_data / "row_mapping.jsonl").read_text().splitlines()]
        current = deepcopy(self.config)
        current["data"].update(input_root=str(self.new_data), prepared_root=str(self.new_prepared))
        for source, child in dataset_configs(current):
            store = Store(self.new_prepared / source)
            validate_directories(child, store, ["train", "validation"])
            self.assertEqual(store.get_metadata("prepare_identity")["input_manifest_sha256"],
                             file_digest(self.new_data / "manifest.json"))
            origins = []
            query_sets = {}
            for split in ("train", "validation"):
                table = pq.read_table(self.new_data / source / f"{split}.parquet")
                old_examples = {old: Store(self.old_prepared / source).examples(old) for old in ("train", "validation")}
                selected = [m for m in mapping if m["source"] == source and m["split"] == split]
                examples = task_examples(child, store, split)
                self.assertEqual(len(examples), table.num_rows)
                for index, (raw, example, origin) in enumerate(zip(table.to_pylist(), examples, selected)):
                    old, i = origin["parent_split"], origin["parent_row_index"]
                    origins.append((old, i))
                    self.assertEqual(raw, self.tables[source, old].to_pylist()[i])
                    expected = adapt_row(raw, source, split, index, memory_id=old_examples[old][i].memory_id)
                    self.assertEqual(asdict(example), asdict(expected))
                query_sets[split] = set(table.column("question").to_pylist())
            self.assertFalse(query_sets["train"] & query_sets["validation"])
            self.assertCountEqual(origins, [(split, i) for split, n in (("train", 8), ("validation", 2)) for i in range(n)])
            for name in ("memory_records.jsonl", "directories.jsonl", "directory_manifest.json"):
                self.assertEqual(file_digest(self.new_prepared / source / name),
                                 file_digest(self.old_prepared / source / name))
            for split in (("test",) if source == "locomo" else ("train", "validation", "test")):
                self.assertEqual(file_digest(self.new_data / source / f"{split}.parquet"),
                                 file_digest(self.old_data / source / f"{split}.parquet"))
        self.assertEqual(before_data, fixtures.tree_hashes(self.old_data))
        self.assertEqual(before_prepared, fixtures.tree_hashes(self.old_prepared))
        with self.assertRaisesRegex(ValueError, "fresh output roots"):
            resplit_locomo(self.config, self.new_data, self.new_prepared)


if __name__ == "__main__":
    unittest.main()
