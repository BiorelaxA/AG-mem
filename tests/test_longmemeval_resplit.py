from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import digest, file_digest
from agmem.longmemeval_resplit import SOURCES, SPLITS, plan_assignments, resplit_longmemeval


class LongMemEvalResplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.input = self.home / "input"
        self.output = self.home / "output"
        self.rows = {}
        for source in SOURCES:
            self.rows[source] = {}
            for split, begin, end in [("train", 0, 34), ("validation", 34, 35), ("test", 35, 50)]:
                rows = [{"question_id": f"q{i:03d}", "question": f"Question {i}",
                         "question_type": ["a", "b", "c"][i % 3],
                         "answer": f"Gold {i}", "haystack_sessions": json.dumps([[{"content": f"raw {i}"}]]),
                         "haystack_session_ids": [f"session_{i}"], "other": {"number": i, "flag": True}}
                        for i in range(begin, end)]
                self.rows[source][split] = rows if source == SOURCES[0] else list(reversed(rows))

    def fixture(self):
        entries = []
        for source in SOURCES:
            (self.input / source).mkdir(parents=True)
            files = {}
            for split in SPLITS:
                p = self.input / source / f"{split}.parquet"
                table = pa.Table.from_pylist(self.rows[source][split]).replace_schema_metadata({b"fixture": b"unaltered"})
                pq.write_table(table, p)
                files[split] = {"sha256": file_digest(p)}
            entries.append({"source": source, "output_files": files})
        (self.input / "manifest.json").write_text(json.dumps({"sources": entries}))

    def test_assignment_is_shared_deterministic_and_preserves_original_membership(self):
        selected, report = plan_assignments(self.rows)
        self.assertEqual(report["counts_after"], {"train": 40, "validation": 5, "test": 5})
        self.assertEqual(report["moved_from_test"], {"train": 6, "validation": 4})
        for source in SOURCES:
            for split in ("train", "validation"):
                self.assertTrue(all(selected[row["question_id"]] == split for row in self.rows[source][split]))
        reversed_rows = {s: {split: list(reversed(rows)) for split, rows in data.items()}
                         for s, data in self.rows.items()}
        self.assertEqual(plan_assignments(reversed_rows), (selected, report))

    def test_duplicate_queries_and_cross_variant_mismatch_are_rejected(self):
        cases = []
        rows = deepcopy(self.rows)
        rows[SOURCES[0]]["test"][0]["question"] = rows[SOURCES[0]]["train"][0]["question"]
        cases.append(rows)
        rows = deepcopy(self.rows)
        rows[SOURCES[1]]["test"][0]["question_type"] = "different"
        cases.append(rows)
        rows = deepcopy(self.rows)
        rows[SOURCES[1]]["train"][0], rows[SOURCES[1]]["test"][0] = (
            rows[SOURCES[1]]["test"][0], rows[SOURCES[1]]["train"][0])
        cases.append(rows)
        for rows in cases:
            with self.subTest(), self.assertRaises(ValueError):
                plan_assignments(rows)

    def test_full_write_preserves_all_fields_and_provenance(self):
        self.fixture()
        originals = {p: file_digest(p) for p in self.input.rglob("*") if p.is_file()}
        report = resplit_longmemeval(self.input, self.output)
        self.assertEqual(report["status"], "complete")
        mapping = [json.loads(line) for line in (self.output / "row_mapping.jsonl").read_text().splitlines()]
        assignments = {}
        for source in SOURCES:
            used = set()
            for split, count in [("train", 40), ("validation", 5), ("test", 5)]:
                p = self.output / source / f"{split}.parquet"
                got = pq.read_table(p)
                self.assertEqual(got.num_rows, count)
                self.assertEqual(got.schema.metadata, {b"fixture": b"unaltered"})
                selected = [r for r in mapping if r["source"] == source and r["new_split"] == split]
                for index, (row, ref) in enumerate(zip(got.to_pylist(), selected)):
                    expected = self.rows[source][ref["old_split"]][ref["old_row_index"]]
                    self.assertEqual(row, expected)
                    self.assertEqual(ref["new_row_index"], index)
                    self.assertEqual(ref["new_bank_id"], "bank_" + digest([source, split, file_digest(p), index])[:24])
                    oldp = self.input / source / (ref["old_split"] + ".parquet")
                    self.assertEqual(ref["old_bank_id"], "bank_" + digest([
                        source, ref["old_split"], file_digest(oldp), ref["old_row_index"]])[:24])
                    used.add((ref["old_split"], ref["old_row_index"]))
                    self.assertEqual(assignments.setdefault(row["question_id"], split), split)
                if split != "test":
                    self.assertEqual(got.to_pylist()[:len(self.rows[source][split])], self.rows[source][split])
            self.assertEqual(len(used), 50)
        self.assertEqual({p: file_digest(p) for p in originals}, originals)
        with self.assertRaisesRegex(ValueError, "fresh output root"):
            resplit_longmemeval(self.input, self.output)

    def test_dry_run_has_no_output_and_rejects_changed_input(self):
        self.fixture()
        report = resplit_longmemeval(self.input, self.output, dry_run=True)
        self.assertEqual(report["status"], "planned")
        self.assertFalse(self.output.exists())
        p = self.input / SOURCES[0] / "test.parquet"
        p.write_bytes(p.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "differs from its manifest"):
            resplit_longmemeval(self.input, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
