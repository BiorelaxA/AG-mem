import tempfile
import unittest
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.build_deepseek_train_range import missing_parents, train_banks
from agmem.core import file_digest
from agmem.hierarchical.adapters import iter_banks


class TrainRangeTests(unittest.TestCase):
    def test_backfill_keeps_fallbacks_and_original_parent_ids_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.parquet"
            rows = [{"question_id": str(i), "haystack_session_ids": ["a","b","c"],
                     "haystack_dates": ["2022-01-01"] * 3,
                     "haystack_sessions": [[{"role": "user", "content": str(k)}] for k in range(3)]}
                    for i in range(3)]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            sha = file_digest(path)
            banks = list(train_banks(path, sha, 1, 3, count_tokens=len))
            base, target = Path(tmp)/"base", Path(tmp)/"output"
            spec = {"name":"split_longmemeval_m", "trajectory_range_1based":[1,3], "exclude_committed_root":base}
            for root, statuses in ((base, [(0,1,"complete"), (0,2,"fallback"), (0,3,"complete"),
                                           (1,1,"failed"), (1,2,"fallback"), (1,3,"pending")]),
                                   (target, [(1,1,"fallback")])):
                root.mkdir()
                db = sqlite3.connect(root/"build_journal.sqlite3")
                db.executescript("CREATE TABLE metadata(key,value); CREATE TABLE sessions(bank_id,parent_id,status,directory);")
                db.execute("INSERT INTO metadata VALUES ('identity',?)", (json.dumps({
                    "protocol":"merged_directory_snapshot.v1","dataset":spec["name"],"split":"train","input_sha256":sha}),))
                for i, n, status in statuses:
                    db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (banks[i]["bank_id"], f"p{n:04d}",status,
                               None if status in {"failed","pending"} else "{}"))
                db.commit()
                db.close()
            wanted = missing_parents(spec, sha, [range(3)] * 3, target)
            self.assertEqual(wanted, {1:{"p0003"}, 2:{"p0001","p0002","p0003"}})
            partial, untouched = train_banks(path, sha, 1, 3, count_tokens=len, wanted=wanted)
            self.assertEqual(partial["sessions"], banks[1]["sessions"][2:])
            self.assertEqual({s["parent_id"] for s in partial["source_map"]}, {"p0003"})
            self.assertEqual(untouched, banks[2])

    def test_cross_row_group_selection_matches_unmodified_original_ids_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.parquet"
            rows = [{"question_id": str(i), "haystack_session_ids": [str(i)],
                     "haystack_dates": ["2022-01-01"],
                     "haystack_sessions": [[{"role": "user", "content": str(i)}]]} for i in range(9)]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
            spec = {"name": "split_longmemeval_m", "format": "longmemeval_parquet",
                    "input_files": {"train": str(path)}}
            original = list(iter_banks(spec, ["train"], count_tokens=len))
            for first, last in ((3, 7), (4, 4), (5, 9)):
                with self.subTest(first=first, last=last):
                    actual = list(train_banks(path, file_digest(path), first, last, count_tokens=len))
                    self.assertEqual(actual, original[first-1:last])
