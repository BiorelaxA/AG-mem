import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.build_deepseek_train_range import train_banks
from agmem.core import file_digest
from agmem.hierarchical.adapters import iter_banks


class TrainRangeTests(unittest.TestCase):
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
