from copy import deepcopy
import json
import unittest

import pyarrow.parquet as pq

from agmem.core import digest, file_digest
from agmem.longmemeval_fixed_split import plan_fixed_assignments, resplit_fixed_test
from agmem.longmemeval_resplit import SOURCES, SPLITS
from tests import test_longmemeval_resplit as fixtures


class FixedTestSplitTests(unittest.TestCase):
    setUp = fixtures.LongMemEvalResplitTests.setUp
    fixture = fixtures.LongMemEvalResplitTests.fixture
    def test_fixed_assignment_keeps_test_and_validation_and_aligns_variants(self):
        assignment, report = plan_fixed_assignments(self.rows, validation_rows=5)
        self.assertEqual(report['counts_after'], {'train': 30, 'validation': 5, 'test': 15})
        self.assertEqual(report['moved_from_train_to_validation'], 4)
        for source in SOURCES:
            for split in ('test', 'validation'):
                self.assertTrue(all(assignment[row['question_id']] == split for row in self.rows[source][split]))
        reversed_rows = {source: {split: list(reversed(rows)) for split, rows in splits.items()}
                         for source, splits in self.rows.items()}
        self.assertEqual(plan_fixed_assignments(reversed_rows, validation_rows=5), (assignment, report))
        self.assertEqual(len(set(report['added_validation_question_ids'])), 4)

    def test_test_bytes_all_fields_mapping_and_safe_idempotence(self):
        self.fixture()
        original = {p: file_digest(p) for p in self.input.rglob('*') if p.is_file()}
        report = resplit_fixed_test(self.input, self.output, validation_rows=5)
        mappings = [json.loads(line) for line in (self.output / 'row_mapping.jsonl').read_text().splitlines()]
        for source in SOURCES:
            source_seen = set()
            for split in SPLITS:
                output = self.output / source / f'{split}.parquet'
                rows = pq.read_table(output)
                refs = [r for r in mappings if r['source'] == source and r['new_split'] == split]
                self.assertEqual(rows.num_rows, report['allocation']['counts_after'][split])
                self.assertEqual(rows.schema.metadata, {b'fixture': b'unaltered'})
                for i, (row, ref) in enumerate(zip(rows.to_pylist(), refs)):
                    self.assertEqual(row, self.rows[source][ref['old_split']][ref['old_row_index']])
                    self.assertEqual(ref['new_row_index'], i)
                    self.assertEqual(ref['new_sample_id'], f'{source}/{split}/{i:06d}')
                    self.assertEqual(ref['new_bank_id'], 'bank_' + digest([source, split, file_digest(output), i])[:24])
                    source_seen.add((ref['old_split'], ref['old_row_index']))
                if split == 'validation':
                    n = len(self.rows[source]['validation'])
                    self.assertEqual(rows.to_pylist()[:n], self.rows[source]['validation'])
                if split == 'test':
                    self.assertEqual(output.read_bytes(), (self.input / source / 'test.parquet').read_bytes())
                    self.assertTrue(all(r['old_bank_id'] == r['new_bank_id'] for r in refs))
            self.assertEqual(len(source_seen), 50)
        self.assertEqual({p: file_digest(p) for p in original}, original)
        before = {p: (file_digest(p), p.stat().st_mtime_ns) for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(resplit_fixed_test(self.input, self.output, validation_rows=5), report)
        self.assertEqual({p: (file_digest(p), p.stat().st_mtime_ns) for p in before}, before)
        with self.assertRaisesRegex(ValueError, 'differs'):
            resplit_fixed_test(self.input, self.output, validation_rows=5, seed=43)
        path = self.output / SOURCES[0] / 'train.parquet'
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'changed'):
            resplit_fixed_test(self.input, self.output, validation_rows=5)

    def test_fixed_dry_run_and_refuse_mismatched_original_membership(self):
        self.fixture()
        report = resplit_fixed_test(self.input, self.output, validation_rows=5, dry_run=True)
        self.assertEqual(report['status'], 'planned')
        self.assertFalse(self.output.exists())
        rows = deepcopy(self.rows)
        rows[SOURCES[1]]['train'][0], rows[SOURCES[1]]['test'][0] = (
            rows[SOURCES[1]]['test'][0], rows[SOURCES[1]]['train'][0])
        with self.assertRaisesRegex(ValueError, 'memberships'):
            plan_fixed_assignments(rows, validation_rows=5)
        rows = deepcopy(self.rows)
        rows[SOURCES[0]]['test'][0]['question'] = '  QUESTION 0  '
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            plan_fixed_assignments(rows, validation_rows=5)

    def test_existing_mapping_tampering_is_rejected(self):
        self.fixture()
        resplit_fixed_test(self.input, self.output, validation_rows=5)
        path = self.output / 'row_mapping.jsonl'
        path.write_text(path.read_text().replace('q000', 'q999'))
        with self.assertRaisesRegex(ValueError, 'mapping'):
            resplit_fixed_test(self.input, self.output, validation_rows=5)


if __name__ == '__main__':
    unittest.main()
