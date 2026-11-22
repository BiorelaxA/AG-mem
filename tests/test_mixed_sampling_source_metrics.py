"""Mixed-source telemetry distinguishes sampled, eligible, and updated groups."""
from collections import Counter
from copy import deepcopy
from itertools import islice
import random
import unittest

from agmem.core import Example
from agmem.dapo_sampling import DynamicGroupSampler
from agmem.monitoring import write_sampling_metrics
from agmem.rewards import score_trajectory
from tests.test_dapo_sampling import settings, group_for


REWARD = {'formula': 'f1_cost_v1', 'read_call_cost': .05,
          'format_penalty': .5, 'format_penalty_mode': 'per_id'}


def config_for(**overrides):
    config = settings(filter_metric='nonzero_advantage', **overrides)
    config['reward'] = dict(REWARD)
    return config


def f1_group(example, predictions, calls):
    group = group_for(example, rewards=[0] * len(calls), judges=[None] * len(calls))
    for row, prediction, count in zip(group, predictions, calls):
        row.prediction, row.n_read_calls = prediction, count
        score_trajectory(example, row, REWARD, max_ids=1)
    return group


def mixed_examples(locomo=4, lmes=4):
    return [Example(f'{name}/train/{i:06d}', name, 'train', f'{name}-memory-{i}', f'question {i}',
                    gold_answers=['gold answer'])
            for name, count in (('locomo', locomo), ('split_longmemeval_s', lmes)) for i in range(count)]


class CaptureWriter:
    def __init__(self):
        self.values = {}
        self.flushed = False

    def add_scalar(self, name, value, global_step):
        self.values[name] = value, global_step

    def flush(self):
        self.flushed = True


class MixedSamplingSourceMetricsTests(unittest.TestCase):
    def test_every_epoch_shuffles_whole_combined_list_without_changing_groups(self):
        examples = mixed_examples()
        config = config_for(epochs=2, group_size=8, target_groups=8, candidate_groups=32)
        sampler = DynamicGroupSampler(config, examples)

        def collect(example, index):
            # A source-dependent constant cost shift must not couple advantages.
            calls = 2 if example.source == 'locomo' else 8
            return f1_group(example, ['wrong', 'gold answer'] * 4, [calls] * 8)

        batches = []
        for epoch in range(2):
            groups, report = sampler.collect(collect, 7)
            batches.append(groups)
            expected = list(examples)
            random.Random(config['seed'] + epoch).shuffle(expected)
            self.assertEqual([g[0].sample_id for g in groups], [e.sample_id for e in expected])
            self.assertNotEqual([g[0].sample_id for g in groups], [e.sample_id for e in examples])
            self.assertEqual(report['source_selected'], {'locomo': 4, 'split_longmemeval_s': 4})
            self.assertTrue(report['source_selection_ready'])
            for group in groups:
                self.assertEqual(len(group), 8)
                self.assertEqual(len({(r.sample_id, r.memory_id) for r in group}), 1)
                self.assertAlmostEqual(sum(r.advantage for r in group), 0, places=10)
                for row, reference in zip(group, groups[0]):
                    self.assertAlmostEqual(row.advantage, reference.advantage, places=10)
        self.assertNotEqual([g[0].sample_id for g in batches[0]], [g[0].sample_id for g in batches[1]])

    def test_drained_accepted_sibling_is_not_reported_as_selected(self):
        sampler = DynamicGroupSampler(config_for(target_groups=2, candidate_groups=32), mixed_examples())
        recorded = []

        def windows(candidates):
            while True:
                current = list(islice(candidates, 4))
                if not current:
                    return
                for example, index in current:
                    predictions = ['wrong', 'wrong'] if index == 0 else ['wrong', 'gold answer']
                    yield example, index, f1_group(example, predictions, [1, 2])

        groups, report = sampler.collect(None, 7, collect_candidates=windows,
                                        record_candidate=lambda _, row: recorded.append(deepcopy(row)))
        self.assertEqual([r['selected'] for r in recorded], [False, True, True, False])
        self.assertEqual(sum(report['source_candidates'].values()), 4)
        self.assertEqual(sum(report['source_accepted'].values()), 3)
        self.assertEqual(sum(report['source_selected'].values()), len(groups))
        self.assertEqual(len(groups), 2)
        returned = Counter(g[0].sample_id.split('/')[0] for g in groups)
        for source, n in report['source_candidates'].items():
            rows = [r for r in recorded if r['source'] == source]
            self.assertEqual(report['source_selected'][source], returned[source])
            metrics = report['source_metrics'][source]
            self.assertEqual(metrics['candidate_groups'], n)
            self.assertEqual(metrics['acceptance_rate'], sum(r['accepted'] for r in rows) / n)
            self.assertAlmostEqual(metrics['answer_f1'], sum(r['answer_f1'] for r in rows) / n)
            self.assertAlmostEqual(metrics['filtered_f1_all_zero_rate'],
                                   sum(r['filtered_f1_all_zero'] for r in rows) / n)

    def test_discarded_dataset_tail_has_zero_selected_counts_even_in_saved_round(self):
        sampler = DynamicGroupSampler(config_for(target_groups=4, candidate_groups=8), mixed_examples(2, 1))
        rounds = []
        groups, report = sampler.collect(lambda e, _: f1_group(e, ['wrong', 'gold answer'], [1, 2]), 7,
                                        record_round=lambda r: rounds.append(deepcopy(r)))
        self.assertEqual(groups, [])
        self.assertEqual(report['source_accepted'], {'locomo': 2, 'split_longmemeval_s': 1})
        self.assertEqual(report['source_retained'], report['source_accepted'])
        self.assertEqual(report['source_dropped_tail'], report['source_retained'])
        self.assertEqual(report['source_selected'], {'locomo': 0, 'split_longmemeval_s': 0})
        self.assertFalse(rounds[-1]['source_selection_ready'])
        self.assertEqual(rounds[-1]['source_selected'], report['source_selected'])
        self.assertTrue(all('selected_answer_f1' not in row for row in report['source_metrics'].values()))

    def test_exhausted_sampling_budget_does_not_claim_partial_groups_were_used(self):
        sampler = DynamicGroupSampler(config_for(target_groups=4, candidate_groups=2, max_num_gen_batches=1),
                                      mixed_examples())
        rounds = []
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            sampler.collect(lambda e, _: f1_group(e, ['wrong', 'gold answer'], [1, 2]), 7,
                            record_round=lambda r: rounds.append(deepcopy(r)))
        self.assertEqual(sum(rounds[0]['source_accepted'].values()), 2)
        self.assertEqual(sum(rounds[0]['source_retained'].values()), 2)
        self.assertEqual(sum(rounds[0]['source_selected'].values()), 0)

    def test_candidate_and_selected_source_means_are_separate(self):
        sampler = DynamicGroupSampler(config_for(target_groups=2, candidate_groups=8), mixed_examples(1, 1))
        groups, report = sampler.collect(lambda e, _: f1_group(e,
            ['gold answer', 'gold answer'] if e.source == 'locomo' else ['wrong', 'gold answer'], [1, 2]), 7)
        self.assertEqual(len(groups), 2)
        self.assertEqual(report['source_metrics']['locomo']['answer_f1'], 1)
        self.assertEqual(report['source_metrics']['split_longmemeval_s']['answer_f1'], .5)
        for source in report['source_metrics']:
            self.assertEqual(report['source_metrics'][source]['answer_f1'],
                             report['source_metrics'][source]['selected_answer_f1'])

    def test_tensorboard_writes_source_counts_and_rates_and_accepts_legacy_reports(self):
        writer = CaptureWriter()
        report = {'candidate_index': 40, 'source_candidates': {'locomo': 6, 'split_longmemeval_s': 2},
                  'source_accepted': {'locomo': 4, 'split_longmemeval_s': 1},
                  'source_selected': {'locomo': 3, 'split_longmemeval_s': 1},
                  'source_retained': {'locomo': 3, 'split_longmemeval_s': 1},
                  'source_metrics': {'locomo': {'acceptance_rate': 2/3, 'answer_f1': .5},
                                     'split_longmemeval_s': {'acceptance_rate': .5, 'answer_f1': .2}}}
        write_sampling_metrics(writer, report)
        self.assertEqual(writer.values['sampling/source_selected/locomo'], (3, 40))
        self.assertEqual(writer.values['sampling/source_accepted/split_longmemeval_s'], (1, 40))
        self.assertEqual(writer.values['sampling/by_source/split_longmemeval_s/answer_f1'], (.2, 40))
        self.assertEqual(writer.values['sampling/by_source/locomo/acceptance_rate'], (2/3, 40))
        self.assertTrue(writer.flushed)
        legacy = CaptureWriter()
        write_sampling_metrics(legacy, {'candidate_index': 4, 'source_candidates': {'locomo': 4}})
        self.assertEqual(legacy.values['sampling/source_candidates/locomo'], (4, 4))
        self.assertFalse(any('source_selected' in key for key in legacy.values))


if __name__ == '__main__':
    unittest.main()
