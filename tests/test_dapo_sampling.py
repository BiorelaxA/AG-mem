from copy import deepcopy
from itertools import islice
import math
import unittest
from unittest.mock import Mock

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.dapo_sampling import DynamicGroupSampler, assess_group, validate_dapo_config


def settings(**overrides):
    config = {
        "seed": 23,
        "training": {"algorithm": "dapo", "group_size": 2, "candidate_groups": 3,
                     "target_groups": 2, "epochs": 1, "max_num_gen_batches": 4,
                     "mini_batch_groups": 1, "micro_batch_rollouts": 1, "ppo_epochs": 1,
                     "kl_coefficient": 0, "filter_metric": "reward", "filter_min_std": 1e-6,
                     "advantage_epsilon": 1e-6, "advantage_std_floor": 0,
                     "overlong_filter": True},
        "reward": {"formula": "judge_gated_v1", "likelihood": {"mode": "off"}},
        "reader": {"rollout_temperature": 1.0},
    }
    config["training"].update(overrides)
    return config


def judge_cost_settings(**overrides):
    config = settings(filter_metric="judge", **overrides)
    config["reward"]["formula"] = "judge_cost_v2"
    return config


def examples(count):
    return [Example(f"source{i % 2}/train/{i}", f"source{i % 2}", "train", f"memory{i}", f"q{i}")
            for i in range(count)]


def group_for(example, rewards=(-1.0, 1.0), judges=None, auxiliary=None, version=7, truncated=()):
    judges = [0.0 if reward < 0 else 1.0 for reward in rewards] if judges is None else judges
    auxiliary = [0.0] * len(rewards) if auxiliary is None else auxiliary
    group = []
    for i, (reward, judge, extra) in enumerate(zip(rewards, judges, auxiliary)):
        completion = Completion("action", [1, 2, 3], [4, 5], [-1.25, -2.5], truncated=i in truncated)
        row = Trajectory(example.sample_id, i, example.memory_id, policy_version=version,
                         decisions=[Decision([], completion)], reward=reward, judge_score=judge,
                         auxiliary_reward=extra, evidence=[{"id": str(i)}], prediction=str(i))
        group.append(row)
    return group


class DynamicSamplingTests(unittest.TestCase):
    def test_32_candidates_refill_until_16_complete_eight_rollout_groups(self):
        config = settings(group_size=8, candidate_groups=32, target_groups=16)
        sampler = DynamicGroupSampler(config, examples(80))
        candidates, rounds = [], []
        def collect(example, candidate_id):
            # Only one quarter of candidate queries has a useful signal.
            rewards = [-1.0, 1.0] * 4 if candidate_id % 4 == 0 else [-1.0] * 8
            return group_for(example, rewards)
        groups, report = sampler.collect(collect, 7, record_candidate=lambda group, result: candidates.append(result),
                                         record_round=lambda result: rounds.append(deepcopy(result)))
        self.assertEqual(len(groups), 16)
        self.assertTrue(all(len(group) == 8 for group in groups))
        self.assertEqual(report["generation_batches"], 2)
        self.assertEqual(report["candidate_groups"], 61)
        self.assertEqual([row["selected_groups"] for row in rounds], [8, 16])
        self.assertEqual(sum(row["selected"] for row in candidates), 16)
        self.assertEqual(report["accepted_rate"], 16 / 61)
        self.assertEqual(report["judge_all_wrong_rate"], 45 / 61)
        self.assertEqual(sum(report["source_candidates"].values()), 61)
        self.assertEqual(sampler.state_dict()["candidate_index"], 61)
        self.assertEqual(sampler.state_dict()["batch_index"], 1)
        self.assertTrue(all(row.policy_version == 7 for group in groups for row in group))

    def test_serial_sampling_stops_at_target_without_consuming_next_candidate(self):
        sampler = DynamicGroupSampler(settings(), examples(4))
        recorded = []
        groups, report = sampler.collect(lambda example, index: group_for(example), 7,
                                         record_candidate=lambda group, result: recorded.append(result))
        self.assertEqual(len(groups), 2)
        self.assertEqual(report["accepted_groups"], 2)
        self.assertEqual([row["selected"] for row in recorded], [True, True])
        self.assertEqual([row["candidate_id"] for row in recorded], [0, 1])
        self.assertEqual(sampler.next_index, 2)
        self.assertEqual({row["batch_id"] for row in recorded}, {"batch-00000000"})

    def test_budget_exhaustion_is_explicit_and_does_not_return_partial_batch(self):
        config = settings(max_num_gen_batches=2)
        sampler = DynamicGroupSampler(config, examples(20))
        rounds = []
        def collect(example, index):
            return group_for(example, (-1.0, 1.0) if index == 0 else (-1.0, -1.0))
        with self.assertRaisesRegex(RuntimeError, "exhausted 2 generation batches: 1/2"):
            sampler.collect(collect, 7, record_round=lambda report: rounds.append(deepcopy(report)))
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[-1]["candidate_groups"], 6)
        self.assertEqual(sampler.state_dict()["batch_index"], 0)

    def test_nonpositive_budget_allows_additional_generation_rounds(self):
        for budget in (0, -1):
            with self.subTest(budget=budget):
                sampler = DynamicGroupSampler(settings(max_num_gen_batches=budget), examples(10))
                def collect(example, index):
                    return group_for(example, (-1.0, 1.0) if index in (6, 8) else (-1.0, -1.0))
                groups, report = sampler.collect(collect, 7)
                self.assertEqual(len(groups), 2)
                self.assertEqual(report["generation_batches"], 3)
                self.assertEqual(report["candidate_groups"], 9)

    def test_zero_signal_dataset_is_filtered_before_any_optimizer_update(self):
        sampler = DynamicGroupSampler(settings(max_num_gen_batches=0), examples(5))
        update = Mock(side_effect=AssertionError("Zero-signal group reached the optimizer"))
        groups, report = sampler.collect(lambda example, index: group_for(example, (-1.0, -1.0)), 7)
        for group in groups:
            update(group)
        update.assert_not_called()
        self.assertEqual(groups, [])
        self.assertTrue(report["dataset_exhausted"])
        self.assertEqual(report["zero_reward_variance_rate"], 1.0)
        self.assertEqual(report["no_unmasked_signal_rate"], 1.0)
        self.assertEqual(report["candidate_groups"], 5)

    def test_incomplete_dataset_tail_is_reported_and_not_silently_shrunk(self):
        sampler = DynamicGroupSampler(settings(), examples(1))
        groups, report = sampler.collect(lambda example, index: group_for(example), 7)
        self.assertEqual(groups, [])
        self.assertTrue(report["dataset_exhausted"])
        self.assertEqual(report["dropped_tail_groups"], 1)
        self.assertEqual(sampler.state_dict()["batch_index"], 0)

    def test_collect_rejects_incomplete_or_foreign_behavior_groups(self):
        for failure in ("short", "changed_version", "mixed_version", "wrong_query", "mixed_memory"):
            with self.subTest(failure=failure):
                sampler = DynamicGroupSampler(settings(), examples(4))
                def collect(example, index):
                    group = group_for(example)
                    if failure == "short":
                        group.pop()
                    elif failure == "changed_version":
                        for row in group:
                            row.policy_version = 8
                    elif failure == "mixed_version":
                        group[1].policy_version = 8
                    elif failure == "wrong_query":
                        for row in group:
                            row.sample_id = "different-query"
                    else:
                        group[1].memory_id = "different-memory"
                    return group
                with self.assertRaises(ValueError):
                    sampler.collect(collect, 7)

    def test_resume_cursor_preserves_future_candidates_and_epoch_shuffle(self):
        config = settings(epochs=2, candidate_groups=3, target_groups=2)
        source = examples(3)
        original = DynamicGroupSampler(config, source)
        original.collect(lambda example, index: group_for(example), 7)
        state = original.state_dict()
        restored = DynamicGroupSampler(config, source, **state)
        left, right = [], []
        def collector(records):
            def collect(example, index):
                records.append((example.sample_id, index))
                return group_for(example, version=9)
            return collect
        # The second batch crosses an epoch boundary (position 2 then 0).
        groups_left, report_left = original.collect(collector(left), 9)
        groups_right, report_right = restored.collect(collector(right), 9)
        self.assertEqual(left, right)
        self.assertEqual([index for _, index in left], [2, 3])
        self.assertEqual(report_left, report_right)
        self.assertEqual([[row.to_dict() for row in group] for group in groups_left],
                         [[row.to_dict() for row in group] for group in groups_right])
        self.assertEqual(original.state_dict(), restored.state_dict())
        self.assertEqual(original.state_dict()["epoch"], 1)
        self.assertEqual(original.state_dict()["next_index"], 1)


class StreamingSamplingTests(unittest.TestCase):
    @staticmethod
    def collector(score, *, on_yield=None, on_close=None):
        """Model the production collector's fixed four-group windows."""
        def collect(candidates):
            try:
                while True:
                    pending = list(islice(candidates, 4))
                    if not pending:
                        break
                    for example, index in pending:
                        if on_yield:
                            on_yield(example, index)
                        yield example, index, score(example, index)
            finally:
                if on_close:
                    on_close()
        return collect

    def test_lazy_refill_preserves_all_candidates_order_and_complete_groups(self):
        config = judge_cost_settings(group_size=8, candidate_groups=32, target_groups=16)
        source = examples(80)
        sampler = DynamicGroupSampler(config, source)
        recorded, rounds, closed = [], [], []
        fallback = Mock(side_effect=AssertionError("Serial fallback must not run"))

        def score(example, index):
            return group_for(example, [-1, 1] * 4 if index % 4 == 0 else [-1] * 8)

        def before_yield(example, index):
            self.assertEqual(sampler.candidate_index, index)
            self.assertGreater(sampler.next_index, index)
            self.assertLessEqual(sampler.next_index - index, 4)

        def record_round(report):
            # The collector is drained before round reports or optimizer work.
            self.assertEqual(len(closed), report["generation_batches"])
            rounds.append(deepcopy(report))

        groups, report = sampler.collect(
            fallback, 7,
            collect_candidates=self.collector(score, on_yield=before_yield, on_close=lambda: closed.append(True)),
            record_candidate=lambda group, result: recorded.append(deepcopy(result)),
            record_round=record_round)
        fallback.assert_not_called()
        self.assertEqual(len(groups), 16)
        self.assertTrue(all(len(group) == 8 for group in groups))
        self.assertEqual([item["candidate_id"] for item in recorded], list(range(64)))
        self.assertEqual([item["candidate_id"] for item in recorded if item["selected"]], list(range(0, 64, 4)))
        self.assertEqual([item["selected_groups"] for item in rounds], [8, 16])
        self.assertEqual(report["generation_batches"], 2)
        self.assertEqual(report["candidate_groups"], 64)
        self.assertEqual(sampler.state_dict()["candidate_index"], 64)
        self.assertEqual(sampler.state_dict()["batch_index"], 1)

        serial = DynamicGroupSampler(config, source)
        serial_groups, serial_report = serial.collect(score, 7)
        # Serial sampling stops immediately at candidate 60; concurrent
        # sampling also finishes the three already admitted window siblings.
        self.assertEqual(serial_report["candidate_groups"], 61)
        self.assertEqual([[row.to_dict() for row in group] for group in groups],
                         [[row.to_dict() for row in group] for group in serial_groups])
        self.assertEqual(serial.candidate_index, 61)

    def test_full_target_stops_before_admitting_another_window(self):
        sampler = DynamicGroupSampler(settings(group_size=8, candidate_groups=32, target_groups=16), examples(40))
        recorded = []
        groups, report = sampler.collect(None, 7, collect_candidates=self.collector(
            lambda example, index: group_for(example, [-1, 1] * 4)),
            record_candidate=lambda group, result: recorded.append(result))
        self.assertEqual(len(groups), 16)
        self.assertEqual(report["accepted_groups"], 16)
        self.assertEqual([row["selected"] for row in recorded], [True] * 16)
        self.assertEqual([row["candidate_id"] for row in recorded], list(range(16)))
        self.assertEqual(sampler.candidate_index, 16)

    def test_refill_stops_at_36_candidates_instead_of_finishing_64(self):
        config = settings(group_size=8, candidate_groups=32, target_groups=16)
        sampler = DynamicGroupSampler(config, examples(80))
        recorded, rounds = [], []

        def score(example, index):
            mixed = index < 14 or index in (34, 35)
            return group_for(example, [-1, 1] * 4 if mixed else [-1] * 8)

        groups, report = sampler.collect(None, 7, collect_candidates=self.collector(score),
            record_candidate=lambda group, result: recorded.append(result), record_round=rounds.append)
        self.assertEqual(len(groups), 16)
        self.assertEqual([row["selected_groups"] for row in rounds], [14, 16])
        self.assertEqual(report["generation_batches"], 2)
        self.assertEqual(report["candidate_groups"], 36)
        self.assertEqual([row["candidate_id"] for row in recorded], list(range(36)))
        self.assertEqual(sampler.candidate_index, 36)
        self.assertEqual(sampler.next_index, 36)
        self.assertFalse(report["dataset_exhausted"])

    def test_mid_window_target_drains_siblings_and_resume_starts_at_next_candidate(self):
        config, source = settings(candidate_groups=32, target_groups=2), examples(20)
        sampler = DynamicGroupSampler(config, source)
        recorded, closed = [], []
        groups, report = sampler.collect(None, 7, collect_candidates=self.collector(
            lambda example, index: group_for(example), on_close=lambda: closed.append(True)),
            record_candidate=lambda group, result: recorded.append(result))
        self.assertEqual(len(groups), 2)
        self.assertEqual(closed, [True])
        self.assertEqual([row["selected"] for row in recorded], [True, True, False, False])
        self.assertEqual(report["accepted_groups"], 4)
        self.assertEqual(sampler.next_index, 4)
        restored = DynamicGroupSampler(config, source, **sampler.state_dict())
        seen = []
        restored.collect(None, 9, collect_candidates=self.collector(
            lambda example, index: group_for(example, version=9),
            on_yield=lambda example, index: seen.append(index)))
        self.assertEqual(seen, [4, 5, 6, 7])
        self.assertEqual(restored.candidate_index, 8)
        self.assertEqual(restored.next_index, 8)

    def test_failure_in_admitted_sibling_is_not_hidden_after_target_is_reached(self):
        sampler = DynamicGroupSampler(settings(candidate_groups=32, target_groups=1), examples(20))
        recorded, closed = [], []

        def score(example, index):
            if index == 2:
                raise RuntimeError("in-flight scoring failed")
            return group_for(example)

        with self.assertRaisesRegex(RuntimeError, "in-flight scoring failed"):
            sampler.collect(None, 7, collect_candidates=self.collector(
                score, on_close=lambda: closed.append(True)),
                record_candidate=lambda group, result: recorded.append(result))
        self.assertEqual([row["selected"] for row in recorded], [True, False])
        self.assertEqual(closed, [True])
        self.assertEqual(sampler.batch_index, 0)

    def test_rejects_out_of_order_duplicate_missing_unrequested_or_foreign_results(self):
        for failure in ("out_of_order", "duplicate", "missing", "unrequested", "foreign", "early_end"):
            with self.subTest(failure=failure):
                sampler = DynamicGroupSampler(settings(), examples(5))
                closed = []

                def collect(candidates):
                    try:
                        if failure == "unrequested":
                            example = examples(1)[0]
                            yield example, 0, group_for(example)
                            return
                        if failure == "early_end":
                            return
                        pending = list(candidates)
                        if failure == "out_of_order":
                            pending.reverse()
                        elif failure == "duplicate":
                            pending.insert(1, pending[0])
                        elif failure == "missing":
                            pending.pop()
                        elif failure == "foreign":
                            example, index = pending[0]
                            example = deepcopy(example)
                            example.query = "changed question"
                            pending[0] = example, index
                        for example, index in pending:
                            yield example, index, group_for(example)
                    finally:
                        closed.append(True)

                with self.assertRaisesRegex(ValueError, "DAPO candidate"):
                    sampler.collect(None, 7, collect_candidates=collect)
                self.assertEqual(closed, [True])
                self.assertEqual(sampler.batch_index, 0)
                self.assertEqual(sampler.candidate_index, {"duplicate": 1, "missing": 2}.get(failure, 0))

    def test_invalid_groups_close_stream_before_raising_and_do_not_commit_candidate(self):
        for failure in ("short", "version", "sample", "memory", "score"):
            with self.subTest(failure=failure):
                sampler = DynamicGroupSampler(settings(), examples(5))
                closed = []

                def score(example, index):
                    group = group_for(example)
                    if failure == "short":
                        group.pop()
                    elif failure == "version":
                        group[-1].policy_version = 8
                    elif failure == "sample":
                        group[-1].sample_id = "another sample"
                    elif failure == "memory":
                        group[-1].memory_id = "another memory"
                    else:
                        group[-1].judge_score = float("nan")
                    return group

                with self.assertRaises(ValueError):
                    sampler.collect(None, 7, collect_candidates=self.collector(score, on_close=lambda: closed.append(True)))
                self.assertEqual(closed, [True])
                self.assertEqual(sampler.candidate_index, 0)

    def test_scoring_or_audit_failure_closes_stream_before_propagation(self):
        for failure in ("scoring", "recording"):
            with self.subTest(failure=failure):
                sampler = DynamicGroupSampler(settings(), examples(5))
                closed = []

                def score(example, index):
                    if failure == "scoring":
                        raise RuntimeError("scoring failed")
                    return group_for(example)

                def record(group, result):
                    raise RuntimeError("recording failed")

                with self.assertRaisesRegex(RuntimeError, failure + " failed"):
                    sampler.collect(None, 7, collect_candidates=self.collector(score, on_close=lambda: closed.append(True)),
                                    record_candidate=record)
                self.assertEqual(closed, [True])
                self.assertEqual(sampler.batch_index, 0)

    def test_streaming_refill_budget_and_dataset_tail_match_serial(self):
        config = settings(max_num_gen_batches=2)
        sampler = DynamicGroupSampler(config, examples(20))
        closed, reports = [], []

        def score(example, index):
            return group_for(example, (-1, 1) if index == 0 else (-1, -1))

        with self.assertRaisesRegex(RuntimeError, "exhausted 2 generation batches: 1/2"):
            sampler.collect(None, 7, collect_candidates=self.collector(score, on_close=lambda: closed.append(True)),
                            record_round=lambda report: reports.append(deepcopy(report)))
        self.assertEqual(len(closed), 2)
        self.assertEqual(reports[-1]["candidate_groups"], 6)
        self.assertEqual(sampler.candidate_index, 6)
        self.assertEqual(sampler.batch_index, 0)

        for size in (1, 3, 5):
            with self.subTest(size=size):
                source = examples(size)
                parallel, serial = DynamicGroupSampler(config, source), DynamicGroupSampler(config, source)
                result = parallel.collect(None, 7, collect_candidates=self.collector(score))
                expected = serial.collect(score, 7)
                self.assertEqual(result, expected)
                self.assertEqual(parallel.state_dict(), serial.state_dict())
                self.assertEqual(result[0], [])
                self.assertTrue(result[1]["dataset_exhausted"])
                self.assertEqual(result[1]["dropped_tail_groups"], 1)

    def test_stream_resume_preserves_candidate_ids_and_shuffle_across_epoch(self):
        config, source = settings(epochs=2), examples(5)
        sampler = DynamicGroupSampler(config, source)
        sampler.collect(None, 7, collect_candidates=self.collector(lambda example, index: group_for(example)))
        restored = DynamicGroupSampler(config, source, **sampler.state_dict())
        original_seen, restored_seen = [], []

        def score(example, index):
            return group_for(example, version=9)

        original_result = sampler.collect(None, 9, collect_candidates=self.collector(
            score, on_yield=lambda example, index: original_seen.append((example.sample_id, index))))
        restored_result = restored.collect(None, 9, collect_candidates=self.collector(
            score, on_yield=lambda example, index: restored_seen.append((example.sample_id, index))))
        self.assertEqual(original_seen, restored_seen)
        self.assertEqual([index for _, index in original_seen], [3, 4, 5])
        self.assertEqual(original_result, restored_result)
        self.assertEqual(sampler.state_dict(), restored.state_dict())
        self.assertEqual(sampler.epoch, 1)
        self.assertEqual(sampler.next_index, 1)


class GroupAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.example = examples(1)[0]

    def test_advantages_use_complete_group_and_keep_negative_rollouts(self):
        config = settings(group_size=4)
        group = group_for(self.example, (-3.0, -1.0, 0.0, 4.0), judges=(0, 0, 1, 1))
        probabilities = [deepcopy(row.decisions[0].completion.old_logprobs) for row in group]
        result = assess_group(group, config)
        self.assertTrue(result["accepted"])
        std = math.sqrt((9 + 1 + 0 + 16) / 4)
        for row, expected in zip(group, (-3.0, -1.0, 0.0, 4.0)):
            self.assertAlmostEqual(row.advantage, expected / (std + 1e-6))
        self.assertEqual(len(group), 4)
        self.assertLess(group[0].advantage, 0)
        self.assertGreater(group[-1].advantage, 0)
        self.assertEqual([row.decisions[0].completion.old_logprobs for row in group], probabilities)

    def test_each_question_is_normalized_separately(self):
        config = settings()
        low_scale = group_for(self.example, (-0.2, -0.1), judges=(0, 0))
        high_scale = group_for(self.example, (-10, 10), judges=(0, 1))
        assess_group(low_scale, config)
        assess_group(high_scale, config)
        self.assertAlmostEqual(low_scale[0].advantage, -0.05 / (0.05 + 1e-6))
        self.assertAlmostEqual(high_scale[0].advantage, -10 / (10 + 1e-6))

    def test_judge_uniform_groups_can_have_cost_or_format_signal(self):
        cases = [((0.99, 0.98), (1, 1), "judge_all_correct"),
                 ((-1.0, -1.1), (0, 0), "judge_all_wrong")]
        for rewards, judges, expected in cases:
            with self.subTest(expected=expected):
                group = group_for(self.example, rewards, judges)
                result = assess_group(group, settings())
                self.assertTrue(result["accepted"])
                self.assertTrue(result[expected])
                self.assertTrue(result["cost_format_only_signal"])
                self.assertFalse(result["auxiliary_only_signal"])
                self.assertFalse(result["judge_mixed"])
                # Selecting the Judge as filter metric is intentionally different.
                self.assertFalse(assess_group(group, settings(filter_metric="judge"))["accepted"])

    def test_auxiliary_only_signal_is_identified_and_std_floor_limits_amplification(self):
        config = settings()
        config["reward"]["likelihood"] = {"mode": "reward", "coefficient": 0.1,
                                          "advantage_std_floor": 0.2}
        validate_dapo_config(config)
        group = group_for(self.example, (-1.01, -0.99), judges=(0, 0), auxiliary=(-0.01, 0.01))
        result = assess_group(group, config)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["auxiliary_only_signal"])
        self.assertFalse(result["cost_format_only_signal"])
        self.assertAlmostEqual(group[0].advantage, -0.01 / (0.2 + 1e-6))
        self.assertLess(abs(group[0].advantage), 0.051)

    def test_overlong_mask_removes_groups_without_remaining_advantage_signal(self):
        config = settings(group_size=3)
        # The one unmasked rollout equals the complete group's mean; both
        # nonzero advantages belong to trajectories masked for truncation.
        group = group_for(self.example, (-1, 0, 1), judges=(0, 0, 1), truncated=(0, 2))
        result = assess_group(group, config)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["no_unmasked_signal"])
        self.assertGreater(result["reward_std"], 0)
        self.assertAlmostEqual(result["truncated_rollout_fraction"], 2 / 3)
        config["training"]["overlong_filter"] = False
        self.assertTrue(assess_group(group, config)["accepted"])
        all_truncated = group_for(self.example, (-1, 1), truncated=(0, 1))
        self.assertFalse(assess_group(all_truncated, settings())["accepted"])

    def test_statistics_measure_candidate_diversity_and_reject_nonfinite_scores(self):
        group = group_for(self.example)
        group[1].evidence = deepcopy(group[0].evidence)
        group[1].prediction = group[0].prediction
        result = assess_group(group, settings())
        self.assertEqual(result["unique_evidence_fraction"], 0.5)
        self.assertEqual(result["unique_prediction_fraction"], 0.5)
        for field, value in (("reward", float("nan")), ("reward", float("inf")), ("judge_score", None)):
            with self.subTest(field=field, value=value):
                invalid = deepcopy(group)
                setattr(invalid[0], field, value)
                with self.assertRaises(ValueError):
                    assess_group(invalid, settings())

    def test_configuration_enforces_batch_units_and_auxiliary_scaling(self):
        for key, value in (("group_size", 1), ("candidate_groups", 0), ("target_groups", 0),
                           ("mini_batch_groups", 0), ("mini_batch_groups", 3),
                           ("micro_batch_rollouts", 0), ("micro_batch_rollouts", 3), ("ppo_epochs", 2),
                           ("max_micro_batch_tokens", 0), ("max_micro_batch_tokens", True),
                           ("max_num_gen_batches", True), ("kl_coefficient", 0.01),
                           ("filter_metric", "unknown"), ("filter_min_std", float("nan")),
                           ("advantage_epsilon", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_dapo_config(settings(**{key: value}))
        config = settings()
        config["reward"]["likelihood"] = {"mode": "reward", "coefficient": 0.1}
        with self.assertRaisesRegex(ValueError, "advantage_std_floor"):
            validate_dapo_config(config)

    def test_real_batching_accepts_multiple_groups_and_rollouts_in_declared_units(self):
        for mini, micro in ((1, 2), (2, 2), (2, 4)):
            validate_dapo_config(settings(mini_batch_groups=mini, micro_batch_rollouts=micro))
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_dapo_config(settings(target_groups=3, mini_batch_groups=2))


class JudgeCostGroupTests(unittest.TestCase):
    def test_actual_training_tokens_must_include_both_judge_outcomes_after_masking(self):
        config = judge_cost_settings(group_size=8)
        for minority in (0, 1):
            judges = [minority] + [1 - minority] * 7
            rewards = [2 * judge - 1 for judge in judges]
            for cause in ("minority_truncated", "minority_empty", "all_truncated", "all_empty"):
                with self.subTest(minority=minority, cause=cause):
                    group = group_for(examples(1)[0], rewards, judges)
                    indices = range(8) if cause.startswith("all_") else [0]
                    for index in indices:
                        completion = group[index].decisions[0].completion
                        if cause.endswith("truncated"):
                            completion.truncated = True
                        else:
                            completion.token_ids = []
                            completion.old_logprobs = []
                    report = assess_group(group, config)
                    self.assertTrue(report["judge_mixed"])
                    self.assertFalse(report["accepted"])
                    self.assertTrue(report["no_unmasked_judge_signal"])
                    self.assertEqual(len(group), 8)
                    if cause == "minority_truncated":
                        unfiltered = deepcopy(config)
                        unfiltered["training"]["overlong_filter"] = False
                        self.assertTrue(assess_group(group, unfiltered)["accepted"])

    def test_uniform_judge_groups_are_filtered_despite_cost_and_format_advantages(self):
        config = judge_cost_settings(group_size=8)
        for judge, rewards in ((0, [-1.0, -1.18] * 4), (1, [1.0, 0.82] * 4)):
            with self.subTest(judge=judge):
                group = group_for(examples(1)[0], rewards, judges=[judge] * 8)
                report = assess_group(group, config)
                self.assertFalse(report["accepted"])
                self.assertTrue(report["cost_format_only_signal"])
                self.assertTrue(any(row.advantage > 0 for row in group))
                self.assertTrue(any(row.advantage < 0 for row in group))

    def test_eight_rollout_mixed_groups_keep_correct_positive_wrong_negative_at_cost_bounds(self):
        config = judge_cost_settings(group_size=8)
        for correct in range(1, 8):
            judges = [1] * correct + [0] * (8 - correct)
            # For each rollout, maximize the reward disadvantage relative to
            # every peer, then minimize it; this exercises n<=8 and f<=1 bounds.
            for focus in range(8):
                for low in (False, True):
                    rewards = [2 * judge - 1 - (0.18 if (i == focus) == low else 0)
                               for i, judge in enumerate(judges)]
                    group = group_for(examples(1)[0], rewards, judges=judges)
                    report = assess_group(group, config)
                    self.assertTrue(report["accepted"])
                    mean = sum(rewards) / 8
                    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / 8)
                    self.assertEqual(len(group), 8)
                    for row, judge in zip(group, judges):
                        self.assertGreater(row.advantage, 0) if judge else self.assertLess(row.advantage, 0)
                        self.assertAlmostEqual(row.advantage, (row.reward - mean) / (std + 1e-6))

    def test_new_protocol_rejects_std_floor_and_auxiliary_reward(self):
        for section, key, value, message in (("training", "advantage_std_floor", 0.5, "advantage_std_floor=0"),
                                              ("reward", "likelihood", {"mode": "reward"}, "off/diagnostic")):
            config = judge_cost_settings()
            config[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate_dapo_config(config)
            with self.assertRaisesRegex(ValueError, message):
                assess_group(group_for(examples(1)[0]), config)
        for mode in ("off", "diagnostic"):
            config = judge_cost_settings()
            config["reward"]["likelihood"]["mode"] = mode
            validate_dapo_config(config)

    def test_sampling_diagnostics_cover_rejected_candidates_and_keep_complete_mixed_groups(self):
        config = judge_cost_settings(group_size=8, candidate_groups=4, target_groups=2)
        def collect(example, index):
            judges = ([0] * 8 if index == 0 else [1] * 8 if index == 1 else [0, 1] * 4)
            rewards = [2 * judge - 1 - 0.01 * i for i, judge in enumerate(judges)]
            group = group_for(example, rewards, judges)
            for i, row in enumerate(group):
                row.n_read_calls = i
            return group
        groups, report = DynamicGroupSampler(config, examples(4)).collect(collect, 7)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(group) == 8 for group in groups))
        self.assertEqual(report["judge_accuracy"], 0.5)
        self.assertAlmostEqual(report["reward"], -0.035)
        self.assertEqual(report["read_calls"], 3.5)
        self.assertEqual(report["judge_all_correct_rate"], 0.25)
        self.assertEqual(report["judge_all_wrong_rate"], 0.25)
        self.assertEqual(report["judge_mixed_rate"], 0.5)
        self.assertEqual(report["selected_groups"], 2)


class RewardOnlyGroupTests(unittest.TestCase):
    def setUp(self):
        self.config = judge_cost_settings(group_size=8, filter_min_std=0.0)
        self.config["training"]["filter_metric"] = "reward"
        self.config["reward"]["read_call_cost"] = 0.0
        self.example = examples(1)[0]
        validate_dapo_config(self.config)

    def test_uniform_correctness_with_format_signal_keeps_all_eight_rollouts(self):
        for judge in (0, 1):
            rewards = [2 * judge - 1 - 0.1 * (i % 2) for i in range(8)]
            group = group_for(self.example, rewards, judges=[judge] * 8)
            with self.subTest(judge=judge):
                result = assess_group(group, self.config)
                self.assertTrue(result["accepted"])
                self.assertTrue(result["cost_format_only_signal"])
                self.assertTrue(result["no_unmasked_judge_signal"])
                self.assertEqual(len(group), 8)
                self.assertEqual(sum(row.advantage > 0 for row in group), 4)
                self.assertEqual(sum(row.advantage < 0 for row in group), 4)

    def test_identical_decimal_rewards_have_exactly_zero_advantages(self):
        for reward in (1.0, 0.9, 0.7, -1.0, -1.1, -1.3):
            with self.subTest(reward=reward):
                group = group_for(self.example, [reward] * 8)
                result = assess_group(group, self.config)
                self.assertFalse(result["accepted"])
                self.assertTrue(result["zero_reward_variance"])
                self.assertEqual(result["reward_std"], 0.0)
                self.assertEqual([row.advantage for row in group], [0.0] * 8)

    def test_masked_minority_judge_no_longer_blocks_nonzero_trainable_advantages(self):
        for minority in (0, 1):
            judges = [minority] + [1 - minority] * 7
            group = group_for(self.example, [2 * j - 1 for j in judges], judges, truncated=(0,))
            result = assess_group(group, self.config)
            self.assertTrue(result["accepted"])
            self.assertTrue(result["no_unmasked_judge_signal"])
            self.assertEqual(len(group), 8)
            self.assertTrue(all(row.advantage != 0 for row in group[1:]))

    def test_no_signal_in_trainable_tokens_is_still_filtered(self):
        # Six remaining trajectories equal the full-group reward mean.
        for cause in ("truncated", "empty_tokens"):
            group = group_for(self.example, [-1.0, -1.25] + [-1.125] * 6, judges=[0] * 8)
            for row in group[:2]:
                completion = row.decisions[0].completion
                if cause == "truncated":
                    completion.truncated = True
                else:
                    completion.token_ids = []
                    completion.old_logprobs = []
            with self.subTest(cause=cause):
                result = assess_group(group, self.config)
                self.assertFalse(result["accepted"])
                self.assertTrue(result["no_unmasked_signal"])
                self.assertGreater(result["reward_std"], 0)

    def test_collection_selects_uniform_and_mixed_groups_without_dropping_rollouts(self):
        config = deepcopy(self.config)
        config["training"].update(candidate_groups=4, target_groups=3)
        recorded = []

        def collect(example, index):
            judges = [0] * 8 if index < 2 else [1] * 8 if index == 2 else [0, 1] * 4
            rewards = [2 * j - 1 - (0.1 * (i % 2) if index else 0) for i, j in enumerate(judges)]
            return group_for(example, rewards, judges)

        groups, report = DynamicGroupSampler(config, examples(4)).collect(
            collect, 7, record_candidate=lambda group, result: recorded.append(result))
        self.assertEqual([r["selected"] for r in recorded], [False, True, True, True])
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(group) == 8 for group in groups))
        self.assertEqual(report["selected_groups"], 3)
        self.assertEqual(report["generation_batches"], 1)


if __name__ == "__main__":
    unittest.main()
