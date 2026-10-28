from copy import deepcopy
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
        self.assertEqual(report["candidate_groups"], 64)
        self.assertEqual([row["selected_groups"] for row in rounds], [8, 16])
        self.assertEqual(sum(row["selected"] for row in candidates), 16)
        self.assertEqual(report["accepted_rate"], 0.25)
        self.assertEqual(report["judge_all_wrong_rate"], 0.75)
        self.assertEqual(sum(report["source_candidates"].values()), 64)
        self.assertEqual(sampler.state_dict()["candidate_index"], 64)
        self.assertEqual(sampler.state_dict()["batch_index"], 1)
        self.assertTrue(all(row.policy_version == 7 for group in groups for row in group))

    def test_generation_batch_is_finished_and_extra_accepted_groups_are_audited(self):
        sampler = DynamicGroupSampler(settings(), examples(4))
        recorded = []
        groups, report = sampler.collect(lambda example, index: group_for(example), 7,
                                         record_candidate=lambda group, result: recorded.append(result))
        self.assertEqual(len(groups), 2)
        self.assertEqual(report["accepted_groups"], 3)
        self.assertEqual([row["selected"] for row in recorded], [True, True, False])
        self.assertEqual([row["candidate_id"] for row in recorded], [0, 1, 2])
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
        source = examples(5)
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
        # The second batch crosses an epoch boundary (positions 3,4 then 0).
        groups_left, report_left = original.collect(collector(left), 9)
        groups_right, report_right = restored.collect(collector(right), 9)
        self.assertEqual(left, right)
        self.assertEqual([index for _, index in left], [3, 4, 5])
        self.assertEqual(report_left, report_right)
        self.assertEqual([[row.to_dict() for row in group] for group in groups_left],
                         [[row.to_dict() for row in group] for group in groups_right])
        self.assertEqual(original.state_dict(), restored.state_dict())
        self.assertEqual(original.state_dict()["epoch"], 1)
        self.assertEqual(original.state_dict()["next_index"], 1)


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

    def test_new_protocol_rejects_reward_filter_std_floor_and_auxiliary_reward(self):
        for section, key, value, message in (("training", "filter_metric", "reward", "filter_metric=judge"),
                                              ("training", "advantage_std_floor", 0.5, "advantage_std_floor=0"),
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


if __name__ == "__main__":
    unittest.main()
