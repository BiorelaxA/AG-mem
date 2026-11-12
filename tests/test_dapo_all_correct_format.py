"""All-correct format contrasts supplement the original Judge-mixed filter."""
from copy import deepcopy
import unittest

from agmem.dapo_sampling import DynamicGroupSampler, assess_group, validate_dapo_config
from tests.test_dapo_sampling import examples, group_for, judge_cost_settings


FILTER = "judge_or_all_correct_format"


def config_for(**overrides):
    config = judge_cost_settings(**overrides)
    config["training"]["filter_metric"] = FILTER
    config["reward"].update(format_penalty=0.5, format_penalty_mode="per_id", read_call_cost=0.01)
    return config


def format_group(example, counts=(0, 1), judges=None, read_calls=None):
    judges = [1] * len(counts) if judges is None else judges
    read_calls = [1] * len(counts) if read_calls is None else read_calls
    rewards = [2 * judge - 1 - 0.01 * calls - 0.5 * count
               for judge, calls, count in zip(judges, read_calls, counts)]
    group = group_for(example, rewards, judges=judges)
    for row, count, calls in zip(group, counts, read_calls):
        row.format_violation_count = count
        row.format_failure = count > 0
        row.format_reward = -0.5 * count
        row.n_read_calls = calls
        row.read_cost = 0.01 * calls
    return group


class AllCorrectFormatTests(unittest.TestCase):
    def setUp(self):
        self.example = examples(1)[0]
        self.config = config_for()

    def test_clean_and_penalized_correct_rollouts_are_kept_with_opposite_advantages(self):
        validate_dapo_config(self.config)
        group = format_group(self.example)
        old_logprobs = [deepcopy(row.decisions[0].completion.old_logprobs) for row in group]
        report = assess_group(group, self.config)
        self.assertTrue(report["accepted"])
        self.assertTrue(report["all_correct_format_signal"])
        self.assertTrue(report["accepted_all_correct_format"])
        self.assertTrue(report["judge_all_correct"])
        self.assertTrue(report["no_unmasked_judge_signal"])
        self.assertGreater(group[0].advantage, 0)
        self.assertLess(group[1].advantage, 0)
        self.assertAlmostEqual(group[0].advantage, 0.25 / (0.25 + 1e-6))
        self.assertEqual(old_logprobs, [row.decisions[0].completion.old_logprobs for row in group])
        self.assertEqual(len(group), 2)

    def test_all_wrong_cost_only_and_penalized_only_groups_are_rejected(self):
        cases = {
            "all_wrong": format_group(self.example, judges=(0, 0)),
            "cost_only_correct": format_group(self.example, counts=(0, 0), read_calls=(1, 2)),
            "different_nonzero_penalties": format_group(self.example, counts=(1, 2)),
            "same_nonzero_penalty": format_group(self.example, counts=(1, 1)),
            "same_zero_penalty": format_group(self.example, counts=(0, 0)),
        }
        for name, group in cases.items():
            with self.subTest(case=name):
                report = assess_group(group, self.config)
                self.assertFalse(report["accepted"])
                self.assertFalse(report["all_correct_format_signal"])
                self.assertFalse(report["accepted_all_correct_format"])

    def test_both_format_outcomes_must_have_trainable_tokens(self):
        for missing in (0, 1):
            for cause in ("truncated", "empty"):
                with self.subTest(missing=missing, cause=cause):
                    group = format_group(self.example)
                    completion = group[missing].decisions[0].completion
                    if cause == "truncated":
                        completion.truncated = True
                    else:
                        completion.token_ids = []
                        completion.old_logprobs = []
                    report = assess_group(group, self.config)
                    self.assertFalse(report["accepted"])
                    self.assertFalse(report["all_correct_format_signal"])
                    self.assertFalse(report["accepted_all_correct_format"])
                    if cause == "truncated":
                        unmasked = deepcopy(self.config)
                        unmasked["training"]["overlong_filter"] = False
                        self.assertTrue(assess_group(group, unmasked)["accepted"])

    def test_no_trainable_tokens_is_rejected(self):
        for cause in ("truncated", "empty"):
            with self.subTest(cause=cause):
                group = format_group(self.example)
                for row in group:
                    completion = row.decisions[0].completion
                    if cause == "truncated":
                        completion.truncated = True
                    else:
                        completion.token_ids = []
                        completion.old_logprobs = []
                report = assess_group(group, self.config)
                self.assertFalse(report["accepted"])
                self.assertTrue(report["no_unmasked_signal"])

    def test_masked_wrong_outcome_cannot_turn_a_mixed_group_into_all_correct(self):
        for cause in ("truncated", "empty"):
            with self.subTest(cause=cause):
                group = format_group(self.example, counts=(0, 1, 0), judges=(1, 1, 0))
                completion = group[2].decisions[0].completion
                if cause == "truncated":
                    completion.truncated = True
                else:
                    completion.token_ids = []
                    completion.old_logprobs = []
                report = assess_group(group, config_for(group_size=3))
                self.assertTrue(report["judge_mixed"])
                self.assertFalse(report["judge_all_correct"])
                self.assertFalse(report["accepted"])
                self.assertFalse(report["all_correct_format_signal"])

    def test_mixed_groups_still_require_both_judge_outcomes_after_masking(self):
        group = format_group(self.example, counts=(0, 0), judges=(0, 1))
        report = assess_group(group, self.config)
        self.assertTrue(report["accepted"])
        self.assertFalse(report["accepted_all_correct_format"])
        self.assertFalse(report["all_correct_format_signal"])
        for missing in (0, 1):
            masked = deepcopy(group)
            masked[missing].decisions[0].completion.truncated = True
            self.assertFalse(assess_group(masked, self.config)["accepted"])

    def test_format_standard_deviation_must_exceed_threshold(self):
        for threshold, accepted in ((0.249, True), (0.25, False), (0.251, False)):
            with self.subTest(threshold=threshold):
                config = config_for(filter_min_std=threshold)
                report = assess_group(format_group(self.example), config)
                self.assertEqual(report["accepted"], accepted)
                self.assertEqual(report["accepted_all_correct_format"], accepted)
        # The original mixed branch applies the same strict threshold to Judge scores.
        config = config_for(filter_min_std=0.5)
        self.assertFalse(assess_group(format_group(self.example, judges=(0, 1)), config)["accepted"])

    def test_format_contrast_without_reward_advantage_is_not_accepted(self):
        group = format_group(self.example)
        for row in group:
            row.reward = 0.5
        report = assess_group(group, self.config)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["accepted_all_correct_format"])
        self.assertTrue(report["zero_reward_variance"])
        self.assertTrue(report["no_unmasked_signal"])
        self.assertEqual([row.advantage for row in group], [0.0, 0.0])

    def test_old_judge_filter_keeps_its_existing_semantics(self):
        config = deepcopy(self.config)
        config["training"]["filter_metric"] = "judge"
        report = assess_group(format_group(self.example), config)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["accepted_all_correct_format"])
        self.assertTrue(assess_group(format_group(self.example, judges=(0, 1)), config)["accepted"])

    def test_dynamic_sampling_selects_format_group_and_reports_its_contribution(self):
        sampler = DynamicGroupSampler(config_for(candidate_groups=4), examples(5))
        candidates, rounds = [], []

        def collect(example, candidate_id):
            if candidate_id == 0:
                return format_group(example, judges=(0, 0))
            if candidate_id == 1:
                return format_group(example, counts=(0, 0), read_calls=(1, 2))
            if candidate_id == 2:
                return format_group(example)
            return format_group(example, counts=(0, 0), judges=(0, 1))

        groups, report = sampler.collect(
            collect, 7,
            record_candidate=lambda group, result: candidates.append(deepcopy(result)),
            record_round=lambda result: rounds.append(deepcopy(result)))
        self.assertEqual(len(groups), 2)
        self.assertEqual([row["candidate_id"] for row in candidates if row["selected"]], [2, 3])
        self.assertEqual([row.judge_score for row in groups[0]], [1, 1])
        self.assertEqual([row.format_reward for row in groups[0]], [0, -0.5])
        self.assertEqual(report["candidate_groups"], 4)
        self.assertEqual(report["accepted_groups"], 2)
        self.assertEqual(report["accepted_rate"], 0.5)
        for metrics in (report, rounds[-1]):
            self.assertEqual(metrics["all_correct_format_signal_rate"], 0.25)
            self.assertEqual(metrics["accepted_all_correct_format_rate"], 0.25)
        self.assertTrue(candidates[2]["accepted_all_correct_format"])
        self.assertFalse(candidates[3]["accepted_all_correct_format"])


if __name__ == "__main__":
    unittest.main()
