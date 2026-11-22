"""Process rewards allow all-correct groups without terminal reward contrasts."""
from copy import deepcopy
import unittest

from agmem.dapo_sampling import DynamicGroupSampler, assess_group, validate_dapo_config
from tests.test_dapo_all_correct_format import format_group
from tests.test_dapo_sampling import examples, group_for, judge_cost_settings


def config_for(**overrides):
    config = judge_cost_settings()
    config["training"].update(filter_metric="exclude_all_wrong", mini_batch_groups=2,
                              process_reward={"enabled": True}, **overrides)
    config["reader"]["prompt_version"] = "interaction_history_v4"
    config["retrieval"] = {"id_protocol": "strict_v2"}
    config["reward"].update(format_penalty=0.5, format_penalty_mode="per_id", read_call_cost=0.01)
    return config


def mask_row(row, cause):
    completion = row.decisions[0].completion
    if cause == "truncated":
        completion.truncated = True
    else:
        completion.token_ids = []
        completion.old_logprobs = []


class AllCorrectProcessTests(unittest.TestCase):
    def setUp(self):
        self.example = examples(1)[0]
        self.config = config_for()

    def test_new_filter_requires_enabled_process_rewards(self):
        validate_dapo_config(self.config)
        for process_reward in ({}, {"enabled": False}):
            with self.subTest(process_reward=process_reward):
                config = deepcopy(self.config)
                config["training"]["process_reward"] = process_reward
                with self.assertRaisesRegex(ValueError, "exclude_all_wrong requires"):
                    validate_dapo_config(config)
                with self.assertRaisesRegex(ValueError, "exclude_all_wrong requires"):
                    assess_group(format_group(self.example), config)

    def test_all_correct_constant_reward_preserves_zero_terminal_advantages(self):
        for counts in ((0, 0), (1, 1), (3, 3)):
            with self.subTest(counts=counts):
                group = format_group(self.example, counts=counts)
                old_logprobs = [deepcopy(row.decisions[0].completion.old_logprobs) for row in group]
                report = assess_group(group, self.config)
                self.assertTrue(report["accepted"])
                self.assertTrue(report["accepted_all_correct"])
                self.assertTrue(report["accepted_zero_terminal_advantage"])
                self.assertTrue(report["zero_reward_variance"])
                self.assertTrue(report["no_unmasked_signal"])
                self.assertFalse(report["accepted_all_correct_format"])
                self.assertFalse(report["no_trainable_correct_signal"])
                self.assertEqual([row.advantage for row in group], [0.0, 0.0])
                self.assertEqual(old_logprobs, [row.decisions[0].completion.old_logprobs for row in group])

    def test_all_correct_cost_only_and_nonzero_format_contrasts_are_accepted(self):
        cases = {
            "cost_only": format_group(self.example, counts=(0, 0), read_calls=(1, 2)),
            "both_penalized": format_group(self.example, counts=(1, 2)),
            "clean_and_penalized": format_group(self.example),
        }
        for name, group in cases.items():
            with self.subTest(case=name):
                report = assess_group(group, self.config)
                self.assertTrue(report["accepted_all_correct"])
                self.assertFalse(report["accepted_zero_terminal_advantage"])
                self.assertEqual(report["accepted_all_correct_format"], name == "clean_and_penalized")

    def test_all_wrong_groups_are_rejected_even_with_cost_or_format_variance(self):
        for counts, calls in (((0, 0), (1, 1)), ((0, 0), (1, 2)), ((0, 1), (1, 1))):
            with self.subTest(counts=counts, calls=calls):
                group = format_group(self.example, counts=counts, judges=(0, 0), read_calls=calls)
                report = assess_group(group, self.config)
                self.assertFalse(report["accepted"])
                self.assertFalse(report["accepted_all_correct"])
                self.assertFalse(report["accepted_zero_terminal_advantage"])
                self.assertTrue(report["no_trainable_correct_signal"])

    def test_mixed_group_is_accepted_even_when_terminal_rewards_are_tied(self):
        group = format_group(self.example, counts=(0, 4), judges=(0, 1))
        report = assess_group(group, self.config)
        self.assertTrue(report["accepted"])
        self.assertTrue(report["judge_mixed"])
        self.assertTrue(report["accepted_zero_terminal_advantage"])
        self.assertFalse(report["accepted_all_correct"])
        self.assertEqual([row.advantage for row in group], [0.0, 0.0])

    def test_mixed_group_requires_trainable_correct_but_not_trainable_wrong_outcome(self):
        for cause in ("truncated", "empty"):
            for missing in (0, 1):
                with self.subTest(cause=cause, missing=missing):
                    group = format_group(self.example, counts=(0, 0), judges=(0, 1))
                    mask_row(group[missing], cause)
                    report = assess_group(group, self.config)
                    self.assertEqual(report["accepted"], missing == 0)
                    self.assertEqual(report["no_trainable_correct_signal"], missing == 1)
                    self.assertFalse(report["accepted_all_correct"])
                    if cause == "truncated":
                        config = deepcopy(self.config)
                        config["training"]["overlong_filter"] = False
                        self.assertTrue(assess_group(group, config)["accepted"])

    def test_all_correct_groups_need_at_least_one_trainable_rollout(self):
        for cause in ("truncated", "empty"):
            for num_masked in (1, 2):
                with self.subTest(cause=cause, num_masked=num_masked):
                    group = format_group(self.example, counts=(0, 0))
                    for row in group[:num_masked]:
                        mask_row(row, cause)
                    report = assess_group(group, self.config)
                    self.assertEqual(report["accepted"], num_masked == 1)
                    self.assertEqual(report["accepted_zero_terminal_advantage"], num_masked == 1)
                    self.assertEqual(report["no_trainable_correct_signal"], num_masked == 2)

    def test_zero_terminal_metric_checks_whole_group_not_only_unmasked_rows(self):
        group = group_for(self.example, rewards=(0.5, 1.0, 1.5), judges=(1, 1, 1), truncated=(0, 2))
        report = assess_group(group, self.config)
        self.assertTrue(report["accepted"])
        self.assertTrue(report["no_unmasked_signal"])
        self.assertFalse(report["accepted_zero_terminal_advantage"])
        self.assertEqual(group[1].advantage, 0.0)

    def test_new_filter_does_not_require_reward_or_judge_std_threshold(self):
        config = config_for(filter_min_std=10)
        for judges in ((1, 1), (0, 1)):
            with self.subTest(judges=judges):
                self.assertTrue(assess_group(format_group(self.example, judges=judges), config)["accepted"])

    def test_dynamic_sampling_selects_all_correct_zero_terminal_groups_after_refill(self):
        sampler = DynamicGroupSampler(config_for(candidate_groups=2), examples(6))
        candidates, rounds = [], []

        def collect(example, candidate_id):
            if candidate_id in (0, 2):
                return format_group(example, judges=(0, 0))
            return format_group(example, counts=(0, 0))

        groups, report = sampler.collect(
            collect, 7,
            record_candidate=lambda group, result: candidates.append(deepcopy(result)),
            record_round=lambda result: rounds.append(deepcopy(result)))
        self.assertEqual(len(groups), 2)
        self.assertEqual([row["candidate_id"] for row in candidates if row["selected"]], [1, 3])
        self.assertTrue(all(row.advantage == 0 for group in groups for row in group))
        self.assertEqual(report["candidate_groups"], 4)
        self.assertEqual(report["accepted_groups"], 2)
        self.assertEqual(report["generation_batches"], 2)
        self.assertEqual(sampler.next_index, 4)
        self.assertEqual([row["selected_groups"] for row in rounds], [1, 2])
        self.assertEqual(report["accepted_all_correct_rate"], .5)
        self.assertEqual(report["accepted_zero_terminal_advantage_rate"], .5)
        self.assertEqual(report["no_trainable_correct_signal_rate"], .5)
        self.assertEqual(report["accepted_all_correct_format_rate"], 0)


if __name__ == "__main__":
    unittest.main()
