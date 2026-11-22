"""Exercise the no-call-cost recipe through retrieval, scoring and filtering."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from agmem.core import load_config
from agmem.dapo_sampling import assess_group
from agmem.env import MemoryEnvironment
from agmem.rewards import RewardScorer
from tests import test_core as fixtures


ROOT = Path(__file__).resolve().parents[1]


class NoCallCostRewardTests(unittest.TestCase):
    tearDown = fixtures.CoreTests.tearDown

    def setUp(self):
        fixtures.CoreTests.setUp(self)
        self.train_config = load_config(ROOT / "configs/train_dapo_max1_no_call_cost_v2.yaml")
        retrieval = self.train_config["retrieval"]
        self.env = MemoryEnvironment(
            self.store, self.builder.version,
            max_ids=retrieval["max_ids_per_call"], max_calls=retrieval["max_read_calls"],
            id_protocol=retrieval["id_protocol"],
            prompt_version=self.train_config["reader"]["prompt_version"],
        )

    def rollout(self, judge_score, *, valid_calls=1, repeated_calls=0, seed=42):
        ids = [entry["id"] for entry in self.directory[:valid_calls]]
        actions = [{"ids": [key], "stop": False} for key in ids]
        actions += [{"ids": [ids[0]], "stop": False} for _ in range(repeated_calls)]
        actions.append({"ids": [], "stop": True})
        row = self.env.run(self.example, fixtures.ScriptedPolicy(actions), seed)
        # Only the policy/Judge are fixtures. Retrieval and reward composition
        # are real; token metadata makes the generated decisions trainable.
        for decision in row.decisions:
            decision.completion.prompt_ids = [1, 2]
            decision.completion.token_ids = [3, 4]
            decision.completion.old_logprobs = [-0.5, -0.75]
        row.prediction = "fixture answer"
        scorer = RewardScorer(
            self.train_config["reward"],
            judge=SimpleNamespace(score=lambda *_: judge_score),
            max_ids=self.env.max_ids,
        )
        return scorer.score(self.example, row)

    def test_valid_read_count_has_no_reward_cost_for_correct_or_wrong_answers(self):
        for judge, expected in ((0, -1.0), (1, 1.0)):
            for calls in (0, 1, 3, 7):
                with self.subTest(judge=judge, calls=calls):
                    row = self.rollout(judge, valid_calls=calls)
                    self.assertEqual(row.n_read_calls, calls)
                    self.assertEqual(len(row.evidence), calls)
                    self.assertEqual(row.raw_read_cost, 0.0)
                    self.assertEqual(row.read_cost, 0.0)
                    self.assertEqual(row.format_violation_count, 0)
                    self.assertEqual(row.reward, expected)
                    self.assertEqual(row.outcome_reward, expected)

    def test_repeated_reads_still_accumulate_one_penalty_per_bad_decision(self):
        for judge in (0, 1):
            for repeated in (1, 2, 3):
                with self.subTest(judge=judge, repeated=repeated):
                    row = self.rollout(judge, valid_calls=2, repeated_calls=repeated)
                    self.assertEqual(row.n_read_calls, 2 + repeated)
                    self.assertEqual(len(row.evidence), 2)
                    self.assertEqual(row.read_cost, 0.0)
                    self.assertEqual(row.format_violation_count, repeated)
                    self.assertTrue(row.format_failure)
                    self.assertAlmostEqual(row.format_reward, -0.1 * repeated)
                    self.assertAlmostEqual(row.reward, 2 * judge - 1 - 0.1 * repeated)

    def test_uniform_judge_group_with_format_differences_is_accepted(self):
        for judge in (0, 1):
            with self.subTest(judge=judge):
                group = [self.rollout(judge, repeated_calls=i % 2, seed=i) for i in range(8)]
                report = assess_group(group, self.train_config)
                self.assertTrue(report["accepted"])
                self.assertTrue(report["cost_format_only_signal"])
                self.assertEqual(report["judge_std"], 0)
                self.assertEqual(len(group), 8)
                for i, row in enumerate(group):
                    if i % 2:
                        self.assertLess(row.advantage, 0)
                    else:
                        self.assertGreater(row.advantage, 0)

    def test_call_count_differences_without_format_differences_are_filtered(self):
        for judge in (0, 1):
            for repeated in (0, 1):
                with self.subTest(judge=judge, repeated=repeated):
                    group = [self.rollout(judge, valid_calls=1 if i % 2 else 3,
                                          repeated_calls=repeated, seed=i) for i in range(8)]
                    self.assertEqual({row.n_read_calls for row in group}, {1 + repeated, 3 + repeated})
                    self.assertEqual({row.format_violation_count for row in group}, {repeated})
                    report = assess_group(group, self.train_config)
                    self.assertFalse(report["accepted"])
                    self.assertEqual(report["reward_std"], 0)
                    self.assertTrue(report["zero_reward_variance"])
                    self.assertEqual([row.advantage for row in group], [0.0] * 8)

    def test_mixed_judge_group_with_same_format_remains_accepted(self):
        group = [self.rollout(i % 2, valid_calls=1 if i % 2 else 3, seed=i) for i in range(8)]
        report = assess_group(group, self.train_config)
        self.assertTrue(report["accepted"])
        self.assertTrue(report["judge_mixed"])
        self.assertFalse(report["cost_format_only_signal"])
        for row in group:
            if row.judge_score:
                self.assertGreater(row.advantage, 0)
            else:
                self.assertLess(row.advantage, 0)


if __name__ == "__main__":
    unittest.main()
