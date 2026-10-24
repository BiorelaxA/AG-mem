from copy import deepcopy
import unittest

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.pipeline import metrics
from agmem.rewards import advantages, score_trajectory


class OutcomeGatedRewardTests(unittest.TestCase):
    config = {"formula": "outcome_gated_v2", "read_call_cost": 0.01, "format_penalty": 0.1}

    def scored(self, prediction, calls, failure=False):
        example = Example("fixture/train/0", "fixture", "train", "memory", "Question?",
                          gold_answers=["red blue"])
        row = Trajectory(example.sample_id, 42, example.memory_id, prediction=prediction,
                         n_read_calls=calls, termination_reason="protocol_failure" if failure else "stop")
        return score_trajectory(example, row, self.config)

    def test_failed_group_no_longer_rewards_zero_reads(self):
        group = [self.scored("unknown", n) for n in [0, 0, 0, 1, 1, 1, 1, 1]]
        self.assertEqual([row.reward for row in group], [-1] * 8)
        self.assertEqual(advantages([row.reward for row in group]), [0] * 8)
        self.assertEqual(group[-1].read_cost, 0)
        self.assertEqual(group[-1].raw_read_cost, 0.01)
        self.assertEqual(self.scored("unknown", 8).reward, -1)

    def test_partial_and_correct_answers_gate_cost_and_keep_format_penalty(self):
        partial = self.scored("red green", 3)
        self.assertEqual(partial.outcome, 0.5)
        self.assertEqual(partial.outcome_reward, 0)
        self.assertAlmostEqual(partial.read_cost, 0.015)
        self.assertAlmostEqual(partial.reward, -0.015)
        self.assertAlmostEqual(self.scored("red blue", 1).reward, 0.99)
        self.assertAlmostEqual(self.scored("red blue", 3).reward, 0.97)
        self.assertEqual(self.scored("red blue", 0).reward, 1)
        row = self.scored("unknown", 2, failure=True)
        row.decisions = [Decision([], Completion("invalid"), {"ids": [], "stop": False})] * 2
        row = score_trajectory(Example("fixture/train/0", "fixture", "train", "memory", "?",
                                       gold_answers=["red blue"]), row, self.config)
        self.assertAlmostEqual(row.reward, -1.1)
        self.assertEqual(row.format_reward, -0.1)

    def test_choice_labels_and_legacy_formula(self):
        example = Example("fixture/train/0", "fixture", "train", "memory", "?",
                          options={"A": "red", "B": "blue"}, gold_answers=["A"], answer_mode="choice")
        row = Trajectory(example.sample_id, 0, example.memory_id, prediction="A", n_read_calls=2)
        self.assertAlmostEqual(score_trajectory(example, row, self.config).reward, 0.98)
        row.prediction = "B"
        self.assertEqual(score_trajectory(example, row, self.config).reward, -1)
        legacy = {k: v for k, v in self.config.items() if k != "formula"}
        self.assertEqual(score_trajectory(example, row, legacy).reward, -0.02)
        invalid = {**self.config, "formula": "typo"}
        with self.assertRaisesRegex(ValueError, "Unknown reward"):
            score_trajectory(example, row, invalid)

    def test_metrics_average_weighted_cost_per_rollout(self):
        group = [self.scored("unknown", 4), self.scored("red blue", 1)]
        for row in group:
            row.decisions = [Decision([], Completion("stop"))]
        report = metrics(group)
        for values in [report['pooled'], report['source_macro'], report['by_source']['fixture']]:
            self.assertAlmostEqual(values['read_cost'], 0.005)
            self.assertAlmostEqual(values['raw_read_cost'], 0.025)
            self.assertAlmostEqual(values['reward'], values['outcome_reward'] - values['read_cost'] + values['format_reward'])


if __name__ == '__main__':
    unittest.main()
