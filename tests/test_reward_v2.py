from copy import deepcopy
import unittest
from types import SimpleNamespace

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.pipeline import metrics
from agmem.rewards import advantages, protocol_metrics, score_trajectory


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


class JudgeCostRewardTests(unittest.TestCase):
    config = {"formula": "judge_cost_v2", "read_call_cost": 0.01, "format_penalty": 0.1,
              "likelihood": {"mode": "off"}}

    def score(self, judge_score, calls, violations=(), prediction="red"):
        example = Example("locomo/train/0", "locomo", "train", "memory", "?", gold_answers=["blue"])
        row = Trajectory(example.sample_id, 0, example.memory_id, prediction=prediction, n_read_calls=calls,
                         decisions=[Decision([], Completion("read"), {"ids": ["A"], "stop": False},
                                             format_violations=list(violations))])
        return score_trajectory(example, row, self.config, judge=SimpleNamespace(score=lambda *_: judge_score))

    def test_every_actual_call_costs_regardless_of_binary_judge_and_f1(self):
        for judge_score in (0, 1):
            for calls in (0, 1, 8):
                with self.subTest(judge=judge_score, calls=calls):
                    # F1 deliberately disagrees with the Judge in both cases.
                    row = self.score(judge_score, calls, prediction="blue" if judge_score == 0 else "red")
                    self.assertEqual(row.answer_f1, 1 - judge_score)
                    self.assertEqual(row.outcome, judge_score)
                    self.assertEqual(row.outcome_reward, 2 * judge_score - 1)
                    self.assertAlmostEqual(row.read_cost, 0.01 * calls)
                    self.assertAlmostEqual(row.reward, 2 * judge_score - 1 - 0.01 * calls)

    def test_repeat_unknown_and_other_violations_pay_one_trajectory_penalty(self):
        row = self.score(0, 8, ("repeated_read", "unknown_id", "duplicate_ids"))
        row.decisions *= 3
        example = Example(row.sample_id, "locomo", "train", "memory", "?", gold_answers=["blue"])
        score_trajectory(example, row, self.config, judge=SimpleNamespace(score=lambda *_: 0))
        self.assertTrue(row.format_failure)
        self.assertEqual(row.format_reward, -0.1)
        self.assertAlmostEqual(row.reward, -1.18)

    def test_unknown_tool_error_is_a_format_failure_only_under_new_reward(self):
        example = Example("locomo/train/0", "locomo", "train", "m", "?", gold_answers=["blue"])
        row = Trajectory(example.sample_id, 0, "m", prediction="red", n_read_calls=1,
                         decisions=[Decision([], Completion("read"), {"ids": ["bad"], "stop": False},
                                             {"ok": False, "error": {"code": "unknown_or_unauthorized_id"}})])
        judge = SimpleNamespace(score=lambda *_: 0)
        legacy = score_trajectory(example, deepcopy(row), {**self.config, "formula": "judge_gated_v1"}, judge=judge)
        current = score_trajectory(example, row, self.config, judge=judge)
        self.assertEqual(legacy.reward, -1)
        self.assertFalse(legacy.format_failure)
        self.assertAlmostEqual(current.reward, -1.11)
        self.assertEqual(protocol_metrics([current])["format_decisions_unknown_id"], 1)

    def test_protocol_metrics_distinguish_repeats_new_evidence_and_budget_rejections(self):
        row = self.score(1, 3)
        def decision(ids, records, *, consumed=True, **metadata):
            return Decision([], Completion("read"), {"ids": ids, "stop": False},
                            {"records": [{"id": item} for item in records], "read_call_consumed": consumed, **metadata})
        row.decisions = [decision(["A"], ["A"], new_ids=["A"], already_read_ids=[], no_new_evidence=False),
                         decision(["A", "B"], ["B"], new_ids=["B"], already_read_ids=["A"], no_new_evidence=False),
                         decision(["A"], [], new_ids=[], already_read_ids=["A"], no_new_evidence=True),
                         decision(["A"], [], consumed=False, new_ids=[], already_read_ids=["A"], no_new_evidence=True)]
        report = protocol_metrics([row])
        self.assertEqual(report["repeated_read_rate"], 1)
        self.assertEqual(report["format_decisions_repeated_read"], 3)
        self.assertEqual(report["repeated_read_calls"], 2)
        self.assertAlmostEqual(report["repeated_read_call_rate"], 2 / 3)
        self.assertEqual(report["no_new_evidence_calls"], 1)
        self.assertAlmostEqual(report["no_new_evidence_call_rate"], 1 / 3)

    def test_legacy_protocol_metrics_can_recover_repeat_and_no_new_evidence_calls(self):
        row = self.score(1, 3)
        row.decisions = [Decision([], Completion("read"), {"ids": ids, "stop": False},
                                  {"records": [{"id": item} for item in ids]})
                         for ids in (["A"], ["A", "B"], ["A"])]
        report = protocol_metrics([row])
        self.assertEqual(report["repeated_read_calls"], 2)
        self.assertEqual(report["no_new_evidence_calls"], 1)

    def test_stop_ids_are_ignored_in_repeat_diagnostics(self):
        row = self.score(1, 1)
        row.decisions = [Decision([], Completion("read"), {"ids": ["A"], "stop": False},
                                  {"records": [{"id": "A"}]}),
                         Decision([], Completion("stop"), {"ids": ["A"], "stop": True})]
        report = protocol_metrics([row])
        self.assertEqual(report["repeated_read_rate"], 0)
        self.assertEqual(report["format_decisions_repeated_read"], 0)
        self.assertEqual(report["repeated_read_calls"], 0)


if __name__ == '__main__':
    unittest.main()
