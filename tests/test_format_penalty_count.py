"""Cumulative format reward counts bad Reader actions, not individual errors."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import yaml

from agmem.core import CapacityError, Completion, Decision, InfrastructureError, load_config, trajectory_from_dict
from agmem.env import MemoryEnvironment
from agmem.rewards import RewardScorer, score_trajectory, validate_format_penalty_mode
from tests import test_core as fixtures


ROOT = Path(__file__).resolve().parents[1]


class FormatPenaltyCountTests(unittest.TestCase):
    setUp = fixtures.CoreTests.setUp
    tearDown = fixtures.CoreTests.tearDown
    reward = {"formula": "judge_cost_v2", "read_call_cost": .01, "format_penalty": .1,
              "format_penalty_mode": "per_decision", "likelihood": {"mode": "off"}}
    stop = {"ids": [], "stop": True}

    @staticmethod
    def read(*ids):
        return {"ids": list(ids), "stop": False}

    def run_actions(self, actions, **options):
        env = MemoryEnvironment(self.store, self.builder.version, id_protocol="strict_v2", **options)
        return env.run(self.example, fixtures.ScriptedPolicy(actions), 42)

    def score(self, row, judge_score=1, *, reward=None):
        row.prediction = "fixture answer"
        judge = SimpleNamespace(score=lambda *_: judge_score)
        return score_trajectory(self.example, row, self.reward if reward is None else reward, judge=judge)

    def test_repeated_actions_accumulate_for_correct_and_wrong_answers(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read(a), self.read(a), self.read(a), self.stop])
        for judge_score, expected in ((1, .77), (0, -1.23)):
            with self.subTest(judge=judge_score):
                scored = self.score(deepcopy(row), judge_score)
                self.assertEqual(scored.n_read_calls, 3)
                self.assertEqual(scored.format_violation_count, 2)
                self.assertTrue(scored.format_failure)
                self.assertAlmostEqual(scored.format_reward, -.2)
                self.assertAlmostEqual(scored.reward, expected)

    def test_multiple_labels_and_repeated_ids_in_one_action_only_count_once(self):
        a, b = [entry["id"] for entry in self.directory[:2]]
        row = self.run_actions([self.read(a), self.read(a, a, b, "unknown", "unknown"), self.stop])
        self.assertEqual(set(row.decisions[1].format_violations),
                         {"repeated_read", "duplicate_ids", "unknown_id", "too_many_ids"})
        scored = self.score(row)
        self.assertEqual(scored.format_violation_count, 1)
        self.assertAlmostEqual(scored.format_reward, -.1)
        self.assertAlmostEqual(scored.reward, .88)
        self.assertEqual([entry["id"] for entry in scored.evidence], [a, b])

    def test_final_over_budget_attempt_counts_once_but_normal_stop_is_free(self):
        a = self.directory[0]["id"]
        for final, expected_count, expected_reward in ((self.stop, 7, .22), (self.read(a), 8, .12)):
            with self.subTest(final=final):
                scored = self.score(self.run_actions([self.read(a)] * 8 + [final]))
                self.assertEqual(scored.n_read_calls, 8)
                self.assertEqual(scored.format_violation_count, expected_count)
                self.assertAlmostEqual(scored.reward, expected_reward)
                if not final["stop"]:
                    self.assertEqual(set(scored.decisions[-1].format_violations),
                                     {"repeated_read", "read_budget_exceeded"})
                    self.assertFalse(scored.decisions[-1].tool_result["read_call_consumed"])

    def test_malformed_json_counts_once_with_or_without_legacy_label(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read(a), self.read(a), "{"])
        self.assertEqual(row.termination_reason, "protocol_failure")
        for legacy in (False, True):
            case = deepcopy(row)
            if legacy:
                case.decisions[-1].format_violations.clear()
            scored = self.score(case)
            self.assertEqual(scored.format_violation_count, 2)
            self.assertAlmostEqual(scored.reward, .78)

    def test_capacity_refusal_does_not_count_as_format_failure_or_read_history(self):
        a = self.directory[0]["id"]
        attempts = []
        def capacity(*_):
            attempts.append(None)
            if len(attempts) == 1:
                raise CapacityError("fixture context full")
        scored = self.score(self.run_actions([self.read(a), self.read(a), self.stop], capacity=capacity))
        self.assertEqual(scored.n_read_calls, 2)
        self.assertEqual(scored.format_violation_count, 0)
        self.assertFalse(scored.format_failure)
        self.assertEqual(scored.format_reward, 0)
        self.assertAlmostEqual(scored.reward, .98)

    def test_valid_stop_ignores_even_duplicate_unknown_or_excess_ids(self):
        row = self.run_actions([{"ids": ["unknown"] * 6, "stop": True}])
        scored = self.score(row)
        self.assertEqual(scored.format_violation_count, 0)
        self.assertEqual(scored.n_read_calls, 0)
        self.assertEqual(scored.reward, 1)

    def test_unknown_empty_and_duplicate_actions_each_count_once(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read("unknown"), self.read(), self.read(a, a), self.stop])
        scored = self.score(row)
        self.assertEqual(scored.format_violation_count, 3)
        self.assertAlmostEqual(scored.format_reward, -.3)
        self.assertAlmostEqual(scored.reward, .67)

    def test_historical_tool_errors_count_but_capacity_and_service_errors_do_not(self):
        row = self.run_actions([self.stop])
        row.decisions = [Decision([], Completion("read"), self.read("A"),
                                  {"error": {"code": code}})
                         for code in ("unknown_or_unauthorized_id", "repeated_read", "max_read_calls",
                                      "context_capacity_exceeded", "http_failure", "cuda_oom")]
        scored = self.score(row)
        self.assertEqual(scored.format_violation_count, 3)
        self.assertAlmostEqual(scored.format_reward, -.3)
        with self.assertRaisesRegex(InfrastructureError, "judge unavailable"):
            score_trajectory(self.example, row, self.reward,
                             judge=SimpleNamespace(score=lambda *_: (_ for _ in ()).throw(
                                 InfrastructureError("judge unavailable"))))

    def test_current_nine_decision_budget_preserves_correctness_priority(self):
        worst_correct = self.score(self.run_actions([self.read("unknown")] * 9))
        best_wrong = self.score(self.run_actions([self.stop]), judge_score=0)
        self.assertEqual(worst_correct.n_read_calls, 8)
        self.assertEqual(worst_correct.format_violation_count, 9)
        self.assertAlmostEqual(worst_correct.reward, .02)
        self.assertEqual(best_wrong.reward, -1)
        self.assertGreater(worst_correct.reward, best_wrong.reward)

    def test_default_old_mode_keeps_one_penalty_and_provenance_shape(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read(a), self.read(a), self.read(a), self.stop])
        legacy = {k: v for k, v in self.reward.items() if k != "format_penalty_mode"}
        scored = self.score(row, reward=legacy)
        self.assertEqual(scored.format_violation_count, 2)
        self.assertAlmostEqual(scored.format_reward, -.1)
        self.assertAlmostEqual(scored.reward, .87)
        old = RewardScorer(legacy).provenance()
        self.assertEqual(set(old), {"formula", "task_output_policy", "judge", "likelihood"})
        self.assertEqual(old, RewardScorer({**legacy, "format_penalty_mode": "per_rollout"}).provenance())
        new = RewardScorer(self.reward).provenance()
        self.assertEqual(new["format_penalty_mode"], "per_decision")
        self.assertEqual(new["format_counting_version"], "reader_decision_v1")
        self.assertNotEqual(old, new)

    def test_serialization_keeps_count_and_missing_legacy_field_loads(self):
        a = self.directory[0]["id"]
        scored = self.score(self.run_actions([self.read(a), self.read(a), self.stop]))
        value = scored.to_dict()
        restored = trajectory_from_dict(value)
        self.assertEqual(restored.to_dict(), value)
        self.assertEqual(restored.format_violation_count, 1)
        del value["format_violation_count"]
        legacy = trajectory_from_dict(value)
        self.assertEqual(legacy.format_violation_count, 0)
        self.assertEqual(self.score(legacy).format_violation_count, 1)

    def test_invalid_modes_and_formula_combinations_fail_before_scoring(self):
        row = self.run_actions([self.stop])
        for mode in (None, True, 1, [], {}, "per_label"):
            bad = {**self.reward, "format_penalty_mode": mode}
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "format_penalty_mode"):
                    RewardScorer(bad)
                with self.assertRaisesRegex(ValueError, "format_penalty_mode"):
                    score_trajectory(self.example, row, bad)
        for formula in ("outcome_cost_v1", "outcome_gated_v2", "judge_gated_v1"):
            with self.subTest(formula=formula), self.assertRaisesRegex(ValueError, "requires judge_cost_v2"):
                validate_format_penalty_mode({**self.reward, "formula": formula})

    def test_config_loader_validates_mode_without_normalizing_old_configs(self):
        base = ROOT / "configs/train_dapo_reader_ids_v2.yaml"
        old = load_config(base)
        self.assertNotIn("format_penalty_mode", old["reward"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "count.yaml"
            for mode in ("per_decision", "per_rollout", "per_id", None, False, []):
                path.write_text(yaml.safe_dump({"extends": str(base), "reward": {"format_penalty_mode": mode}}))
                if mode in ("per_decision", "per_rollout", "per_id"):
                    self.assertEqual(load_config(path)["reward"]["format_penalty_mode"], mode)
                else:
                    with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "format_penalty_mode"):
                        load_config(path)


if __name__ == "__main__":
    unittest.main()
