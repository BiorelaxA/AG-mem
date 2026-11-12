"""Per-ID penalties through real retrieval and scoring, including failed reads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from agmem.core import CapacityError, load_config, trajectory_from_dict
from agmem.env import MemoryEnvironment
from agmem.four_gpu import configs
from agmem.pipeline import metrics
from agmem.resume import validate_resume_config
from agmem.rewards import RewardScorer, validate_format_penalty_mode
from tests import test_core as fixtures


ROOT = Path(__file__).resolve().parents[1]


class IDFormatCostTests(unittest.TestCase):
    setUp = fixtures.CoreTests.setUp
    tearDown = fixtures.CoreTests.tearDown
    stop = {"ids": [], "stop": True}

    @staticmethod
    def read(*ids):
        return {"ids": list(ids), "stop": False}

    def score(self, actions, *, judge=1, max_ids=1, config=None, **options):
        if config is None:
            config = load_config(ROOT / "configs/train_dapo_reader_loop_v3_query9010_id_cost.yaml")
        env = MemoryEnvironment(self.store, self.builder.version, max_ids=max_ids,
                                id_protocol="strict_v2", **options)
        row = env.run(self.example, fixtures.ScriptedPolicy(actions), 42)
        row.prediction = "fixture answer"
        return RewardScorer(config["reward"], max_ids=max_ids,
                            judge=SimpleNamespace(score=lambda *_: judge)).score(self.example, row)

    def test_default_half_point_penalty_preserves_id_and_protocol_counting(self):
        config, _ = configs()
        a, b, c = [e["id"] for e in self.directory[:3]]
        for judge in (0, 1):
            with self.subTest(judge=judge):
                # Excess IDs, repeat, unknown, empty request and malformed JSON.
                row = self.score([self.read(a, b, c), self.read(a, "unknown"), self.read(), "{"],
                                 judge=judge, config=config)
                self.assertEqual(row.format_violation_count, 6)
                self.assertEqual(row.format_reward, -3.0)
                self.assertEqual(row.n_read_calls, 3)
                self.assertAlmostEqual(row.reward, 2 * judge - 1 - .03 - 3.0)
        clean = self.score([self.read(a), self.stop], config=config)
        self.assertEqual(clean.format_reward, 0.0)
        self.assertAlmostEqual(clean.reward, .99)

    def test_each_bad_occurrence_counts_once_even_with_multiple_reasons(self):
        a, b, c = [e["id"] for e in self.directory[:3]]
        for ids, expected in (([a, b, c], 2), ([a, a, a], 2),
                              (["unknown", "unknown", b], 3), (["unknown"], 1)):
            with self.subTest(ids=ids):
                row = self.score([self.read(*ids), self.stop])
                self.assertEqual(row.format_violation_count, expected)
                self.assertAlmostEqual(row.reward, 1 - .01 - .1 * expected)
        row = self.score([self.read(a), self.read(a, a, b, "unknown", "unknown"), self.stop])
        self.assertEqual(set(row.decisions[1].format_violations),
                         {"repeated_read", "duplicate_ids", "unknown_id", "too_many_ids"})
        self.assertEqual(row.format_violation_count, 5)
        self.assertAlmostEqual(row.reward, .48)

    def test_duplicate_occurrences_after_first_count_even_within_capacity(self):
        a, b = [e["id"] for e in self.directory[:2]]
        row = self.score([self.read(a, a, b), self.stop], max_ids=4)
        self.assertEqual(row.format_violation_count, 1)
        self.assertEqual([e["id"] for e in row.evidence], [a, b])
        self.assertAlmostEqual(row.reward, .89)

    def test_json_schema_and_empty_penalties_remain_but_stop_is_free(self):
        for actions, calls, bad in ((["{"], 0, 1), (["{}"], 0, 1), ([self.read(), self.stop], 1, 1),
                                    ([{"ids": ["unknown"] * 3, "stop": True}], 0, 0)):
            with self.subTest(actions=actions):
                row = self.score(actions)
                self.assertEqual(row.format_failure, bool(bad))
                self.assertEqual(row.format_violation_count, bad)
                self.assertAlmostEqual(row.format_reward, -.1 * bad)
                self.assertEqual(row.n_read_calls, calls)
                self.assertAlmostEqual(row.reward, 1 - .01 * calls - .1 * bad)

    def test_over_budget_penalty_accumulates_with_id_penalty_without_extra_call_cost(self):
        a, b = [e["id"] for e in self.directory[:2]]
        for key, bad in ((b, 1), (a, 2), ("unknown", 2)):
            with self.subTest(key=key):
                row = self.score([self.read(a), self.read(key)], max_calls=1)
                self.assertEqual(row.termination_reason, "max_read_calls")
                self.assertIn("read_budget_exceeded", row.decisions[-1].format_violations)
                self.assertFalse(row.decisions[-1].tool_result["read_call_consumed"])
                self.assertEqual(row.n_read_calls, 1)
                self.assertEqual(row.format_violation_count, bad)
                self.assertAlmostEqual(row.reward, .99 - .1 * bad)

    def test_failed_or_unexecuted_read_does_not_make_id_repeated(self):
        a, b = [e["id"] for e in self.directory[:2]]
        attempts = []
        def capacity(*_):
            attempts.append(None)
            if len(attempts) == 1:
                raise CapacityError("context full")
        row = self.score([self.read(a), self.read(a), self.stop], capacity=capacity)
        self.assertEqual(row.format_violation_count, 0)
        self.assertAlmostEqual(row.reward, .98)
        row = self.score([self.read(a, b), self.read(b), self.stop])
        self.assertEqual(row.format_violation_count, 1)  # Only b's first, excess occurrence.
        self.assertEqual([e["id"] for e in row.evidence], [a, b])
        self.assertAlmostEqual(row.reward, .88)

    def test_original_protocol_penalties_accumulate_with_multiple_bad_ids(self):
        a, b, c = [e["id"] for e in self.directory[:3]]
        row = self.score([self.read(a, b, c), self.read(), "{"])
        self.assertEqual(row.format_violation_count, 4)  # Two excess IDs + empty + JSON.
        self.assertAlmostEqual(row.format_reward, -.4)
        self.assertAlmostEqual(row.reward, .58)
        row = self.score([self.read(a), self.read(a, a, b, "unknown")], max_calls=1)
        self.assertEqual(row.format_violation_count, 5)  # Four bad positions + exhausted budget.
        self.assertAlmostEqual(row.format_reward, -.5)
        self.assertAlmostEqual(row.reward, .49)

    def test_cost_is_charged_for_correct_and_wrong_answers_including_bad_calls(self):
        a = self.directory[0]["id"]
        for judge in (0, 1):
            with self.subTest(judge=judge):
                row = self.score([self.read(a), self.read(a), self.read("unknown"), self.read(), self.stop], judge=judge)
                self.assertEqual(row.n_read_calls, 4)
                self.assertEqual(row.raw_read_cost, .04)
                self.assertEqual(row.read_cost, .04)
                self.assertEqual(row.format_violation_count, 3)
                self.assertAlmostEqual(row.reward, 2 * judge - 1 - .04 - .3)
                self.assertEqual(trajectory_from_dict(row.to_dict()).to_dict(), row.to_dict())
                report = metrics([row])["pooled"]
                self.assertEqual(report["format_violation_count"], 3)
                self.assertEqual(report["read_cost"], .04)

    def test_new_protocol_has_distinct_provenance_and_cannot_resume_old_reward(self):
        old = load_config(ROOT / "configs/train_dapo_reader_loop_v3_query9010.yaml")
        new = load_config(ROOT / "configs/train_dapo_reader_loop_v3_query9010_id_cost.yaml")
        self.assertEqual({**old["data"], "sources": ["locomo"]}, new["data"])
        self.assertEqual(old["reward"]["format_penalty_mode"], "per_decision")
        self.assertEqual(old["reward"]["read_call_cost"], 0)
        self.assertNotEqual(old["training"]["output_dir"], new["training"]["output_dir"])
        self.assertEqual(RewardScorer(new["reward"]).provenance()["format_counting_version"], "reader_id_with_protocol_v2")
        changed = deepcopy(old)
        changed["reward"] = new["reward"]
        with self.assertRaisesRegex(ValueError, "Resume configuration changed: reward"):
            validate_resume_config(old, changed)
        with self.assertRaisesRegex(ValueError, "requires judge_cost_v2"):
            validate_format_penalty_mode({"formula": "outcome_cost_v1", "format_penalty_mode": "per_id"})


if __name__ == "__main__":
    unittest.main()
