"""Exercise the one-ID boundary across Reader state, execution and scoring."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from agmem.core import Completion, Decision, Trajectory, load_config
from agmem.env import MemoryEnvironment
from agmem.pipeline import make_environment, make_reward_scorer
from agmem.prompts import configured_reader_input_contract
from agmem.resume import validate_reader_input_contract, validate_resume_config
from agmem.rewards import RewardScorer
from tests import test_core as fixtures


ROOT = Path(__file__).resolve().parents[1]


class RetrievalMaxOneTests(unittest.TestCase):
    tearDown = fixtures.CoreTests.tearDown
    stop = {"ids": [], "stop": True}

    def setUp(self):
        fixtures.CoreTests.setUp(self)
        self.train = load_config(ROOT / "configs/train_dapo_max1_v2.yaml")
        self.ids = [entry["id"] for entry in self.directory]

    @staticmethod
    def read(*ids):
        return {"ids": list(ids), "stop": False}

    def run_actions(self, actions):
        retrieval = self.train["retrieval"]
        env = MemoryEnvironment(self.store, self.builder.version,
                                max_ids=retrieval["max_ids_per_call"],
                                max_calls=retrieval["max_read_calls"],
                                id_protocol=retrieval["id_protocol"],
                                prompt_version=self.train["reader"]["prompt_version"])
        self.policy = fixtures.ScriptedPolicy(actions)
        row = env.run(self.example, self.policy, 42)
        row.prediction = "fixture answer"
        scorer = RewardScorer(self.train["reward"], max_ids=retrieval["max_ids_per_call"],
                              judge=SimpleNamespace(score=lambda *_: 1))
        return scorer.score(self.example, row)

    def test_one_id_is_legal_and_stop_still_ignores_its_id_list(self):
        row = self.run_actions([self.read(self.ids[0]), {"ids": self.ids[:3], "stop": True}])
        self.assertEqual([item["id"] for item in row.evidence], self.ids[:1])
        self.assertEqual(row.n_read_calls, 1)
        self.assertEqual(row.format_violation_count, 0)
        self.assertAlmostEqual(row.reward, .99)
        system = self.policy.messages[0][0]["content"]
        self.assertRegex(system, r"(?:exactly 1|1 to 1)")
        self.assertNotIn("1 to 2", system)
        self.assertNotIn("first 2 IDs", system)

    def test_two_or_more_ids_read_only_original_first_and_count_once(self):
        for count in (2, 3, 4):
            with self.subTest(count=count), patch.object(self.store, "record", wraps=self.store.record) as fetch:
                row = self.run_actions([self.read(*self.ids[:count]), self.stop])
                step = row.decisions[0]
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(step.action["ids"], self.ids[:count])
                self.assertEqual(step.tool_result["selected_ids"], self.ids[:1])
                self.assertTrue(step.tool_result["ids_truncated"])
                self.assertEqual(step.format_violations, ["too_many_ids"])
                state = json.loads(self.policy.messages[1][1]["content"])
                self.assertEqual(state["read_ids"], self.ids[:1])
                self.assertEqual([item["id"] for item in row.evidence], self.ids[:1])
                self.assertEqual(row.format_violation_count, 1)
                self.assertAlmostEqual(row.format_reward, -.1)
                self.assertAlmostEqual(row.reward, .89)

    def test_ignored_second_id_is_available_for_the_next_call(self):
        a, b = self.ids[:2]
        row = self.run_actions([self.read(a, b), self.read(b), self.stop])
        before = json.loads(self.policy.messages[1][1]["content"])
        after = json.loads(self.policy.messages[2][1]["content"])
        self.assertEqual(before["read_ids"], [a])
        self.assertEqual(after["read_ids"], [a, b])
        self.assertEqual(row.decisions[1].tool_result["new_ids"], [b])
        self.assertEqual(row.decisions[1].format_violations, [])
        self.assertEqual(row.format_violation_count, 1)
        self.assertAlmostEqual(row.reward, .88)

    def test_unknown_first_id_does_not_backfill_from_valid_second_id(self):
        with patch.object(self.store, "record", wraps=self.store.record) as fetch:
            row = self.run_actions([self.read("unknown", self.ids[0]), self.stop])
        self.assertEqual(fetch.call_count, 0)
        self.assertEqual(row.evidence, [])
        self.assertEqual(row.decisions[0].tool_result["selected_ids"], [])
        self.assertEqual(row.decisions[0].tool_result["invalid_ids"], ["unknown"])
        self.assertEqual(set(row.decisions[0].format_violations), {"too_many_ids", "unknown_id"})
        self.assertEqual(row.format_violation_count, 1)
        self.assertAlmostEqual(row.format_reward, -.1)

    def test_penalty_accumulates_per_bad_decision_not_per_flag(self):
        a, b, c = self.ids[:3]
        row = self.run_actions([self.read(a), self.read(a, a, "unknown", b),
                                self.read(b, c), self.stop])
        self.assertEqual(set(row.decisions[1].format_violations),
                         {"too_many_ids", "duplicate_ids", "unknown_id", "repeated_read"})
        self.assertEqual(row.decisions[1].tool_result["new_ids"], [])
        self.assertEqual([item["id"] for item in row.evidence], [a, b])
        self.assertEqual(row.n_read_calls, 3)
        self.assertEqual(row.format_violation_count, 2)
        self.assertAlmostEqual(row.format_reward, -.2)
        self.assertAlmostEqual(row.reward, .77)

    def test_runtime_factories_and_raw_rescoring_use_one_id_limit(self):
        policy = SimpleNamespace(check_capacity=lambda _: None)
        judge = SimpleNamespace(score=lambda *_: 1)
        with patch("agmem.pipeline.validate_directory_version", return_value=self.builder.version), \
                patch("agmem.policy.load_tokenizer", return_value=object()), \
                patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture"}), \
                patch("agmem.judges.JudgeScorer", return_value=judge), \
                patch("agmem.likelihood.ReferenceLikelihoodScorer", return_value=None):
            env = make_environment(self.train, self.store, policy)
            scorer = make_reward_scorer(self.train, self.store)
        self.assertEqual(env.max_ids, 1)
        self.assertEqual(scorer.max_ids, 1)
        row = Trajectory(self.example.sample_id, 42, self.example.memory_id,
                         prediction="fixture answer", n_read_calls=1, termination_reason="stop",
                         decisions=[Decision([], Completion("read"), self.read(*self.ids[:2]))])
        scorer.score(self.example, row)
        self.assertEqual(row.format_violation_count, 1)
        self.assertAlmostEqual(row.reward, .89)

    def test_only_limit_and_output_change_and_old_checkpoint_cannot_resume(self):
        previous = load_config(ROOT / "configs/train_dapo_max2_v2.yaml")
        self.assertEqual(previous["retrieval"]["max_ids_per_call"], 2)
        switched = deepcopy(previous)
        switched["retrieval"]["max_ids_per_call"] = 1
        old_contract = configured_reader_input_contract(previous)
        self.assertNotEqual(old_contract, configured_reader_input_contract(self.train))
        with self.assertRaisesRegex(ValueError, "Resume configuration changed: retrieval"):
            validate_resume_config(previous, switched)
        with self.assertRaisesRegex(ValueError, "Reader input contract changed"):
            validate_reader_input_contract({"config": previous, "reader_input_contract": old_contract}, switched)
        switched["training"]["output_dir"] = self.train["training"]["output_dir"]
        self.assertEqual(switched, self.train)


if __name__ == "__main__":
    unittest.main()
