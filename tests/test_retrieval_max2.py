"""Keep the two-ID Reader contract, execution and cumulative scoring aligned."""
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


class RetrievalMaxTwoTests(unittest.TestCase):
    tearDown = fixtures.CoreTests.tearDown
    stop = {"ids": [], "stop": True}

    def setUp(self):
        fixtures.CoreTests.setUp(self)
        self.train = load_config(ROOT / "configs/train_dapo_max2_v2.yaml")
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
        return env.run(self.example, self.policy, 42)

    def score(self, row):
        row.prediction = "fixture answer"
        scorer = RewardScorer(self.train["reward"], max_ids=self.train["retrieval"]["max_ids_per_call"],
                              judge=SimpleNamespace(score=lambda *_: 1))
        return scorer.score(self.example, row)

    def test_one_or_two_new_ids_are_legal_and_prompt_uses_the_same_limit(self):
        for count in (1, 2):
            with self.subTest(count=count):
                row = self.score(self.run_actions([self.read(*self.ids[:count]), self.stop]))
                self.assertEqual([item["id"] for item in row.evidence], self.ids[:count])
                self.assertEqual(row.format_violation_count, 0)
                self.assertEqual(row.format_reward, 0)
                self.assertAlmostEqual(row.reward, .99)
                system = self.policy.messages[0][0]["content"]
                self.assertIn("Select 1 to 2 distinct directory IDs", system)
                self.assertIn("only the first 2 IDs", system)
                self.assertNotIn("first 4 IDs", system)

    def test_three_or_four_requested_ids_read_only_first_two_and_count_once(self):
        for count in (3, 4):
            with self.subTest(count=count), patch.object(self.store, "record", wraps=self.store.record) as fetch:
                row = self.score(self.run_actions([self.read(*self.ids[:count]), self.stop]))
                step = row.decisions[0]
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(step.action["ids"], self.ids[:count])
                self.assertEqual(step.tool_result["selected_ids"], self.ids[:2])
                self.assertTrue(step.tool_result["ids_truncated"])
                self.assertEqual(step.format_violations, ["too_many_ids"])
                self.assertEqual([item["id"] for item in row.evidence], self.ids[:2])
                state = json.loads(self.policy.messages[1][1]["content"])
                self.assertEqual(state["read_ids"], self.ids[:2])
                self.assertEqual(row.n_read_calls, 1)
                self.assertEqual(row.format_violation_count, 1)
                self.assertAlmostEqual(row.format_reward, -.1)
                self.assertAlmostEqual(row.reward, .89)

    def test_ignored_third_id_remains_unread_and_can_be_retrieved_next(self):
        a, b, c = self.ids[:3]
        row = self.score(self.run_actions([self.read(a, b, c), self.read(c), self.stop]))
        before_retry = json.loads(self.policy.messages[1][1]["content"])
        after_retry = json.loads(self.policy.messages[2][1]["content"])
        self.assertEqual(before_retry["read_ids"], [a, b])
        self.assertEqual(after_retry["read_ids"], [a, b, c])
        self.assertEqual(row.decisions[1].tool_result["already_read_ids"], [])
        self.assertEqual(row.decisions[1].tool_result["new_ids"], [c])
        self.assertEqual(row.decisions[1].format_violations, [])
        self.assertEqual(row.format_violation_count, 1)
        self.assertAlmostEqual(row.reward, .88)

    def test_two_over_limit_actions_accumulate_two_penalties(self):
        row = self.score(self.run_actions([self.read(*self.ids[:3]), self.read(*self.ids[3:6]), self.stop]))
        self.assertEqual([item["id"] for item in row.evidence], self.ids[:2] + self.ids[3:5])
        self.assertEqual(row.n_read_calls, 2)
        self.assertEqual(row.format_violation_count, 2)
        self.assertAlmostEqual(row.format_reward, -.2)
        self.assertAlmostEqual(row.reward, .78)

    def test_multiple_flags_in_one_action_do_not_multiply_its_penalty(self):
        a, b = self.ids[:2]
        row = self.score(self.run_actions([self.read(a), self.read(a, a, "unknown", b),
                                          self.read(a), self.stop]))
        self.assertEqual(set(row.decisions[1].format_violations),
                         {"too_many_ids", "duplicate_ids", "unknown_id", "repeated_read"})
        # Execution takes the original two-ID prefix; ignored entries do not fill
        # slots occupied by repeated or duplicate IDs.
        self.assertEqual(row.decisions[1].tool_result["selected_ids"], [a])
        self.assertEqual(row.decisions[1].tool_result["new_ids"], [])
        self.assertEqual([item["id"] for item in row.evidence], [a])
        self.assertEqual(row.format_violation_count, 2)
        self.assertAlmostEqual(row.format_reward, -.2)
        self.assertAlmostEqual(row.reward, .77)

    def test_valid_stop_keeps_existing_ignore_ids_semantics(self):
        row = self.score(self.run_actions([self.read(*self.ids[:2]),
                                          {"ids": self.ids[:4], "stop": True}]))
        self.assertEqual(row.termination_reason, "stop")
        self.assertEqual(row.n_read_calls, 1)
        self.assertEqual(row.format_violation_count, 0)
        self.assertAlmostEqual(row.reward, .99)

    def test_runtime_factories_pass_two_to_environment_and_reward_scorer(self):
        policy = SimpleNamespace(check_capacity=lambda _: None)
        judge = SimpleNamespace(score=lambda *_: 1)
        with patch("agmem.pipeline.validate_directory_version", return_value=self.builder.version), \
                patch("agmem.policy.load_tokenizer", return_value=object()), \
                patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture"}), \
                patch("agmem.judges.JudgeScorer", return_value=judge), \
                patch("agmem.likelihood.ReferenceLikelihoodScorer", return_value=None):
            env = make_environment(self.train, self.store, policy)
            scorer = make_reward_scorer(self.train, self.store)
        self.assertEqual(env.max_ids, 2)
        self.assertEqual(scorer.max_ids, 2)
        self.assertEqual(env.prompt_version, "evidence_ids_v2")
        # Re-scoring a raw action with no environment violation labels must use
        # the configured threshold too, rather than the helper's legacy default 4.
        row = Trajectory(self.example.sample_id, 42, self.example.memory_id,
                         prediction="fixture answer", n_read_calls=1, termination_reason="stop",
                         decisions=[Decision([], Completion("read"), self.read(*self.ids[:3]))])
        scorer.score(self.example, row)
        self.assertEqual(row.format_violation_count, 1)
        self.assertAlmostEqual(row.reward, .89)

    def test_limit_switch_preserves_old_recipe_and_rejects_ordinary_resume(self):
        previous = load_config(ROOT / "configs/train_dapo_reader_ids_format_count_v2.yaml")
        self.assertEqual(previous["retrieval"]["max_ids_per_call"], 4)
        self.assertEqual(self.train["retrieval"]["max_ids_per_call"], 2)
        self.assertEqual(self.train["reward"], previous["reward"])
        old_contract = configured_reader_input_contract(previous)
        self.assertNotEqual(old_contract, configured_reader_input_contract(self.train))
        switched = deepcopy(previous)
        switched["retrieval"]["max_ids_per_call"] = 2
        with self.assertRaisesRegex(ValueError, "Resume configuration changed: retrieval"):
            validate_resume_config(previous, switched)
        with self.assertRaisesRegex(ValueError, "Reader input contract changed"):
            validate_reader_input_contract({"config": previous, "reader_input_contract": old_contract}, switched)


if __name__ == "__main__":
    unittest.main()
