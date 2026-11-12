"""The directory-process recipe binds prompts, limits and legacy terminal rewards."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from agmem.core import Example, digest, load_config
from agmem.dapo_sampling import assess_group, validate_dapo_config
from agmem.env import MemoryEnvironment
from agmem.prompts import (HISTORY_ANSWER_TEMPLATE, configured_reader_input_contract,
                           reader_messages, task_input_contract, task_messages)
from agmem.rewards import RewardScorer
from agmem.services import TaskAgent
from tests import test_core as fixtures
from tests.test_dapo_all_correct_format import format_group
from tests.test_dapo_sampling import examples


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/train_dapo_directory_process_v1.yaml"


class DirectoryProcessConfigTests(unittest.TestCase):
    def test_recipe_keeps_terminal_reward_and_accepts_correct_process_groups_with_fresh_batches(self):
        config = load_config(RECIPE)
        previous = load_config(ROOT / "configs/train_dapo_reader_loop_v3_query9010_format05.yaml")
        validate_dapo_config(config)
        self.assertEqual(config["reward"], previous["reward"])
        self.assertEqual(config["training"]["algorithm"], "dapo")
        self.assertEqual(config["training"]["filter_metric"], "exclude_all_wrong")
        self.assertTrue(config["training"]["process_reward"]["enabled"])
        self.assertEqual(config["training"]["epochs"], 3)
        self.assertEqual(config["training"]["learning_rate"], previous["training"]["learning_rate"])
        self.assertEqual(config["training"]["group_size"], 8)
        self.assertEqual(config["training"]["target_groups"], config["training"]["mini_batch_groups"])
        self.assertEqual(config["training"]["mini_batch_groups"], 8)
        self.assertEqual(config["training"]["validation_every"], 4)
        self.assertEqual(config["retrieval"], {"max_ids_per_call": 2, "max_read_calls": 8,
                                                "id_protocol": "strict_v2"})
        self.assertEqual(config["task_agent"]["prompt_version"], "history_answer_evidence_v2")
        self.assertEqual(config["reader"]["prompt_version"], "interaction_history_v5")
        self.assertEqual(config["training"]["checkpoint_every"], 0)
        self.assertNotEqual(config["training"]["output_dir"], previous["training"]["output_dir"])
        self.assertNotEqual(configured_reader_input_contract(config), configured_reader_input_contract(previous))
        example = examples(1)[0]
        self.assertFalse(assess_group(format_group(example, judges=(0, 0)), config)["accepted"])
        self.assertTrue(assess_group(format_group(example, judges=(0, 1)), config)["accepted"])
        self.assertTrue(assess_group(format_group(example), config)["accepted"])
        self.assertTrue(assess_group(format_group(example, counts=(0, 0), read_calls=(1, 2)), config)["accepted"])
        constant = format_group(example, counts=(0, 0))
        self.assertTrue(assess_group(constant, config)["accepted"])
        self.assertEqual([row.advantage for row in constant], [0.0, 0.0])

    def test_query_only_task_template_excludes_public_context_and_internal_fields(self):
        example = Example("fixture/train/1", "fixture", "train", "memory", "Which city?",
                          observation="PRIVATE_OBSERVATION", options={"A": "PRIVATE_OPTION"},
                          gold_answers=["SECRET_GOLD"])
        evidence = [{"id": "PRIVATE_ID", "content": "PRIVATE_SUMMARY", "time": "2023-05-25",
                     "raw_text": "Ada: I visited Paris."}]
        expected = HISTORY_ANSWER_TEMPLATE.format(
            history_chats="Session 1\nDate: 2023-05-25\nAda: I visited Paris.", question=example.query)
        messages = task_messages(example, evidence, "history_answer_query_v1")
        self.assertEqual(messages, [{"role": "user", "content": expected}])
        for forbidden in ("PRIVATE_OBSERVATION", "PRIVATE_OPTION", "SECRET_GOLD", "PRIVATE_ID", "PRIVATE_SUMMARY"):
            self.assertNotIn(forbidden, expected)
        old_prompt = task_messages(example, evidence, "history_answer_v1")[0]["content"]
        self.assertIn("PRIVATE_OBSERVATION", old_prompt)
        self.assertIn("PRIVATE_OPTION", old_prompt)
        new_contract = task_input_contract("history_answer_query_v1")
        self.assertEqual(new_contract["question_rendering"], "query_only_v1")
        self.assertNotEqual(new_contract, task_input_contract("history_answer_v1"))
        self.assertEqual(new_contract["task_prompt_sha256"], digest(HISTORY_ANSWER_TEMPLATE))

    def test_query_only_task_agent_scores_inner_answer_and_keeps_complete_raw_output(self):
        calls = []
        def complete(messages, max_tokens, *, validator, metadata, require_stop):
            calls.append((messages, max_tokens, require_stop))
            raw = "<answer>Paris</answer>"
            validator(raw)
            metadata.update(finish_reason="stop", truncated=False)
            return raw
        agent = TaskAgent(SimpleNamespace(complete=complete), 2048, "history_answer_query_v1")
        example = Example("fixture/train/1", "fixture", "train", "memory", "Which city?",
                          observation="INTERNAL_CONTEXT", gold_answers=["SECRET_GOLD"])
        metadata = {}
        answer, task_hash = agent.answer(example, [], metadata=metadata)
        self.assertEqual(answer, "Paris")
        self.assertEqual(metadata["raw_text"], "<answer>Paris</answer>")
        self.assertEqual(metadata["answer_parser"], "answer_tag_v1")
        self.assertEqual(task_hash, digest(calls[0][0]))
        self.assertEqual(calls[0][1:], (2048, True))
        self.assertNotIn("INTERNAL_CONTEXT", calls[0][0][0]["content"])

    def test_reader_v4_includes_full_interactions_and_correct_task_visibility(self):
        example = examples(1)[0]
        history = [{"reader_output": '{"ids":["A"],"stop":false}',
                    "action": {"ids": ["A"], "stop": False},
                    "tool_result": {"records": [{"id": "A", "raw_text": "a clue"}],
                                    "read_call_consumed": True}}]
        original = deepcopy(history)
        arguments = {"max_ids": 2, "max_calls": 8, "read_calls_used": 1,
                     "interaction_history": history}
        messages = reader_messages(example, [], [{"id": "A", "raw_text": "a clue"}],
                                   prompt_version="interaction_history_v4", **arguments)
        state = json.loads(messages[1]["content"])
        self.assertEqual(state["interaction_history"], original)
        self.assertEqual(history, original)
        self.assertEqual(state["read_ids"], ["A"])
        self.assertEqual(state["max_ids_per_call"], 2)
        self.assertEqual(state["read_calls_remaining"], 7)
        self.assertIn("Select 1 to 2 distinct directory IDs", messages[0]["content"])
        self.assertIn("up to 8 READ calls", messages[0]["content"])
        self.assertIn("does not see the directory summaries", messages[0]["content"])
        legacy = reader_messages(example, [], [], prompt_version="evidence_loop_v3", **arguments)
        self.assertNotIn("interaction_history", json.loads(legacy[1]["content"]))
        empty = reader_messages(example, [], [], prompt_version="interaction_history_v4")
        self.assertEqual(json.loads(empty[1]["content"])["interaction_history"], [])
        with self.assertRaisesRegex(ValueError, "interaction_history"):
            reader_messages(example, [], [], prompt_version="interaction_history_v4", interaction_history={})


class DirectoryProcessFormatTests(unittest.TestCase):
    setUp = fixtures.CoreTests.setUp
    tearDown = fixtures.CoreTests.tearDown
    stop = {"ids": [], "stop": True}

    def score(self, actions, *, judge=1):
        config = load_config(RECIPE)
        policy = fixtures.ScriptedPolicy(actions)
        env = MemoryEnvironment(self.store, self.builder.version, max_ids=2, max_calls=8,
                                id_protocol="strict_v2", prompt_version="interaction_history_v4")
        row = env.run(self.example, policy, 42)
        row.prediction = "fixture answer"
        scorer = RewardScorer(config["reward"], max_ids=2, judge=SimpleNamespace(score=lambda *_: judge))
        return scorer.score(self.example, row)

    @staticmethod
    def read(*ids):
        return {"ids": list(ids), "stop": False}

    def test_two_ids_are_free_of_format_penalties_and_excess_ids_count_individually(self):
        keys = [entry["id"] for entry in self.directory]
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                row = self.score([self.read(*keys[:count]), self.stop])
                self.assertEqual([record["id"] for record in row.evidence], keys[:min(count, 2)])
                self.assertEqual(row.format_violation_count, max(0, count - 2))
                self.assertAlmostEqual(row.reward, .99 - .5 * max(0, count - 2))

    def test_repeated_duplicate_unknown_and_overlimit_positions_accumulate_once_each(self):
        a, b, c = [entry["id"] for entry in self.directory[:3]]
        # Second action: repeated A, duplicate A, excess C, excess/unknown X.
        for judge in (0, 1):
            row = self.score([self.read(a, b), self.read(a, a, c, "unknown"), self.stop], judge=judge)
            self.assertEqual(row.format_violation_count, 4)
            self.assertEqual(row.format_reward, -2)
            self.assertEqual(row.n_read_calls, 2)
            self.assertAlmostEqual(row.reward, 2 * judge - 1 - .02 - 2)

    def test_ninth_read_attempt_is_penalized_without_ninth_call_cost(self):
        a = self.directory[0]["id"]
        row = self.score([self.read(a)] * 9)
        self.assertEqual(row.n_read_calls, 8)
        self.assertEqual(row.termination_reason, "max_read_calls")
        self.assertFalse(row.decisions[-1].tool_result["read_call_consumed"])
        # Eight repeated occurrences (including the unexecuted ninth request),
        # plus the non-ID protocol violation for exceeding the call budget.
        self.assertEqual(row.format_violation_count, 9)
        self.assertAlmostEqual(row.read_cost, .08)
        self.assertAlmostEqual(row.reward, 1 - .08 - 4.5)


if __name__ == "__main__":
    unittest.main()
