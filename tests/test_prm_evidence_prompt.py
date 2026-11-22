"""Public context, selected summaries and full Reader interaction history stay aligned."""
from copy import deepcopy
import unittest

from agmem.core import CapacityError, Example, digest
from agmem.env import MemoryEnvironment
from agmem.process_rewards import attribution_settings
from agmem.prompts import (HISTORY_EVIDENCE_ANSWER_TEMPLATE, reader_input_contract,
                           reader_messages, task_input_contract, task_messages)
from tests.test_reader_evidence_state import MemoryStore, ScriptedReader, example, state


TASK_VERSION = "history_answer_evidence_v2"
READER_VERSION = "interaction_history_v5"
STOP = {"ids": [], "stop": True}


def read(*ids):
    return {"ids": list(ids), "stop": False}


class EvidenceTaskPromptTests(unittest.TestCase):
    def test_task_includes_public_context_and_selected_summary_with_original_text_only(self):
        item = Example("fixture/train/0", "fixture", "train", "INTERNAL_MEMORY", "Where did Ada travel?",
                       observation="Current date: 2026-01-01", options={"A": "Paris", "B": "Rome"},
                       gold_answers=["PRIVATE_GOLD"], judge_metadata={"private": "PRIVATE_JUDGE"})
        evidence = [{"id": "INTERNAL_ID", "source_session_id": "INTERNAL_SOURCE", "time": "2025-12-01",
                     "content": "Ada traveled to Paris.", "raw_text": "Ada: I visited Paris last month.",
                     "session_hash": "INTERNAL_HASH", "extra": "UNREAD_DIRECTORY"}]
        before = deepcopy(evidence)
        result = task_messages(item, evidence, TASK_VERSION)
        expected = HISTORY_EVIDENCE_ANSWER_TEMPLATE.format(
            history_chats="Session 1\nDate: 2025-12-01\nDirectory summary: Ada traveled to Paris.\n"
                          "Original session:\nAda: I visited Paris last month.",
            question='Where did Ada travel?\n\nPublic observation: Current date: 2026-01-01'
                     '\n\nOptions: {"A":"Paris","B":"Rome"}')
        self.assertEqual(result, [{"role": "user", "content": expected}])
        for secret in ("PRIVATE_GOLD", "PRIVATE_JUDGE", "INTERNAL_MEMORY", "INTERNAL_ID",
                       "INTERNAL_SOURCE", "INTERNAL_HASH", "UNREAD_DIRECTORY"):
            self.assertNotIn(secret, expected)
        self.assertEqual(evidence, before)
        self.assertIn("Keep the answer brief while preserving all information", expected)
        self.assertIn("Prefer the original text", expected)
        self.assertIn("Do not include analysis", expected)

    def test_empty_evidence_keeps_public_observation_and_missing_text_is_rejected(self):
        item = example()
        text = task_messages(item, [], TASK_VERSION)[0]["content"]
        self.assertIn("public-observation", text)
        self.assertIn("History Chats: \n", text)
        self.assertNotIn("Directory summary:", text)
        for bad in ({"content": "summary"}, {"raw_text": "original"},
                    {"raw_text": "original", "content": None}):
            with self.subTest(record=bad), self.assertRaises(ValueError):
                task_messages(item, [bad], TASK_VERSION)

    def test_new_contract_versions_do_not_relabel_existing_runs(self):
        new = task_input_contract(TASK_VERSION)
        self.assertEqual(new["answer_parser"], "answer_tag_partial_v1")
        self.assertEqual(new["completion_policy"], "score_tagged_truncated_v1")
        self.assertEqual(new["task_prompt_sha256"], digest(HISTORY_EVIDENCE_ANSWER_TEMPLATE))
        self.assertEqual(new["history_rendering"], "session_time_summary_raw_text_v1")
        old = task_input_contract("history_answer_query_v1")
        self.assertEqual(old["task_prompt_sha256"], "1a1ba3e945d20b0782afd5e98ab7fb07de4d75ee79c51f83eaf50e18cd9fa192")
        self.assertEqual(old["completion_policy"], "require_stop_v1")
        previous = reader_input_contract("interaction_history_v4", max_ids=2, max_calls=8, id_protocol="strict_v2")
        self.assertEqual(previous["reader_prompt_sha256"], "6410c6719fca6696c0fad396fb33d81dc2d87736f3f9b90338b937b3df002d37")
        current = reader_input_contract(READER_VERSION, max_ids=2, max_calls=8, id_protocol="strict_v2")
        self.assertNotEqual(current["reader_prompt_sha256"], previous["reader_prompt_sha256"])

    def test_reader_v5_keeps_history_and_describes_actual_task_visibility(self):
        interaction = [{"reader_output": '{"ids":["A"],"stop":false}',
                        "action": read("A"), "tool_result": {"records": []}}]
        message = reader_messages(example(), [], [], prompt_version=READER_VERSION,
                                  id_protocol="strict_v2", max_ids=2, max_calls=8,
                                  read_calls_used=1, interaction_history=interaction)
        self.assertIn("public observation and options", message[0]["content"])
        self.assertIn("their timestamps and directory summaries", message[0]["content"])
        self.assertNotIn("does not see the directory summaries", message[0]["content"])
        value = state(message)
        self.assertEqual(value["interaction_history"], interaction)
        self.assertEqual(value["read_calls_remaining"], 7)
        self.assertEqual(value["read_ids"], [])
        with self.assertRaises(ValueError):
            reader_messages(example(), [], [], prompt_version=READER_VERSION, max_calls=-1)


class EvidenceProcessEnvironmentTests(unittest.TestCase):
    def environment(self, store, **kwargs):
        return MemoryEnvironment(store, "directory-v1", max_ids=2, max_calls=8,
                                 id_protocol="strict_v2", prompt_version=READER_VERSION,
                                 task_prompt_version=TASK_VERSION, **kwargs)

    def test_causal_history_and_capacity_preview_include_new_task_evidence(self):
        store, item = MemoryStore(), example()
        a, b = [row["id"] for row in store.memory(item.memory_id)]
        policy = ScriptedReader([read(a), read(a, b), STOP])
        previews, tasks = [], []
        def capacity(reader_input, task_input):
            previews.append(deepcopy(reader_input))
            tasks.append(deepcopy(task_input))
        row = self.environment(store, capacity=capacity).run(item, policy, 17)
        self.assertEqual(previews, policy.messages[1:])
        self.assertEqual(store.reads, [a, b])
        self.assertEqual(len(state(policy.messages[-1])["interaction_history"]), 2)
        self.assertEqual(tasks[-1], task_messages(item, row.evidence, TASK_VERSION))
        self.assertIn("summary-" + a, tasks[0][0]["content"])
        self.assertIn("original-body-for-" + a, tasks[0][0]["content"])
        self.assertNotIn("summary-" + b, tasks[0][0]["content"])
        self.assertIn("public-observation", tasks[0][0]["content"])
        self.assertNotIn("private-gold", str(tasks))

    def test_capacity_rejection_preserves_attempt_without_leaking_rejected_evidence(self):
        store, item = MemoryStore(), example()
        a, b = [row["id"] for row in store.memory(item.memory_id)]
        policy = ScriptedReader([read(a), read(b), STOP])
        def capacity(reader_input, task_input):
            if a in state(reader_input)["read_ids"]:
                raise CapacityError("first session rejected")
        row = self.environment(store, capacity=capacity).run(item, policy, 17)
        self.assertEqual([record["id"] for record in row.evidence], [b])
        rejected = state(policy.messages[1])["interaction_history"][0]["tool_result"]
        self.assertEqual(rejected["records"], [])
        self.assertEqual(rejected["error"]["code"], "context_capacity_exceeded")
        self.assertNotIn("original-body-for-" + a, str(policy.messages[1:]))

    def test_process_scorer_accepts_v5_but_still_requires_strict_interaction_protocol(self):
        config = {"reader": {"prompt_version": READER_VERSION},
                  "retrieval": {"id_protocol": "strict_v2"},
                  "training": {"mini_batch_groups": 8, "target_groups": 8,
                               "process_reward": {"enabled": True}}}
        self.assertTrue(attribution_settings(config)["enabled"])
        config["retrieval"]["id_protocol"] = "strict_v1"
        with self.assertRaises(ValueError):
            attribution_settings(config)


if __name__ == "__main__":
    unittest.main()
