"""Full Reader interaction history remains causal and capacity checked."""
from copy import deepcopy
import json
import unittest

from agmem.core import CapacityError, Completion
from agmem.env import MemoryEnvironment
from agmem.prompts import task_messages
from tests.test_reader_evidence_state import MemoryStore, ScriptedReader, example, state


STOP = {"ids": [], "stop": True}


def read(*ids):
    return {"ids": list(ids), "stop": False}


class ProcessEnvironmentTests(unittest.TestCase):
    def environment(self, store, **kwargs):
        return MemoryEnvironment(store, "directory-v1", max_ids=2, max_calls=8,
                                 id_protocol="strict_v2", prompt_version="interaction_history_v4",
                                 task_prompt_version="history_answer_query_v1", **kwargs)

    def test_full_prior_history_and_complete_capacity_previews_match_real_inputs(self):
        store, item = MemoryStore(), example()
        a, b = [row["id"] for row in store.memory(item.memory_id)]
        reader = ScriptedReader([read(a), read(a, b, "unknown"), read(b), STOP])
        previews, task_inputs = [], []
        def capacity(reader_input, task_input):
            previews.append(deepcopy(reader_input))
            task_inputs.append(deepcopy(task_input))
        row = self.environment(store, capacity=capacity).run(item, reader, 12)
        self.assertEqual(previews, reader.messages[1:])
        self.assertEqual(row.n_read_calls, 3)
        self.assertEqual(store.reads, [a, b])
        for index, messages in enumerate(reader.messages):
            value = state(messages)
            history = value["interaction_history"]
            self.assertEqual(len(history), index)
            self.assertEqual(value["read_calls_used"], index)
            for position, interaction in enumerate(history):
                decision = row.decisions[position]
                self.assertEqual(interaction, {"reader_output": decision.completion.text,
                                                "action": decision.action,
                                                "tool_result": decision.tool_result})
            self.assertNotIn("private-gold", str(messages))
            self.assertNotIn("private-judge", str(messages))
        self.assertNotIn("original-body-for-" + b, str(reader.messages[1]))
        second_history = state(reader.messages[2])["interaction_history"]
        self.assertEqual(second_history[1]["tool_result"]["records"][0]["id"], b)
        self.assertEqual(second_history[1]["tool_result"]["invalid_ids"], ["unknown"])
        self.assertEqual(state(reader.messages[3])["interaction_history"][2]["tool_result"]["records"], [])
        self.assertEqual(task_inputs[0], task_messages(item, row.decisions[0].tool_result["records"],
                                                       "history_answer_query_v1"))
        self.assertTrue(all("public-observation" not in str(value) for value in task_inputs))
        self.assertTrue(all("summary-" not in str(value) for value in task_inputs))
        # Previously constructed states are immutable even if a later audit
        # annotates the live trajectory after generation has completed.
        row.decisions[0].tool_result["records"][0]["raw_text"] = "changed after rollout"
        self.assertNotIn("changed after rollout", str(reader.messages))

    def test_capacity_failure_rolls_back_evidence_and_retains_the_attempt(self):
        store, item = MemoryStore(), example()
        a, b = [row["id"] for row in store.memory(item.memory_id)]
        reader = ScriptedReader([read(a), read(b), STOP])
        previews = []
        def capacity(reader_input, task_input):
            value = state(reader_input)
            previews.append(deepcopy(reader_input))
            if a in value["read_ids"]:
                raise CapacityError("rejected first evidence")
        row = self.environment(store, capacity=capacity).run(item, reader, 9)
        self.assertEqual([record["id"] for record in row.evidence], [b])
        self.assertEqual(len(previews), 3)
        self.assertEqual(previews[1], reader.messages[1])
        self.assertEqual(previews[2], reader.messages[2])
        failed_state = state(reader.messages[1])
        self.assertEqual(failed_state["read_ids"], [])
        self.assertEqual(failed_state["read_calls_used"], 1)
        interaction = failed_state["interaction_history"][0]
        self.assertEqual(interaction["action"], read(a))
        self.assertEqual(interaction["tool_result"]["records"], [])
        self.assertEqual(interaction["tool_result"]["new_ids"], [])
        self.assertEqual(interaction["tool_result"]["error"]["code"], "context_capacity_exceeded")
        self.assertNotIn("original-body-for-" + a, str(reader.messages[1:]))
        self.assertFalse(row.decisions[0].format_violations)

    def test_history_capacity_failure_does_not_silently_truncate_or_score(self):
        store, item = MemoryStore(), example()
        a = store.memory(item.memory_id)[0]["id"]
        reader = ScriptedReader([read(a), STOP])
        previews = []
        def capacity(reader_input, task_input):
            previews.append(state(reader_input))
            raise CapacityError("even failure history does not fit")
        with self.assertRaisesRegex(CapacityError, "even failure history"):
            self.environment(store, capacity=capacity).run(item, reader, 5)
        self.assertEqual(len(reader.messages), 1)
        self.assertEqual(len(previews), 2)
        self.assertEqual(previews[1]["retrieved_evidence"], [])
        self.assertEqual(previews[1]["interaction_history"][0]["tool_result"]["records"], [])

    def test_eight_read_limit_and_two_id_batches_preserve_actual_history(self):
        class LargeStore(MemoryStore):
            def memory(self, memory_id):
                return [{"id": f"{memory_id}-{index}", "source_session_id": str(index),
                         "time": "2026-01-01", "order": index, "raw_text": f"original-{index}"}
                        for index in range(18)]
        store, item = LargeStore(), example()
        keys = [row["id"] for row in store.memory(item.memory_id)]
        reader = ScriptedReader([read(*keys[index:index + 2]) for index in range(0, 18, 2)])
        row = self.environment(store).run(item, reader, 5)
        self.assertEqual(row.n_read_calls, 8)
        self.assertEqual(len(row.evidence), 16)
        self.assertEqual(row.termination_reason, "max_read_calls")
        self.assertEqual(store.reads, keys[:16])
        self.assertEqual(len(state(reader.messages[-1])["interaction_history"]), 8)
        self.assertFalse(row.decisions[-1].tool_result["read_call_consumed"])
        self.assertEqual(row.decisions[-1].format_violations, ["read_budget_exceeded"])

    def test_batched_rollouts_keep_history_separate(self):
        store, item = MemoryStore(), example()
        observed = []
        class Reader:
            def complete_batch(self, messages, seeds, *, max_batch_tokens):
                values = [state(message) for message in messages]
                observed.append(values)
                return [Completion(json.dumps(STOP if value["interaction_history"]
                                               else read(value["directory"][index]["id"])))
                        for index, value in enumerate(values)]
        rows = self.environment(store).run_batch(item, Reader(), [10, 20])
        self.assertEqual([value["interaction_history"] for value in observed[0]], [[], []])
        self.assertEqual([value["interaction_history"][0]["action"]["ids"] for value in observed[1]],
                         [[rows[0].evidence[0]["id"]], [rows[1].evidence[0]["id"]]])
        self.assertNotEqual(rows[0].evidence[0]["id"], rows[1].evidence[0]["id"])

    def test_task_capacity_uses_selected_prompt_in_all_historical_protocols(self):
        for protocol in ("strict_v1", "repair_v2", "strict_v2"):
            with self.subTest(protocol=protocol):
                store, item = MemoryStore(), example()
                a = store.memory(item.memory_id)[0]["id"]
                reader = ScriptedReader([read(a), STOP])
                task_inputs = []
                env = MemoryEnvironment(store, "directory-v1", id_protocol=protocol,
                                        prompt_version="evidence_loop_v3",
                                        task_prompt_version="history_answer_query_v1",
                                        capacity=lambda _, task: task_inputs.append(task))
                row = env.run(item, reader, 3)
                self.assertEqual(task_inputs, [task_messages(item, row.evidence, "history_answer_query_v1")])
                self.assertNotIn("interaction_history", state(reader.messages[-1]))

    def test_full_history_requires_defined_strict_tool_feedback(self):
        for protocol in ("strict_v1", "repair_v2"):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, "requires.*strict_v2"):
                MemoryEnvironment(MemoryStore(), "directory-v1", id_protocol=protocol,
                                  prompt_version="interaction_history_v4")


if __name__ == "__main__":
    unittest.main()
