"""Complete directory visibility and exact multi-leaf retrieval without GPUs."""

from copy import deepcopy
import json
import unittest

from agmem.core import CapacityError, Completion, Example, canonical
from agmem.full_directory_env import FullDirectoryMemoryEnvironment
from agmem.full_directory_prompts import (full_directory_reader_input_contract,
                                         full_directory_reader_system)
from agmem.hierarchical.records import render_records
from agmem.hierarchical_env import HierarchicalMemoryEnvironment
from agmem.hierarchical_prompts import hierarchical_task_messages
from tests.test_hierarchical_environment import Policy, Store, read, STOP


class FullDirectoryEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.example = Example("locomo/train/0", "locomo", "train", "mem_fixture", "QUESTION",
                               observation="PUBLIC_OBSERVATION", options={"A": "PUBLIC_OPTION"},
                               gold_answers=["PRIVATE_GOLD"], judge_metadata={"answer": "PRIVATE_JUDGE"})

    def env(self, **kwargs):
        return FullDirectoryMemoryEnvironment(self.store, "frozen_v1", count_tokens=len, **kwargs)

    def run_actions(self, actions, **kwargs):
        policy = Policy(actions)
        trajectory = self.env(**kwargs).run(self.example, policy, 100)
        return trajectory, [json.loads(message[-1]["content"]) for message in policy.messages]

    def test_first_and_every_step_have_all_directories_without_charging_navigation(self):
        trajectory, states = self.run_actions([
            read("p0001:c0001", "p0003:c0002"), read("p0004:c0001"), STOP])
        for state in states:
            self.assertEqual([row["id"] for row in state["parent_directory"]],
                             ["p0001", "p0002", "p0003", "p0004"])
            self.assertEqual([row["parent_id"] for row in state["child_directories"]],
                             ["p0001", "p0002", "p0003", "p0004"])
            self.assertEqual(sum(len(row["children"]) for row in state["child_directories"]), 8)
            self.assertEqual(state["child_directories"][2]["parent_title"], "Session 3")
            self.assertNotIn("open_parent_ids_lru", state)
            self.assertEqual(state["options"], {"A": "PUBLIC_OPTION"})
        self.assertEqual(states[0]["read_calls_used"], 0)
        self.assertEqual(states[0]["retrieved_evidence"], [])
        self.assertEqual(states[0]["interaction_history"], [])
        self.assertNotIn("Exact original", canonical(states[0]))
        self.assertEqual(trajectory.n_read_calls, 2)

    def test_two_cross_session_reads_return_both_exact_segments_and_sorted_evidence(self):
        trajectory, states = self.run_actions([read("p0003:c0002", "p0001:c0001"), STOP])
        result = trajectory.decisions[0].tool_result
        self.assertEqual(result["new_ids"], ["p0003:c0002", "p0001:c0001"])
        self.assertEqual(result["already_read_ids"], [])
        self.assertEqual(result["duplicate_ids"], [])
        self.assertEqual(trajectory.decisions[0].format_violations, [])
        self.assertEqual(result["retrieval_violations"], [])
        self.assertEqual(result["action_kind"], "segment")
        self.assertTrue(result["ok"])
        self.assertTrue(result["read_call_consumed"])
        self.assertEqual(result["read_calls_used"], 1)
        self.assertEqual([row["id"] for row in trajectory.evidence], ["p0001:c0001", "p0003:c0002"])
        self.assertEqual(states[-1]["read_ids"], ["p0001:c0001", "p0003:c0002"])
        by_id = {row["id"]: row for row in result["records"]}
        self.assertEqual(by_id["p0001:c0001"]["raw_text"], render_records(self.store.data[1][0]["records"][:2]))
        self.assertEqual(by_id["p0003:c0002"]["raw_text"], render_records(self.store.data[1][2]["records"][2:]))
        for row in states[-1]["retrieved_evidence"]:
            self.assertEqual(set(row), {"id", "parent_id", "source_session_id", "time", "title", "content", "raw_text"})

    def test_mixed_invalid_or_already_read_ids_still_return_new_valid_ids(self):
        trajectory, states = self.run_actions([
            read("unknown", "p0001:c0001"), read("p0001:c0001", "p0002:c0001"),
            read("p0003", "p0003:c0002"), STOP])
        first, second, third = [step.tool_result for step in trajectory.decisions[:3]]
        self.assertEqual(first["invalid_ids"], ["unknown"])
        self.assertEqual(first["invalid_id_reasons"], {"unknown": "unknown"})
        self.assertEqual(first["new_ids"], ["p0001:c0001"])
        self.assertEqual(second["already_read_ids"], ["p0001:c0001"])
        self.assertEqual(second["new_ids"], ["p0002:c0001"])
        self.assertEqual(third["invalid_ids"], ["p0003"])
        self.assertEqual(third["invalid_id_reasons"], {"p0003": "nonleaf"})
        self.assertEqual(third["new_ids"], ["p0003:c0002"])
        self.assertEqual(len(states[-1]["interaction_history"]), 3)
        self.assertEqual(states[-1]["read_ids"], ["p0001:c0001", "p0002:c0001", "p0003:c0002"])
        self.assertNotIn("visible_navigation", canonical(trajectory.to_dict()))

    def test_same_batch_duplicate_is_distinct_from_previously_read(self):
        trajectory, states = self.run_actions([
            read("p0001:c0001", "p0001:c0001"), read("p0001:c0001", "p0001:c0001"), STOP])
        first, second = trajectory.decisions[:2]
        self.assertEqual(first.tool_result["already_read_ids"], [])
        self.assertEqual(first.tool_result["duplicate_ids"], ["p0001:c0001"])
        self.assertEqual(first.tool_result["new_ids"], ["p0001:c0001"])
        self.assertEqual(first.format_violations, ["duplicate_ids"])
        self.assertEqual(second.tool_result["already_read_ids"], ["p0001:c0001"])
        self.assertEqual(second.tool_result["duplicate_ids"], ["p0001:c0001"])
        self.assertEqual(second.tool_result["records"], [])
        self.assertEqual(set(second.format_violations), {"duplicate_ids", "repeated_read"})
        self.assertEqual(len(trajectory.evidence), 1)
        self.assertEqual(states[-1]["interaction_history"][0]["tool_result"]["duplicate_ids"], ["p0001:c0001"])

    def test_parent_and_unknown_ids_are_never_navigation(self):
        trajectory, _ = self.run_actions([read("p0001", "unknown"), STOP])
        result = trajectory.decisions[0].tool_result
        self.assertEqual(result["invalid_ids"], ["p0001", "unknown"])
        self.assertEqual(result["action_kind"], "invalid")
        self.assertEqual(result["new_ids"], [])
        self.assertEqual(trajectory.evidence, [])
        self.assertEqual(trajectory.n_read_calls, 1)

    def test_empty_and_oversized_reads_are_charged_and_reject_all_ids(self):
        trajectory, states = self.run_actions([
            read(), read("p0001:c0001", "p0002:c0001", "p0003:c0001"),
            read("p0001:c0001", "unknown", "unknown", "p0002"), read("p0001:c0001"), STOP])
        self.assertEqual(trajectory.n_read_calls, 4)
        self.assertEqual(trajectory.decisions[0].format_violations, ["empty_ids"])
        self.assertEqual(trajectory.decisions[1].format_violations, ["too_many_ids"])
        for step in trajectory.decisions[:3]:
            self.assertEqual(step.tool_result["records"], [])
            self.assertEqual(step.tool_result["selected_ids"], [])
            self.assertNotIn("ids_truncated", step.tool_result)
        oversized = trajectory.decisions[2]
        self.assertEqual(oversized.tool_result["invalid_ids"], ["unknown", "p0002"])
        self.assertEqual(oversized.tool_result["duplicate_ids"], ["unknown"])
        self.assertEqual(set(oversized.format_violations), {"too_many_ids", "unknown_id", "duplicate_ids"})
        self.assertEqual(states[3]["read_ids"], [])
        self.assertEqual(trajectory.decisions[3].tool_result["already_read_ids"], [])
        self.assertEqual([row["id"] for row in trajectory.evidence], ["p0001:c0001"])

    def test_budget_exhaustion_keeps_final_feedback_without_cost_or_evidence(self):
        trajectory, states = self.run_actions([
            read("p0001:c0001"), read("p0002:c0001", "unknown")], max_calls=1)
        final = trajectory.decisions[-1]
        self.assertEqual(trajectory.termination_reason, "max_read_calls")
        self.assertEqual(trajectory.n_read_calls, 1)
        self.assertEqual(states[-1]["read_calls_remaining"], 0)
        self.assertEqual(final.tool_result["invalid_ids"], ["unknown"])
        self.assertEqual(final.tool_result["records"], [])
        self.assertFalse(final.tool_result["read_call_consumed"])
        self.assertEqual(final.tool_result["read_calls_remaining"], 0)
        self.assertEqual(final.tool_result["error"]["code"], "max_read_calls")
        self.assertEqual(final.format_violations.count("read_budget_exceeded"), 1)
        self.assertIn("read_budget_exceeded", final.tool_result["retrieval_violations"])
        self.assertEqual([row["id"] for row in trajectory.evidence], ["p0001:c0001"])

    def test_stop_is_free_and_nonempty_stop_has_only_one_format_violation(self):
        for ids in ([], ["p0001"], ["p0001:c0001", "p0001:c0001", "unknown"]):
            with self.subTest(ids=ids):
                trajectory, _ = self.run_actions([{"ids": ids, "stop": True}])
                step = trajectory.decisions[0]
                self.assertEqual(step.tool_result["action_kind"], "stop")
                self.assertEqual(step.tool_result["directory_protocol"], "hierarchical_full_v1")
                self.assertEqual(step.format_violations, ["stop_with_ids"] if ids else [])
                self.assertEqual(step.tool_result["requested_ids"], ids)
                for field in ("records", "new_ids", "invalid_ids", "already_read_ids", "duplicate_ids"):
                    self.assertEqual(step.tool_result[field], [])
                self.assertEqual(trajectory.n_read_calls, 0)
                self.assertEqual(trajectory.evidence, [])
                self.assertEqual(trajectory.termination_reason, "stop")
        trajectory, _ = self.run_actions([read("p0001:c0001"), STOP], max_calls=1)
        self.assertEqual(trajectory.termination_reason, "stop")
        self.assertEqual(trajectory.decisions[-1].format_violations, [])

    def test_invalid_json_or_schema_terminates_before_reading(self):
        for action in ("not JSON", "[]", '{"ids":[],"stop":0}', '{"ids":[1],"stop":false}',
                       '{"ids":[],"stop":false,"extra":true}'):
            with self.subTest(action=action):
                trajectory, _ = self.run_actions([action])
                step = trajectory.decisions[0]
                self.assertEqual(step.format_violations, ["invalid_json_or_schema"])
                self.assertEqual(step.tool_result["action_kind"], "invalid")
                self.assertEqual(step.tool_result["directory_protocol"], "hierarchical_full_v1")
                self.assertEqual(trajectory.termination_reason, "protocol_failure")
                self.assertEqual(trajectory.n_read_calls, 0)

    def test_capacity_rejects_whole_read_without_marking_ids_read_or_shortening_source(self):
        def capacity(reader, task):
            state = json.loads(reader[-1]["content"])
            if "p0002:c0001" in state["read_ids"]:
                self.assertIn("Exact original 2.1", canonical(task))
                self.assertIn("Exact original 2.2", canonical(task))
                raise CapacityError("full original evidence does not fit")
        trajectory, states = self.run_actions([
            read("p0001:c0001", "p0002:c0001"), read("p0001:c0001"), read("p0002:c0001"), STOP],
            capacity=capacity)
        self.assertEqual(states[1]["read_ids"], [])
        self.assertEqual(states[-1]["read_ids"], ["p0001:c0001"])
        self.assertEqual(len(states[-1]["child_directories"]), 4)
        for index in (0, 2):
            result = trajectory.decisions[index].tool_result
            self.assertEqual(result["error"]["code"], "context_capacity_exceeded")
            self.assertEqual(result["records"], [])
            self.assertEqual(result["new_ids"], [])
            self.assertEqual(result["already_read_ids"], [])
        self.assertEqual(trajectory.decisions[1].tool_result["already_read_ids"], [])
        self.assertEqual(trajectory.n_read_calls, 3)

    def test_initial_full_directory_or_failure_history_capacity_error_propagates(self):
        initial_policy = Policy([STOP])
        def initial_capacity(reader, task):
            self.assertEqual(len(json.loads(reader[-1]["content"])["child_directories"]), 4)
            raise CapacityError("full directory does not fit")
        with self.assertRaisesRegex(CapacityError, "full directory"):
            self.env(capacity=initial_capacity).run(self.example, initial_policy, 0)
        self.assertEqual(initial_policy.messages, [])
        def history_capacity(reader, task):
            if json.loads(reader[-1]["content"])["interaction_history"]:
                raise CapacityError("history does not fit")
        with self.assertRaisesRegex(CapacityError, "history"):
            self.run_actions([read("p0001:c0001")], capacity=history_capacity)

    def test_public_reader_and_selected_task_evidence_exclude_private_and_unread_source(self):
        trajectory, states = self.run_actions([read("p0001:c0001"), STOP])
        for text in (canonical(states), canonical(trajectory.evidence),
                     canonical(hierarchical_task_messages(self.example, trajectory.evidence))):
            for forbidden in ("PRIVATE_GOLD", "PRIVATE_JUDGE", "UNSELECTED_HEADER", "Exact original 1.3",
                              "Exact original 2.1"):
                self.assertNotIn(forbidden, text)
        task = canonical(hierarchical_task_messages(self.example, trajectory.evidence))
        for forbidden in ("PARENT_ONLY_SUMMARY", "CHILD_SUMMARY_1_2", "p0002:c0001"):
            self.assertNotIn(forbidden, task)
        for expected in ("QUESTION", "PUBLIC_OBSERVATION", "PUBLIC_OPTION", "CHILD_SUMMARY_1_1",
                         "Exact original 1.1", "Exact original 1.2"):
            self.assertIn(expected, task)
        self.assertNotIn("records", states[-1]["interaction_history"][0]["tool_result"])

    def test_frontier_makes_future_children_unavailable(self):
        trajectory, states = self.run_actions([read("p0002:c0001", "p0001:c0001"), STOP], visible_frontier=1)
        self.assertEqual([row["id"] for row in states[0]["parent_directory"]], ["p0001"])
        self.assertEqual(len(states[0]["child_directories"]), 1)
        result = trajectory.decisions[0].tool_result
        self.assertEqual(result["invalid_ids"], ["p0002:c0001"])
        self.assertEqual(result["invalid_id_reasons"], {"p0002:c0001": "unavailable"})
        self.assertEqual(result["new_ids"], ["p0001:c0001"])

    def test_batch_rollouts_keep_read_history_independent(self):
        class BatchPolicy:
            def complete_batch(self, messages, seeds, max_batch_tokens):
                return [Completion(canonical(STOP if json.loads(message[-1]["content"])["read_ids"]
                                             else read("p0001:c0001", "p0002:c0002"))) for message in messages]
        rows = self.env().run_batch(self.example, BatchPolicy(), [10, 20])
        self.assertEqual([row.n_read_calls for row in rows], [1, 1])
        self.assertEqual([len(row.evidence) for row in rows], [2, 2])
        self.assertIsNot(rows[0].evidence[0], rows[1].evidence[0])
        self.assertEqual(rows[0].decisions[0].tool_result["already_read_ids"], [])
        self.assertEqual(rows[1].decisions[0].tool_result["already_read_ids"], [])

    def test_contract_validation_and_old_environment_remain_distinct(self):
        before = deepcopy(self.store.data)
        self.run_actions([read("p0001:c0001"), STOP])
        self.assertEqual(self.store.data, before)
        policy = Policy([read("p0001"), STOP])
        old = HierarchicalMemoryEnvironment(self.store, "frozen_v1", count_tokens=len).run(self.example, policy, 0)
        self.assertEqual(json.loads(policy.messages[0][-1]["content"])["open_child_directories"], [])
        self.assertEqual(old.decisions[0].tool_result["action_kind"], "navigation")
        self.assertEqual(old.decisions[0].tool_result["directory_protocol"], "hierarchical_v1")
        self.assertEqual(old.evidence, [])
        contract = full_directory_reader_input_contract()
        self.assertEqual(contract["prompt_version"], "hierarchical_full_v1")
        self.assertEqual(contract["retention"], "cumulative")
        self.assertEqual(contract["context_blocks"], 0)
        self.assertNotEqual(contract["reader_prompt_sha256"],
                            full_directory_reader_input_contract(max_calls=2)["reader_prompt_sha256"])
        self.assertIn("between 1 and 2", full_directory_reader_system())
        for kwargs in ({"max_ids": 0}, {"max_ids": 1}, {"max_ids": 3}, {"max_ids": True}, {"max_calls": 0},
                       {"id_protocol": "strict_v1"}, {"prompt_version": "hierarchical_v1"},
                       {"task_prompt_version": "legacy_v1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.env(**kwargs)


if __name__ == "__main__":
    unittest.main()
