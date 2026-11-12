"""Hierarchical ORM state, exact evidence and charged READ protocol without GPUs."""

from copy import deepcopy
import json
import unittest

from agmem.core import CapacityError, Completion, Example, canonical
from agmem.hierarchical.binding import bind_directory
from agmem.hierarchical.records import make_session
from agmem.hierarchical_env import HierarchicalMemoryEnvironment
from agmem.hierarchical_prompts import (compact_interaction_history, hierarchical_reader_input_contract,
                                        hierarchical_task_input_contract, hierarchical_task_messages)
from agmem.rewards import invalid_id_count, protocol_metrics, score_trajectory


def bank():
    sessions, directories = [], []
    for number in range(1, 5):
        pid = f"p{number:04d}"
        records = [{"record_id": f"{pid}:r{i:04d}", "seq": i + 1, "kind": "message",
                    "speaker": "Alice" if i % 2 else "Bob", "call_id": None,
                    "text": f"Exact original {number}.{i}", "attachments": []} for i in range(1, 5)]
        header = {"record_id": f"{pid}:h0001", "seq": 1, "kind": "task_instruction",
                  "speaker": "user", "call_id": None, "text": "UNSELECTED_HEADER", "attachments": []}
        source = make_session(bank_id="bank_fixture", parent_id=pid, order=number, records=records,
                              header_records=[header], time={"raw": "2023-01-01", "iso": "2023-01-01", "timezone": None},
                              participants=["Alice", "Bob"], count_tokens=len)
        result = {"parent": {"title": f"Session {number}", "content": f"PARENT_ONLY_SUMMARY_{number}"},
                  "children": [{"end_block": 2, "title": "First topic", "content": f"CHILD_SUMMARY_{number}_1",
                                "entities": ["Alice"], "start_reason": "session_start"},
                               {"end_block": 4, "title": "Second topic", "content": f"CHILD_SUMMARY_{number}_2",
                                "entities": ["Bob"], "start_reason": "semantic"}],
                  "status": "complete", "mode": "single_session"}
        sessions.append(source)
        directories.append(bind_directory(source, result, {"model": "fixture", "model_revision": "v1"},
                                          "character-counter", count_tokens=len))
    return directories, sessions


class Store:
    def __init__(self):
        self.data = bank()

    def hierarchical_bank(self, memory_id, version):
        if memory_id != "mem_fixture" or version != "frozen_v1":
            raise ValueError("Wrong frozen bank identity")
        return deepcopy(self.data)


def read(*ids):
    return {"ids": list(ids), "stop": False}


STOP = {"ids": [], "stop": True}


class Policy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages, seed):
        self.messages.append(deepcopy(messages))
        action = next(self.actions)
        return Completion(action if isinstance(action, str) else canonical(action))


class HierarchicalEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.example = Example("locomo/train/0", "locomo", "train", "mem_fixture", "QUESTION",
                               observation="PUBLIC_OBSERVATION", gold_answers=["PRIVATE_GOLD"])

    def env(self, **kwargs):
        return HierarchicalMemoryEnvironment(self.store, "frozen_v1", count_tokens=len, **kwargs)

    def run_actions(self, actions, **kwargs):
        policy = Policy(actions)
        trajectory = self.env(**kwargs).run(self.example, policy, 100)
        return trajectory, [json.loads(msg[-1]["content"]) for msg in policy.messages]

    def test_initial_view_and_parent_navigation_do_not_add_task_evidence(self):
        trajectory, states = self.run_actions([read("p0001"), STOP])
        self.assertEqual(len(states[0]["parent_directory"]), 4)
        self.assertEqual(states[0]["open_child_directories"], [])
        self.assertEqual(states[0]["retrieved_evidence"], [])
        self.assertNotIn("p0001:c0001", canonical(states[0]))
        self.assertNotIn("PRIVATE_GOLD", canonical(states))
        self.assertEqual(states[1]["open_parent_ids_lru"], ["p0001"])
        self.assertEqual(trajectory.n_read_calls, 1)
        self.assertEqual(trajectory.evidence, [])
        result = trajectory.decisions[0].tool_result
        self.assertEqual(result["action_kind"], "navigation")
        self.assertTrue(result["directory_navigation_success"])
        self.assertEqual(result["records"], [])
        self.assertEqual(result["visible_navigation_ids"], [])

    def test_zero_read_episode_identifies_hierarchical_protocol(self):
        for action, kind in ((STOP, "stop"), ("not JSON", "invalid")):
            with self.subTest(kind=kind):
                trajectory, _ = self.run_actions([action])
                result = trajectory.decisions[0].tool_result
                self.assertEqual(result["directory_protocol"], "hierarchical_v1")
                self.assertEqual(result["action_kind"], kind)
                self.assertFalse(result["read_call_consumed"])
                self.assertEqual(result["records"], [])
                self.assertEqual(trajectory.n_read_calls, 0)

    def test_exact_child_raw_summary_and_id_remain_after_parent_eviction(self):
        trajectory, states = self.run_actions([
            read("p0001"), read("p0001:c0001"), read("p0002"), read("p0003"), STOP])
        state = states[-1]
        self.assertEqual(state["open_parent_ids_lru"], ["p0002", "p0003"])
        self.assertEqual(state["read_ids"], ["p0001:c0001"])
        row = trajectory.evidence[0]
        self.assertEqual(row["id"], "p0001:c0001")
        self.assertEqual(row["content"], "CHILD_SUMMARY_1_1")
        self.assertIn("Exact original 1.1", row["raw_text"])
        self.assertIn("Exact original 1.2", row["raw_text"])
        for forbidden in ("Exact original 1.3", "UNSELECTED_HEADER", "PARENT_ONLY_SUMMARY", "PRIVATE_GOLD"):
            self.assertNotIn(forbidden, canonical(trajectory.evidence))
        task = canonical(hierarchical_task_messages(self.example, trajectory.evidence))
        for value in ("QUESTION", "PUBLIC_OBSERVATION", "p0001:c0001", "CHILD_SUMMARY_1_1"):
            self.assertIn(value, task)
        for forbidden in ("CHILD_SUMMARY_1_2", "PARENT_ONLY_SUMMARY", "p0002:c0001", "PRIVATE_GOLD"):
            self.assertNotIn(forbidden, task)
        self.assertNotIn("CHILD_SUMMARY_1_2", canonical(state))
        self.assertNotIn("records", canonical(state["interaction_history"]))

    def test_lru_access_on_child_read_and_parent_reopen(self):
        trajectory, states = self.run_actions([
            read("p0001"), read("p0002"), read("p0001:c0001"), read("p0003"), read("p0002"), STOP])
        self.assertEqual(states[3]["open_parent_ids_lru"], ["p0002", "p0001"])
        self.assertEqual(states[4]["open_parent_ids_lru"], ["p0001", "p0003"])
        self.assertEqual(states[5]["open_parent_ids_lru"], ["p0003", "p0002"])
        result = trajectory.decisions[4].tool_result
        self.assertTrue(result["repeated_navigation"])
        self.assertFalse(result["visible_navigation_noop"])
        self.assertEqual(result["evicted_parent_ids"], ["p0001"])
        self.assertEqual(result["visible_navigation_ids"], [])
        self.assertEqual(trajectory.decisions[4].format_violations, [])

    def test_visible_parent_and_already_read_child_each_receive_one_penalty(self):
        trajectory, states = self.run_actions([
            read("p0001"), read("p0001"), read("p0001:c0001"), read("p0001:c0001"), STOP])
        self.assertEqual(trajectory.n_read_calls, 4)
        self.assertEqual(trajectory.decisions[1].format_violations, ["visible_navigation"])
        self.assertTrue(trajectory.decisions[1].tool_result["visible_navigation_noop"])
        repeated = trajectory.decisions[3]
        self.assertEqual(repeated.format_violations, ["repeated_read"])
        self.assertEqual(repeated.tool_result["records"], [])
        self.assertEqual(repeated.tool_result["already_read_ids"], ["p0001:c0001"])
        self.assertEqual(len(trajectory.evidence), 1)
        self.assertEqual(invalid_id_count(trajectory, max_ids=1), 2)

    def test_visible_navigation_penalizes_every_request_and_preserves_cost(self):
        trajectory, states = self.run_actions([read("p0001"), read("p0001"), read("p0001"), STOP])
        self.assertEqual([d.tool_result.get("visible_navigation_ids", []) for d in trajectory.decisions],
                         [[], ["p0001"], ["p0001"], []])
        self.assertEqual(states[-1]["open_parent_ids_lru"], ["p0001"])
        self.assertEqual(states[-1]["read_ids"], [])
        self.assertEqual(states[-1]["last_tool_error"]["code"], "visible_navigation")
        self.assertEqual(states[-1]["interaction_history"][-1]["tool_result"]["visible_navigation_ids"], ["p0001"])
        trajectory.prediction = self.example.gold_answers[0]
        score_trajectory(self.example, trajectory,
                         {"formula": "f1_cost_v1", "read_call_cost": .05,
                          "format_penalty": .5, "format_penalty_mode": "per_id"}, max_ids=1)
        self.assertEqual(trajectory.n_read_calls, 3)
        self.assertEqual(trajectory.format_violation_count, 2)
        self.assertEqual(trajectory.format_reward, -1.0)
        self.assertAlmostEqual(trajectory.read_cost, .15)
        self.assertAlmostEqual(trajectory.reward, -.15)
        from agmem.pipeline import metrics
        self.assertEqual(metrics([trajectory])["source_macro"]["visible_navigation_penalty_calls"], 2)
        self.assertEqual(protocol_metrics([trajectory])["repeated_read_calls"], 0)

    def test_an_intervening_action_does_not_exempt_a_still_visible_directory(self):
        for other in (read("p0001:c0001"), read("p0002"), read("unknown")):
            with self.subTest(other=other):
                trajectory, _ = self.run_actions([read("p0001"), other, read("p0001"), STOP])
                returned = trajectory.decisions[2]
                self.assertEqual(returned.format_violations, ["visible_navigation"])
                self.assertTrue(returned.tool_result["repeated_navigation"])
                self.assertEqual(returned.tool_result["visible_navigation_ids"], ["p0001"])
                self.assertEqual(protocol_metrics([trajectory])["visible_navigation_penalty_calls"], 1)

    def test_repeated_failed_navigation_is_not_penalized_when_directory_stays_hidden(self):
        def capacity(reader, task):
            if "p0001" in json.loads(reader[-1]["content"])["open_parent_ids_lru"]:
                raise CapacityError("directory exceeds capacity")
        trajectory, states = self.run_actions([read("p0001"), read("p0001"), STOP], capacity=capacity)
        self.assertEqual(trajectory.decisions[0].format_violations, [])
        self.assertEqual(trajectory.decisions[1].format_violations, [])
        self.assertEqual(trajectory.decisions[1].tool_result["visible_navigation_ids"], [])
        self.assertEqual(states[-1]["open_parent_ids_lru"], [])
        self.assertEqual(trajectory.n_read_calls, 2)
        self.assertEqual(invalid_id_count(trajectory, max_ids=1), 0)

    def test_visible_directory_penalty_does_not_require_two_open_directories(self):
        trajectory, states = self.run_actions([read("p0001"), read("p0001:c0001"), read("p0001"), STOP])
        self.assertEqual(states[2]["open_parent_ids_lru"], ["p0001"])
        self.assertEqual(trajectory.decisions[2].tool_result["visible_navigation_ids"], ["p0001"])
        self.assertEqual(invalid_id_count(trajectory, max_ids=1), 1)

    def test_visible_parent_in_excess_suffix_uses_pre_action_view_and_counts_once_per_position(self):
        trajectory, _ = self.run_actions([
            read("p0001"), read("p0002"), read("p0003", "p0001", "p0001"), STOP])
        step = trajectory.decisions[2]
        self.assertEqual(step.tool_result["visible_navigation_ids"], ["p0001"])
        self.assertEqual(step.tool_result["evicted_parent_ids"], ["p0001"])
        self.assertEqual(set(step.format_violations), {"too_many_ids", "duplicate_ids", "visible_navigation"})
        self.assertEqual(invalid_id_count(trajectory, max_ids=1), 2)

    def test_visible_navigation_in_segment_request_suffix_keeps_segment_evidence(self):
        trajectory, _ = self.run_actions([
            read("p0001"), read("p0001:c0001", "p0001"), STOP])
        step = trajectory.decisions[1]
        self.assertEqual(step.tool_result["action_kind"], "segment")
        self.assertEqual(step.tool_result["visible_navigation_ids"], ["p0001"])
        self.assertEqual([record["id"] for record in trajectory.evidence], ["p0001:c0001"])
        self.assertEqual(invalid_id_count(trajectory, max_ids=1), 1)
        self.assertEqual(protocol_metrics([trajectory])["visible_navigation_penalty_calls"], 1)

    def test_unopened_and_evicted_unread_child_are_invalid_until_parent_reopened(self):
        trajectory, states = self.run_actions([
            read("p0001:c0002"), read("p0001"), read("p0002"), read("p0003"),
            read("p0001:c0002"), read("p0001"), read("p0001:c0002"), STOP])
        for index in (0, 4):
            result = trajectory.decisions[index].tool_result
            self.assertEqual(result["invalid_ids"], ["p0001:c0002"])
            self.assertEqual(result["records"], [])
        self.assertEqual(states[5]["read_ids"], [])
        self.assertEqual([row["id"] for row in trajectory.evidence], ["p0001:c0002"])
        self.assertEqual(trajectory.n_read_calls, 7)

    def test_full_request_inspected_but_only_first_id_executes(self):
        trajectory, states = self.run_actions([
            read("p0001", "p0001", "p0001:c0001", "bogus"), STOP])
        decision = trajectory.decisions[0]
        self.assertEqual(set(decision.format_violations), {"too_many_ids", "duplicate_ids", "unknown_id"})
        self.assertEqual(decision.tool_result["invalid_ids"], ["p0001:c0001", "bogus"])
        self.assertTrue(decision.tool_result["directory_navigation_success"])
        self.assertEqual(states[1]["open_parent_ids_lru"], ["p0001"])
        self.assertEqual(states[1]["read_ids"], [])
        self.assertEqual(invalid_id_count(trajectory, max_ids=1), 3)

    def test_ten_attempts_charge_including_invalid_and_stop_is_free(self):
        actions = [read("unknown") for _ in range(10)] + [STOP]
        trajectory, states = self.run_actions(actions)
        self.assertEqual(trajectory.n_read_calls, 10)
        self.assertEqual(trajectory.termination_reason, "stop")
        self.assertEqual(states[-1]["read_calls_remaining"], 0)
        self.assertTrue(all(d.tool_result["read_call_consumed"] for d in trajectory.decisions[:-1]))
        trajectory, _ = self.run_actions([read("p0001") for _ in range(11)])
        final = trajectory.decisions[-1]
        self.assertEqual(trajectory.n_read_calls, 10)
        self.assertEqual(trajectory.termination_reason, "max_read_calls")
        self.assertEqual(final.tool_result["action_kind"], "navigation")
        self.assertFalse(final.tool_result["read_call_consumed"])
        self.assertIn("read_budget_exceeded", final.format_violations)

    def test_capacity_rejection_rolls_back_navigation_and_lru_eviction(self):
        def capacity(reader, task):
            state = json.loads(reader[-1]["content"])
            if "p0003" in state["open_parent_ids_lru"]:
                raise CapacityError("large child directory")
        trajectory, states = self.run_actions([
            read("p0001"), read("p0002"), read("p0003"), read("p0001:c0001"), STOP], capacity=capacity)
        result = trajectory.decisions[2].tool_result
        self.assertEqual(result["error"]["code"], "context_capacity_exceeded")
        self.assertEqual(result["evicted_parent_ids"], [])
        self.assertFalse(result["directory_navigation_success"])
        self.assertEqual(states[3]["open_parent_ids_lru"], ["p0001", "p0002"])
        self.assertEqual([r["id"] for r in trajectory.evidence], ["p0001:c0001"])
        self.assertEqual(trajectory.decisions[2].format_violations, [])

    def test_capacity_rejected_child_never_becomes_already_read(self):
        def capacity(reader, task):
            if "Exact original 1.3" in canonical(task):
                raise CapacityError("large segment")
        trajectory, _ = self.run_actions([
            read("p0001"), read("p0001:c0001"), read("p0001:c0002"), read("p0001:c0002"), STOP], capacity=capacity)
        self.assertEqual(trajectory.n_read_calls, 4)
        for index in (2, 3):
            result = trajectory.decisions[index].tool_result
            self.assertEqual(result["records"], [])
            self.assertEqual(result["already_read_ids"], [])
            self.assertEqual(trajectory.decisions[index].format_violations, [])
        self.assertEqual([r["id"] for r in trajectory.evidence], ["p0001:c0001"])

    def test_history_itself_over_capacity_propagates(self):
        def capacity(reader, task):
            if json.loads(reader[-1]["content"])["interaction_history"]:
                raise CapacityError("history does not fit")
        with self.assertRaises(CapacityError):
            self.run_actions([read("p0001")], capacity=capacity)

    def test_batch_rollouts_have_independent_retrieval_state(self):
        class BatchPolicy:
            def complete_batch(self, messages, seeds, max_batch_tokens):
                completions = []
                for message in messages:
                    state = json.loads(message[-1]["content"])
                    action = read("p0001") if state["read_calls_used"] == 0 else read("p0001:c0001") if not state["read_ids"] else STOP
                    completions.append(Completion(canonical(action)))
                return completions
        rows = self.env().run_batch(self.example, BatchPolicy(), [10, 20])
        self.assertEqual([r.n_read_calls for r in rows], [2, 2])
        self.assertEqual([len(r.evidence) for r in rows], [1, 1])
        self.assertIsNot(rows[0].evidence[0], rows[1].evidence[0])

    def test_protocol_contracts_and_malformed_json(self):
        with self.assertRaises(ValueError):
            self.env(max_ids=2)
        self.assertEqual(hierarchical_reader_input_contract()["context_blocks"], 0)
        self.assertEqual(hierarchical_task_input_contract()["completion_policy"], "score_tagged_truncated_v1")
        trajectory, _ = self.run_actions(["not json"])
        self.assertEqual(trajectory.termination_reason, "protocol_failure")
        self.assertEqual(trajectory.n_read_calls, 0)
        self.assertEqual(compact_interaction_history(trajectory.decisions)[0]["action"], None)


if __name__ == "__main__":
    unittest.main()
