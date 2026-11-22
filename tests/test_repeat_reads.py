"""Strict v2 successful-read history, feedback, and trajectory accounting."""
import json
import unittest
from unittest.mock import patch

from agmem.core import CapacityError, Completion
from agmem.env import MemoryEnvironment
from agmem.rewards import protocol_metrics, score_trajectory
from tests import test_core as fixtures


class RepeatReadTests(unittest.TestCase):
    setUp = fixtures.CoreTests.setUp
    tearDown = fixtures.CoreTests.tearDown
    stop = {"ids": [], "stop": True}

    @staticmethod
    def read(*ids):
        return {"ids": list(ids), "stop": False}

    def run_actions(self, actions, *, max_calls=8, capacity=None, protocol="strict_v2"):
        self.env = MemoryEnvironment(self.store, self.builder.version, max_calls=max_calls,
                                     capacity=capacity, id_protocol=protocol)
        self.policy = fixtures.ScriptedPolicy(actions)
        return self.env.run(self.example, self.policy, 42)

    def score(self, row):
        row.prediction = "wrong"
        return score_trajectory(self.example, row, {"formula": "outcome_cost_v1",
                                                   "read_call_cost": .01, "format_penalty": .1})

    def test_nonadjacent_repeat_uses_successful_history_and_never_refetches(self):
        a, b = [x["id"] for x in self.directory[:2]]
        with patch.object(self.store, "record", wraps=self.store.record) as fetch:
            row = self.run_actions([self.read(a), self.read(b), self.read(a), self.stop])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual([x["id"] for x in row.evidence], [a, b])
        self.assertEqual(row.n_read_calls, 3)
        self.assertEqual(row.termination_reason, "stop")
        repeat = row.decisions[2]
        self.assertEqual(repeat.format_violations, ["repeated_read"])
        self.assertEqual(repeat.tool_result["already_read_ids"], [a])
        self.assertEqual(repeat.tool_result["new_ids"], [])
        self.assertTrue(repeat.tool_result["no_new_evidence"])
        self.assertTrue(repeat.tool_result["read_call_consumed"])

    def test_mixed_old_and_new_fetches_new_only_and_preserves_old_evidence(self):
        a, b = [x["id"] for x in self.directory[:2]]
        action = self.read(a, b)
        with patch.object(self.store, "record", wraps=self.store.record) as fetch:
            row = self.run_actions([self.read(a), action, self.stop])
        self.assertEqual(fetch.call_count, 2)
        step = row.decisions[1]
        self.assertEqual(step.action, action)
        self.assertEqual(json.loads(step.completion.text), action)
        self.assertEqual([x["id"] for x in row.evidence], [a, b])
        self.assertEqual([x["id"] for x in step.tool_result["records"]], [b])
        self.assertEqual(step.tool_result["already_read_ids"], [a])
        self.assertEqual(step.tool_result["new_ids"], [b])
        self.assertFalse(step.tool_result["no_new_evidence"])
        feedback = json.loads(self.policy.messages[2][1]["content"])["last_tool_error"]
        self.assertEqual(feedback, step.tool_result["error"])
        self.assertEqual(feedback["already_read_ids"], [a])
        self.assertEqual(feedback["new_ids"], [b])
        self.assertFalse(feedback["no_new_evidence"])

    def test_all_repeated_feedback_changes_next_state_then_success_clears_error(self):
        a, b = [x["id"] for x in self.directory[:2]]
        row = self.run_actions([self.read(a), self.read(a), self.read(b), self.stop])
        before = json.loads(self.policy.messages[1][1]["content"])
        after = json.loads(self.policy.messages[2][1]["content"])
        self.assertEqual(before["retrieved_evidence"], after["retrieved_evidence"])
        self.assertNotEqual(before, after)
        self.assertEqual(after["last_tool_error"]["code"], "repeated_read")
        self.assertEqual(after["last_tool_error"]["new_ids"], [])
        self.assertTrue(after["last_tool_error"]["no_new_evidence"])
        self.assertNotEqual(row.decisions[1].state_sha256, row.decisions[2].state_sha256)
        self.assertNotIn("last_tool_error", json.loads(self.policy.messages[3][1]["content"]))
        self.assertEqual(self.policy.messages[0][0], self.policy.messages[2][0])

    def test_capacity_failed_attempt_does_not_make_retry_a_repeat(self):
        a = self.directory[0]["id"]
        checks = []
        def capacity(*args):
            checks.append(args)
            if len(checks) == 1:
                raise CapacityError("fixture")
        row = self.run_actions([self.read(a), self.read(a), self.stop], capacity=capacity)
        first, retry = row.decisions[:2]
        self.assertEqual(first.tool_result["error"]["code"], "context_capacity_exceeded")
        self.assertEqual(first.tool_result["new_ids"], [])
        self.assertTrue(first.tool_result["no_new_evidence"])
        self.assertEqual(retry.tool_result["already_read_ids"], [])
        self.assertEqual(retry.tool_result["new_ids"], [a])
        self.assertEqual(retry.format_violations, [])
        self.assertEqual(row.n_read_calls, 2)
        self.assertEqual([x["id"] for x in row.evidence], [a])
        self.assertFalse(self.score(row).format_failure)

    def test_consecutive_repeats_update_budget_feedback_each_time(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read(a), self.read(a), self.read(a), self.stop], max_calls=3)
        first_error = json.loads(self.policy.messages[2][1]["content"])["last_tool_error"]
        second_error = json.loads(self.policy.messages[3][1]["content"])["last_tool_error"]
        self.assertEqual(first_error["read_calls_used"], 2)
        self.assertEqual(first_error["read_calls_remaining"], 1)
        self.assertEqual(second_error["read_calls_used"], 3)
        self.assertEqual(second_error["read_calls_remaining"], 0)
        self.assertNotEqual(row.decisions[2].state_sha256, row.decisions[3].state_sha256)
        self.assertEqual(row.termination_reason, "stop")

    def test_unknown_ids_never_enter_read_history_and_valid_partial_can_succeed(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read("unknown"), self.read("unknown", a), self.stop])
        self.assertEqual(row.n_read_calls, 2)
        self.assertEqual(row.termination_reason, "stop")
        self.assertEqual([x["id"] for x in row.evidence], [a])
        for step in row.decisions[:2]:
            self.assertEqual(step.format_violations, ["unknown_id"])
            self.assertEqual(step.tool_result["already_read_ids"], [])
            self.assertEqual(step.tool_result["invalid_ids"], ["unknown"])
        self.assertEqual(row.decisions[1].tool_result["new_ids"], [a])
        self.assertEqual(self.score(row).format_reward, -.1)

    def test_same_call_duplicates_are_format_failure_but_not_historical_repeat(self):
        a = self.directory[0]["id"]
        with patch.object(self.store, "record", wraps=self.store.record) as fetch:
            row = self.run_actions([self.read(a, a), self.stop])
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(row.decisions[0].format_violations, ["duplicate_ids"])
        self.assertEqual(row.decisions[0].tool_result["already_read_ids"], [])
        self.assertEqual(row.decisions[0].tool_result["new_ids"], [a])
        self.assertEqual(self.score(row).format_reward, -.1)

    def test_empty_and_unknown_calls_consume_budget_and_penalty_does_not_accumulate(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read(), self.read("unknown"), self.read(a), self.read(a), self.stop])
        self.assertEqual(row.n_read_calls, 4)
        self.assertEqual(row.decisions[0].format_violations, ["empty_ids"])
        self.assertTrue(all(d.tool_result["read_call_consumed"] for d in row.decisions[:-1]))
        scored = self.score(row)
        self.assertTrue(scored.format_failure)
        self.assertEqual(scored.format_reward, -.1)
        self.assertAlmostEqual(scored.read_cost, .04)

    def test_full_action_is_audited_but_only_allowed_prefix_is_fetched(self):
        keys = [x["id"] for x in self.directory[:5]]
        action = self.read(*keys, "unknown", keys[0])
        with patch.object(self.store, "record", wraps=self.store.record) as fetch:
            row = self.run_actions([action, self.stop])
        self.assertEqual(fetch.call_count, 4)
        self.assertEqual(row.decisions[0].action, action)
        self.assertEqual(row.decisions[0].format_violations, ["too_many_ids", "duplicate_ids", "unknown_id"])
        self.assertEqual(row.decisions[0].tool_result["invalid_ids"], ["unknown"])
        self.assertEqual(row.decisions[0].tool_result["new_ids"], keys[:4])
        self.assertTrue(row.decisions[0].tool_result["ids_truncated"])
        self.assertEqual([x["id"] for x in row.evidence], keys[:4])

    def test_budget_exceeded_action_is_audited_but_not_executed_or_charged(self):
        a, b = [x["id"] for x in self.directory[:2]]
        with patch.object(self.store, "record", wraps=self.store.record) as fetch:
            row = self.run_actions([self.read(a), self.read(a, b, "unknown")], max_calls=1)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(row.n_read_calls, 1)
        self.assertEqual(row.termination_reason, "max_read_calls")
        last = row.decisions[-1]
        self.assertEqual(last.format_violations, ["unknown_id", "repeated_read", "read_budget_exceeded"])
        self.assertFalse(last.tool_result["read_call_consumed"])
        self.assertEqual(last.tool_result["error"]["code"], "max_read_calls")
        self.assertEqual(last.tool_result["records"], [])
        self.assertEqual(last.tool_result["new_ids"], [])
        self.assertEqual([x["id"] for x in row.evidence], [a])

    def test_valid_extra_read_is_still_explicit_budget_violation(self):
        a, b = [x["id"] for x in self.directory[:2]]
        row = self.run_actions([self.read(a), self.read(b)], max_calls=1)
        self.assertEqual(row.decisions[-1].format_violations, ["read_budget_exceeded"])
        self.assertEqual(self.score(row).format_reward, -.1)

    def test_final_stop_does_not_execute_or_validate_its_ignored_ids(self):
        a = self.directory[0]["id"]
        row = self.run_actions([self.read(a), {"ids": [a, "unknown"] * 4, "stop": True}], max_calls=1)
        self.assertEqual(row.termination_reason, "stop")
        self.assertEqual(row.n_read_calls, 1)
        self.assertIsNone(row.decisions[-1].tool_result)
        self.assertEqual(row.decisions[-1].format_violations, [])
        self.assertFalse(self.score(row).format_failure)

    def test_json_and_schema_failures_remain_scored_protocol_failures(self):
        for action in ["not json", '{"ids":"x","stop":false}', '{"ids":[],"stop":false,"extra":1}']:
            with self.subTest(action=action):
                row = self.run_actions([action])
                self.assertEqual(row.termination_reason, "protocol_failure")
                self.assertEqual(row.n_read_calls, 0)
                self.assertEqual(row.decisions[0].format_violations, ["invalid_json_or_schema"])
                self.assertEqual(self.score(row).format_reward, -.1)

    def test_id_spelling_is_never_repaired(self):
        a = self.directory[2]["id"]
        alias = a.replace("r_00002_", "r_02_")
        row = self.run_actions([self.read(alias), self.read(a), self.stop])
        self.assertEqual(row.decisions[0].format_violations, ["unknown_id"])
        self.assertEqual(row.decisions[1].format_violations, [])
        self.assertEqual(row.decisions[1].tool_result["new_ids"], [a])

    def test_retrieval_history_is_isolated_per_rollout_in_full_batch(self):
        a = self.directory[0]["id"]
        class Batch:
            calls = 0
            def complete_batch(self, messages, seeds, max_batch_tokens):
                self.calls += 1
                text = json.dumps({"ids": [a], "stop": False} if self.calls <= 2 else {"ids": [], "stop": True})
                return [Completion(text) for _ in seeds]
        env = MemoryEnvironment(self.store, self.builder.version, id_protocol="strict_v2")
        rows = env.run_batch(self.example, Batch(), list(range(8)))
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row.termination_reason, "stop")
            self.assertEqual(row.n_read_calls, 2)
            self.assertEqual(row.decisions[0].format_violations, [])
            self.assertEqual(row.decisions[1].format_violations, ["repeated_read"])
            self.assertEqual(self.score(row).format_reward, -.1)

    def test_historical_protocols_still_allow_repeated_reads(self):
        a = self.directory[0]["id"]
        for protocol in ["strict_v1", "repair_v2"]:
            with self.subTest(protocol=protocol), patch.object(self.store, "record", wraps=self.store.record) as fetch:
                row = self.run_actions([self.read(a), self.read(a), self.stop], protocol=protocol)
                self.assertEqual(fetch.call_count, 2)
                self.assertFalse(self.score(row).format_failure)

    def test_protocol_metrics_count_consumed_calls_and_audit_budget_action_separately(self):
        a, b = [x["id"] for x in self.directory[:2]]
        # Three calls: read A, read new B with repeated A, then entirely repeat A.
        # The fourth decision is inspected and penalized, but is not a tool call.
        row = self.run_actions([self.read(a), self.read(a, b), self.read(a), self.read(a)], max_calls=3)
        report = protocol_metrics([row])
        self.assertEqual(report["repeated_read_rate"], 1)
        self.assertEqual(report["repeated_read_calls"], 2)
        self.assertAlmostEqual(report["repeated_read_call_rate"], 2 / 3)
        self.assertEqual(report["no_new_evidence_calls"], 1)
        self.assertAlmostEqual(report["no_new_evidence_call_rate"], 1 / 3)
        self.assertEqual(report["format_decisions_repeated_read"], 3)
        self.assertEqual(report["format_decisions_read_budget_exceeded"], 1)


if __name__ == "__main__":
    unittest.main()
