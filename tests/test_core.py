from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from agmem.core import CapacityError, Completion, Example, digest
from agmem.data import InvalidSample, adapt_row, clean_history
from agmem.env import MemoryEnvironment, parse_action
from agmem.prompts import reader_messages, task_messages
from agmem.pipeline import collect_group, metrics, scored_rollout
from agmem.rewards import advantages, outcome, text_f1
from agmem.services import DirectoryBuilder
from agmem.store import Store


def sample_history(n=7):
    return {"haystack_sessions": [[{"role": "user", "name": "Alice", "content": f"event {i}", "has_answer": i == 3, "num_tokens": 99, "unexpected_gold": "SECRET"}] for i in range(n)],
            "haystack_session_ids": [f"answer_session_{i}" for i in range(n)]}


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class Writer:
    identity = {"model": "fixture-only", "revision": "1"}
    def __init__(self):
        self.calls = []
    def complete(self, messages, max_tokens, cache_scope=None):
        self.calls.append(messages)
        return "A factual session summary"


class ScriptedPolicy:
    policy_version = 0
    def __init__(self, actions):
        self.actions, self.messages, self.index = actions, [], 0
    def complete(self, messages, seed):
        self.messages.append(deepcopy(messages))
        action = self.actions[self.index]
        self.index += 1
        return Completion(action if isinstance(action, str) else json.dumps(action))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.records = clean_history(sample_history())
        self.memory_id = self.store.add_memory(self.records)
        self.writer = Writer()
        self.config = {"lookback_sessions": 3, "content_max_tokens": 300}
        self.builder = DirectoryBuilder(self.store, self.writer, self.config, Tokenizer())
        self.directory = self.builder.build(self.memory_id)
        self.example = Example("fixture/train/0", "fixture", "train", self.memory_id, "Which event?", gold_answers=["SECRET_GOLD"])

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_one_session_one_entry_annotation_allowlist(self):
        self.assertEqual(len(self.directory), 7)
        text = json.dumps(self.records)
        for forbidden in ("has_answer", "num_tokens", "unexpected_gold", "SECRET", "answer_session"):
            self.assertNotIn(forbidden, text)

    def test_empty_session_positions_are_preserved(self):
        raw = sample_history()
        raw["haystack_sessions"][2] = []
        memory_id = self.store.add_memory(clean_history(raw))
        entries = self.builder.build(memory_id)
        self.assertEqual(len(entries), 7)
        self.assertEqual(entries[2]["time_order"]["order"], 2)
        self.assertEqual(entries[2]["content"], "Empty session; no recorded messages.")
        record = self.store.record(self.store.memory(memory_id)[2])
        self.assertEqual(record["raw_text"], "")

    def test_writer_window_and_complete_history_reuse(self):
        value = json.loads(self.writer.calls[6][1]["content"])
        self.assertEqual([row["order"] for row in value["PRECEDING_CONTEXT"]], [3, 4, 5])
        self.assertEqual(value["TARGET_SESSION"]["order"], 6)
        first = json.loads(self.writer.calls[0][1]["content"])
        self.assertEqual(first["PRECEDING_CONTEXT"], [])
        appended = self.store.add_memory(clean_history(sample_history(8)))
        self.builder.build(appended)
        self.assertEqual(len(self.writer.calls), 15)
        changed = sample_history()
        changed["haystack_sessions"][4][0]["content"] = "changed preceding content"
        self.builder.build(self.store.add_memory(clean_history(changed)))
        self.assertEqual(len(self.writer.calls), 22)

    def test_feedback_stop_reset_and_final_evidence(self):
        key = self.directory[2]["id"]
        policy = ScriptedPolicy([{"ids": [key], "stop": False}, {"ids": [], "stop": True}])
        env = MemoryEnvironment(self.store, self.builder.version)
        result = env.run(self.example, policy, 42)
        initial = json.loads(policy.messages[0][1]["content"])
        second = json.loads(policy.messages[1][1]["content"])
        self.assertEqual(initial["retrieved_evidence"], [])
        self.assertIn("event 2", second["retrieved_evidence"][0]["raw_text"])
        self.assertEqual(result.n_read_calls, 1)
        self.assertEqual(result.termination_reason, "stop")
        final = json.loads(task_messages(self.example, result.evidence)[1]["content"])
        self.assertNotIn("directory", final)
        self.assertIn("content", final["retrieved_evidence"][0])
        self.assertNotIn("SECRET_GOLD", json.dumps(policy.messages + task_messages(self.example, result.evidence)))
        reset = env.run(self.example, ScriptedPolicy([{"ids": [], "stop": True}]), 3)
        self.assertEqual(reset.evidence, [])
        self.assertEqual(reset.n_read_calls, 0)

    def test_fenced_reader_json_reads_memory_without_rewriting_generated_text(self):
        key = self.directory[2]["id"]
        read = '```json\n' + json.dumps({"ids": [key], "stop": False}) + '\n```\n'
        stop = '```\n{"ids":[],"stop":true}\n```'
        result = MemoryEnvironment(self.store, self.builder.version).run(self.example, ScriptedPolicy([read, stop]), 42)
        self.assertEqual(result.termination_reason, "stop")
        self.assertEqual(result.n_read_calls, 1)
        self.assertIn("event 2", result.evidence[0]["raw_text"])
        self.assertEqual(result.decisions[0].completion.text, read)
        self.assertEqual(result.decisions[1].completion.text, stop)

    def test_fenced_json_still_rejects_invalid_actions_and_surrounding_prose(self):
        action = '{"ids":[],"stop":true}'
        invalid = [
            '```json\n{"ids":"x","stop":true}\n```',
            '```json\n{"ids":[],"stop":"true"}\n```',
            '```json\n{"ids":[],"stop":true,"extra":1}\n```',
            'Here is the action:\n```json\n' + action + '\n```',
            '```json\n' + action + '\n```\nExtra explanation.',
            '```json\n' + action + '\n```\n```json\n' + action + '\n```',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text)

    def test_batch_cost_dedup_unknown_and_capacity(self):
        key = self.directory[0]["id"]
        env = MemoryEnvironment(self.store, self.builder.version)
        actions = [{"ids": [key], "stop": False}, {"ids": [key], "stop": False}, {"ids": [], "stop": True}]
        result = env.run(self.example, ScriptedPolicy(actions), 0)
        self.assertEqual((result.n_read_calls, len(result.evidence)), (2, 1))
        invalid = [{"ids": [key, key], "stop": False}, {"ids": ["other-memory"], "stop": False}, {"ids": [], "stop": True}]
        result = env.run(self.example, ScriptedPolicy(invalid), 0)
        self.assertEqual((result.n_read_calls, result.evidence), (2, []))
        def reject(*args):
            raise CapacityError("fixture")
        env.capacity = reject
        result = env.run(self.example, ScriptedPolicy([actions[0], actions[-1]]), 0)
        self.assertEqual(result.n_read_calls, 1)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.decisions[0].tool_result["error"]["code"], "context_capacity_exceeded")

    def test_ten_ids_cost_one_call_and_eleven_are_rejected_atomically(self):
        memory_id = self.store.add_memory(clean_history(sample_history(12)))
        entries = self.builder.build(memory_id)
        example = deepcopy(self.example)
        example.memory_id = memory_id
        keys = [entry["id"] for entry in entries]
        stop = {"ids": [], "stop": True}
        env = MemoryEnvironment(self.store, self.builder.version)
        one = env.run(example, ScriptedPolicy([{"ids": keys[:1], "stop": False}, stop]), 0)
        ten = env.run(example, ScriptedPolicy([{"ids": keys[:10], "stop": False}, stop]), 0)
        eleven = env.run(example, ScriptedPolicy([{"ids": keys[:11], "stop": False}, stop]), 0)
        self.assertEqual([one.n_read_calls, ten.n_read_calls, eleven.n_read_calls], [1, 1, 1])
        self.assertEqual([len(one.evidence), len(ten.evidence), len(eleven.evidence)], [1, 10, 0])

    def test_protocol_and_budget_are_not_synthetic_stops(self):
        env = MemoryEnvironment(self.store, self.builder.version, max_calls=1)
        key = self.directory[0]["id"]
        read = {"ids": [key], "stop": False}
        row = env.run(self.example, ScriptedPolicy([read, read]), 0)
        self.assertEqual((row.termination_reason, row.n_read_calls), ("max_read_calls", 1))
        row = env.run(self.example, ScriptedPolicy(['{"ids":["x"],"stop":"true"}']), 0)
        self.assertEqual(row.termination_reason, "protocol_failure")
        self.assertEqual(row.n_read_calls, 0)

    def test_choice_and_text_rewards(self):
        example = Example("x", "x", "train", "m", "q", options={"A": "foo", "B": "bar"}, gold_answers=["A"], answer_mode="choice")
        self.assertEqual(outcome(example, "(a)"), 1)
        self.assertEqual(outcome(example, "B"), 0)
        self.assertEqual(outcome(example, "A or B"), 0)
        self.assertAlmostEqual(text_f1("cat cat dog", "cat dog dog"), 2 / 3)
        self.assertEqual(text_f1("2.5", "25"), 0)
        self.assertEqual(text_f1("-3", "3"), 0)
        self.assertEqual(text_f1("北京", "北京"), 1)
        self.assertEqual(text_f1("", ""), 0)
        self.assertEqual(advantages([1] * 8), [0] * 8)
        self.assertAlmostEqual(sum(advantages([0, 1] * 4)), 0)

    def test_format_reward_covers_json_schema_stop_and_batch_failures_once(self):
        key = self.directory[0]["id"]
        stop = {"ids": [], "stop": True}
        env = MemoryEnvironment(self.store, self.builder.version)
        config = {"reward": {"read_call_cost": 0.01, "format_penalty": 0.1}}
        agent = SimpleNamespace(answer=lambda example, evidence: ("SECRET_GOLD", "hash"))
        cases = [
            ("not JSON", 0),
            ('{"ids":[],["stop":false}', 0),
            ({"ids": [], "stop": "true"}, 0),
            ({"ids": [], "stop": True, "extra": 1}, 0),
            ({"ids": [], "stop": False}, 1),
            ({"ids": [key] * 11, "stop": False}, 1),
            ({"ids": [key, key], "stop": False}, 1),
        ]
        for action, calls in cases:
            with self.subTest(action=action):
                row = scored_rollout(config, env, ScriptedPolicy([action, stop]), agent, self.example, 42)
                self.assertTrue(row.format_failure)
                self.assertEqual(row.outcome, 1.0)  # Even a correct final answer keeps the format penalty.
                self.assertEqual(row.format_reward, -0.1)
                self.assertEqual(row.n_read_calls, calls)
                self.assertAlmostEqual(row.reward, 1 - 0.01 * calls - 0.1)
        row = scored_rollout(config, env, ScriptedPolicy([cases[-1][0]] * 2 + [stop]), agent, self.example, 42)
        self.assertEqual(row.termination_reason, "stop")
        self.assertEqual(row.read_cost, 0.02)
        self.assertEqual(row.format_reward, -0.1)  # Not -0.2 for two invalid batches.
        report = metrics([row])
        self.assertEqual(report["pooled"]["protocol_failure_rate"], 0.0)
        self.assertEqual(report["pooled"]["format_failure_rate"], 1.0)
        self.assertEqual(report["source_macro"]["format_reward"], -0.1)
        # The final decision is still checked even if the read budget prevents execution.
        env.max_calls = 1
        row = scored_rollout(config, env, ScriptedPolicy([
            {"ids": [key], "stop": False}, {"ids": [], "stop": False}
        ]), agent, self.example, 42)
        self.assertEqual(row.termination_reason, "max_read_calls")
        self.assertIsNone(row.decisions[-1].tool_result)
        self.assertEqual(row.read_cost, 0.01)
        self.assertEqual(row.format_reward, -0.1)

    def test_stop_ignores_nonempty_ids_without_reading_or_format_penalty(self):
        key = self.directory[0]["id"]
        config = {"reward": {"read_call_cost": 0.01, "format_penalty": 0.1}}
        agent = SimpleNamespace(answer=lambda example, evidence: ("SECRET_GOLD", "hash"))
        env = MemoryEnvironment(self.store, self.builder.version, max_calls=1)
        for ids in ([], [key], ["unknown"], [key] * 11):
            for read_first in (False, True):
                with self.subTest(ids=ids, read_first=read_first):
                    stop = '```json\n' + json.dumps({"ids": ids, "stop": True}) + '\n```'
                    actions = ([{"ids": [key], "stop": False}] if read_first else []) + [stop]
                    row = scored_rollout(config, env, ScriptedPolicy(actions), agent, self.example, 42)
                    self.assertEqual(row.termination_reason, "stop")
                    self.assertFalse(row.format_failure)
                    self.assertEqual(row.format_reward, 0)
                    self.assertEqual(row.n_read_calls, int(read_first))
                    self.assertEqual(len(row.evidence), int(read_first))
                    self.assertAlmostEqual(row.reward, 1 - 0.01 * int(read_first))
                    self.assertEqual(row.decisions[-1].completion.text, stop)
                    self.assertEqual(row.decisions[-1].action, {"ids": ids, "stop": True})
                    self.assertIsNone(row.decisions[-1].tool_result)
        # A valid stop must not erase a previous invalid retrieval batch.
        row = scored_rollout(config, env, ScriptedPolicy([
            {"ids": [], "stop": False}, {"ids": [key], "stop": True}
        ]), agent, self.example, 42)
        self.assertEqual(row.termination_reason, "stop")
        self.assertEqual(row.format_reward, -0.1)

    def test_valid_actions_unknown_ids_and_capacity_do_not_get_format_penalties(self):
        key = self.directory[0]["id"]
        stop, read = {"ids": [], "stop": True}, {"ids": [key], "stop": False}
        config = {"reward": {"read_call_cost": 0.01, "format_penalty": 0.1}}
        agent = SimpleNamespace(answer=lambda example, evidence: ("wrong", "hash"))
        env = MemoryEnvironment(self.store, self.builder.version)
        cases = [([stop], 0), ([read, stop], 1), ([read, read, stop], 2),
                 (['```json\n' + json.dumps(read) + '\n```', stop], 1),
                 ([{"ids": ["unknown"], "stop": False}, stop], 1)]
        for actions, calls in cases:
            row = scored_rollout(config, env, ScriptedPolicy(actions), agent, self.example, 42)
            self.assertFalse(row.format_failure)
            self.assertEqual(row.format_reward, 0.0)
            self.assertEqual(row.reward, -0.01 * calls)
        def reject(*args):
            raise CapacityError("fixture")
        env.capacity = reject
        row = scored_rollout(config, env, ScriptedPolicy([read, stop]), agent, self.example, 42)
        self.assertFalse(row.format_failure)
        self.assertEqual(row.reward, -0.01)
        # Old standalone configurations, and explicit zero, preserve the original reward.
        for reward_config in ({"read_call_cost": 0.01}, {"read_call_cost": 0.01, "format_penalty": 0.0}):
            row = scored_rollout({"reward": reward_config}, env, ScriptedPolicy(["bad JSON"]), agent, self.example, 42)
            self.assertTrue(row.format_failure)
            self.assertEqual(row.reward, 0.0)

    def test_group_advantages_use_format_adjusted_rewards(self):
        key = self.directory[0]["id"]
        stop = {"ids": [], "stop": True}
        read = {"ids": [key], "stop": False}
        duplicate = {"ids": [key, key], "stop": False}
        policy = ScriptedPolicy([stop, "bad JSON", read, stop, duplicate, duplicate, stop] * 2)
        agent = SimpleNamespace(answer=lambda example, evidence: ("wrong", "hash"))
        config = {"seed": 42, "reward": {"read_call_cost": 0.01, "format_penalty": 0.1},
                  "training": {"advantage_epsilon": 1e-6}}
        group = collect_group(config, MemoryEnvironment(self.store, self.builder.version), policy, agent, self.example, 0)
        expected = [0.0, -0.1, -0.01, -0.12] * 2
        for row, reward, advantage in zip(group, expected, advantages(expected)):
            self.assertAlmostEqual(row.reward, reward)
            self.assertAlmostEqual(row.advantage, advantage)

    def test_native_choice_adapters_and_null_padding(self):
        row = {"question": "q", "answer": "Correct text", "all_options": ["(a) Wrong", "(b) Correct text"], "correct_answer": "(b)"}
        example = adapt_row(row, "personamem_32k", "train", 0)
        self.assertEqual(example.gold_answers, ["B"])
        example = adapt_row({"question": "问题？ A 甲 B 乙 C 丙 D 丁", "answer": "C"}, "zh4o", "train", 0)
        self.assertEqual(example.options["C"], "丙")
        self.assertEqual(example.gold_answers, ["C"])
        with self.assertRaises(InvalidSample):
            adapt_row({"question": "问题？ A 甲 B 乙 C 丙 D 丁", "answer": "coleção"}, "zh4o", "train", 0)
        row = {"question": "q", "answer": "good", "incorrect_answers": '["wrong1","wrong2",null]'}
        example = adapt_row(row, "personamem_v2_128k", "train", 0)
        self.assertEqual(len(example.options), 3)
        self.assertEqual(example.options[example.gold_answers[0]], "good")
        self.assertEqual(as_mapping(example), as_mapping(adapt_row(row, "personamem_v2_128k", "train", 0)))
        self.assertNotIn("gold_answers", example.public_task())


def as_mapping(example):
    return example.public_task(), example.gold_answers


if __name__ == "__main__":
    unittest.main()
