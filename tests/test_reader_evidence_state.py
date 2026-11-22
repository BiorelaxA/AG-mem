"""Reader state exposes successful ID-to-memory bindings without changing actions."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest

from agmem.core import CapacityError, Completion, Example, digest
from agmem.env import MemoryEnvironment
from agmem.prompts import configured_reader_input_contract, reader_input_contract, reader_messages, task_messages
from agmem.rollout_pipeline import collect_candidate_groups


class MemoryStore:
    def __init__(self):
        self.reads = []

    def memory(self, memory_id):
        return [{"id": f"{memory_id}-{suffix}", "source_session_id": memory_id,
                 "time": "2026-01-01", "order": index,
                 "raw_text": f"original-body-for-{memory_id}-{suffix}"}
                for index, suffix in enumerate(("a", "b"))]

    def directory(self, memory_id, version):
        return [{"id": record["id"], "content": f"summary-{record['id']}"}
                for record in self.memory(memory_id)]

    def record(self, reference):
        self.reads.append(reference["id"])
        return reference


class ScriptedReader:
    policy_version = 0

    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages, *, seed):
        self.messages.append(deepcopy(messages))
        return Completion(json.dumps(next(self.actions)), token_ids=[1], old_logprobs=[-0.2])


def example(index=0):
    return Example(f"fixture/train/{index}", "fixture", "train", f"memory-{index}",
                   f"query-{index}", observation="public-observation",
                   options={"A": "one", "B": "two"}, gold_answers=["private-gold"],
                   judge_metadata={"private-judge": "private-value"})


def state(messages):
    return json.loads(messages[1]["content"])


class ReaderEvidenceStateTests(unittest.TestCase):
    def environment(self, store, **kwargs):
        return MemoryEnvironment(store, "directory-v1", id_protocol="strict_v2",
                                 prompt_version="evidence_ids_v2", **kwargs)

    def test_success_and_mixed_reads_keep_ids_bound_to_original_memory_once(self):
        store, item = MemoryStore(), example()
        first, second = [record["id"] for record in store.memory(item.memory_id)]
        reader = ScriptedReader([
            {"ids": [first], "stop": False},
            {"ids": [first, second, "unknown"], "stop": False},
            {"ids": [second, first], "stop": False},
            {"ids": [], "stop": True},
        ])
        previews = []
        def capacity(reader_input, task_input):
            previews.append(deepcopy(reader_input))
            self.assertNotIn("read_ids", state(task_input))
        result = self.environment(store, capacity=capacity).run(item, reader, 12)
        states = [state(messages) for messages in reader.messages]
        self.assertEqual([value["read_ids"] for value in states],
                         [[], [first], [first, second], [first, second]])
        self.assertEqual(previews[0], reader.messages[1])
        self.assertEqual(result.n_read_calls, 3)
        self.assertEqual(result.termination_reason, "stop")
        self.assertEqual(store.reads, [first, second])
        for messages, value in zip(reader.messages, states):
            self.assertEqual(value["query"], item.query)
            self.assertEqual(value["observation"], item.observation)
            self.assertEqual(value["options"], item.options)
            self.assertEqual(value["read_ids"], [record["id"] for record in value["retrieved_evidence"]])
            for record in value["retrieved_evidence"]:
                self.assertEqual(record["raw_text"], f"original-body-for-{record['id']}")
                self.assertEqual(record["content"], f"summary-{record['id']}")
                self.assertEqual(messages[1]["content"].count(record["raw_text"]), 1)
            self.assertNotIn("private-gold", str(messages))
            self.assertNotIn("private-judge", str(messages))
        self.assertEqual(states[2]["last_tool_error"]["invalid_ids"], ["unknown"])
        self.assertTrue(states[3]["last_tool_error"]["no_new_evidence"])
        self.assertEqual(states[3]["last_tool_error"]["new_ids"], [])
        self.assertIn("Never request an ID in read_ids again", reader.messages[0][0]["content"])
        self.assertIn('{"ids":[],"stop":true}', reader.messages[0][0]["content"])

    def test_unknown_and_capacity_rejected_ids_never_become_read_ids(self):
        store, item = MemoryStore(), example()
        first = store.memory(item.memory_id)[0]["id"]
        reader = ScriptedReader([
            {"ids": ["unknown"], "stop": False},
            {"ids": [first], "stop": False},
            {"ids": [], "stop": True},
        ])
        previews = []
        def reject(reader_input, task_input):
            previews.append(state(reader_input))
            raise CapacityError("fixture capacity limit")
        result = self.environment(store, capacity=reject).run(item, reader, 7)
        self.assertEqual([state(messages)["read_ids"] for messages in reader.messages], [[], [], []])
        self.assertEqual([value["read_ids"] for value in previews], [[first]])
        self.assertEqual(result.evidence, [])
        self.assertEqual(state(reader.messages[-1])["last_tool_error"]["code"], "context_capacity_exceeded")

    def test_capacity_preview_uses_new_state_for_all_id_protocols(self):
        for protocol in ("strict_v1", "strict_v2", "repair_v2"):
            with self.subTest(protocol=protocol):
                store, item = MemoryStore(), example()
                first = store.memory(item.memory_id)[0]["id"]
                reader = ScriptedReader([{"ids": [first], "stop": False}, {"ids": [], "stop": True}])
                previews = []
                env = MemoryEnvironment(store, "directory-v1", id_protocol=protocol,
                                        prompt_version="evidence_ids_v2",
                                        capacity=lambda reader_input, task_input: previews.append(reader_input))
                env.run(item, reader, 17)
                self.assertEqual(previews, [reader.messages[1]])
                self.assertEqual(state(previews[0])["read_ids"], [first])

    def test_batched_rollouts_and_reset_keep_independent_read_ids(self):
        store, item = MemoryStore(), example()
        keys = [record["id"] for record in store.memory(item.memory_id)]
        observed = []
        class Reader:
            def complete_batch(self, messages, seeds, *, max_batch_tokens):
                values = [state(message) for message in messages]
                observed.append(values)
                return [Completion(json.dumps({"ids": [] if value["read_ids"] else [keys[index]],
                                               "stop": bool(value["read_ids"])}))
                        for index, value in enumerate(values)]
        env = self.environment(store)
        rows = env.run_batch(item, Reader(), [10, 20])
        self.assertEqual([value["read_ids"] for value in observed[0]], [[], []])
        self.assertEqual([value["read_ids"] for value in observed[1]], [[keys[0]], [keys[1]]])
        self.assertEqual([[record["id"] for record in row.evidence] for row in rows], [[keys[0]], [keys[1]]])
        fresh = ScriptedReader([{"ids": [], "stop": True}])
        env.run(item, fresh, 30)
        self.assertEqual(state(fresh.messages[0])["read_ids"], [])

    def test_cross_query_pipeline_has_no_read_id_or_memory_leak(self):
        store = MemoryStore()
        env = self.environment(store)
        observed = []
        class Reader:
            policy_version = 0
            def complete_batch(self, messages, seeds, *, max_batch_tokens):
                observed.append([state(message) for message in messages])
                return [Completion(json.dumps({"ids": [] if value["read_ids"] else [value["directory"][0]["id"]],
                                               "stop": bool(value["read_ids"])}),
                                   token_ids=[1], old_logprobs=[-0.2])
                        for value in observed[-1]]
        def answer(item, evidence, metadata):
            self.assertEqual([row["id"] for row in evidence], [f"{item.memory_id}-a"])
            return "fixture", item.sample_id
        def score(item, row):
            row.judge_score, row.reward = 1.0, 1.0
            return row
        config = {"seed": 1, "training": {"group_size": 2},
                  "execution": {"sampling_group_concurrency": 4, "reader_batch_size": 4,
                                "task_agent_workers": 2}}
        groups = list(collect_candidate_groups(config, env, Reader(), SimpleNamespace(answer=answer),
                                               SimpleNamespace(score=score), ((example(i), i) for i in range(4))))
        self.assertEqual(len(groups), 4)
        for batch in observed:
            for value in batch:
                expected_id = f"memory-{value['query'].split('-')[-1]}-a"
                self.assertIn(value["read_ids"], ([], [expected_id]))
                self.assertEqual(value["read_ids"], [record["id"] for record in value["retrieved_evidence"]])

    def test_legacy_hash_and_other_role_inputs_are_preserved(self):
        item = Example("fixture/train/0", "fixture", "train", "m", "Which event?",
                       observation="Today is 2026-01-01", options={"A": "one", "B": "two"})
        directory = [{"id": "r_a", "content": "summary"}]
        evidence = [{"id": "r_a", "raw_text": "original", "content": "summary"}]
        error = {"code": "repeated_read", "already_read_ids": ["r_a"]}
        legacy = reader_messages(item, directory, evidence, error, max_ids=4, id_protocol="strict_v2")
        self.assertEqual(digest(legacy), "16946ccb69ced58f22554c12e7f908046c69ea4b5a7576a98e12dd5bcd0158f6")
        self.assertNotIn("read_ids", state(legacy))
        task_before = task_messages(item, evidence)
        reader_messages(item, directory, evidence, error, prompt_version="evidence_ids_v2")
        self.assertEqual(task_messages(item, evidence), task_before)
        self.assertNotIn("read_ids", state(task_before))

    def test_input_contract_hashes_rendered_system_and_rejects_unknown_versions(self):
        for version in ("legacy_v1", "evidence_ids_v2"):
            for max_ids, protocol in ((4, "strict_v2"), (2, "repair_v2")):
                messages = reader_messages(example(), [], [], max_ids=max_ids, id_protocol=protocol,
                                           prompt_version=version)
                contract = reader_input_contract(version, max_ids=max_ids, id_protocol=protocol)
                self.assertEqual(contract["reader_prompt_sha256"],
                                 hashlib.sha256(messages[0]["content"].encode()).hexdigest())
                self.assertEqual(contract["prompt_version"], version)
        for bad in ("", "v3", None, {}, []):
            with self.subTest(version=bad), self.assertRaises(ValueError):
                reader_input_contract(bad)
        with self.assertRaises(ValueError):
            MemoryEnvironment(MemoryStore(), "version", prompt_version="unknown")
        self.assertEqual(configured_reader_input_contract({}), reader_input_contract())
        self.assertEqual(configured_reader_input_contract({"reader": {"prompt_version": "evidence_ids_v2"},
                                                          "retrieval": {"max_ids_per_call": 2,
                                                                        "id_protocol": "strict_v2"}}),
                         reader_input_contract("evidence_ids_v2", max_ids=2, id_protocol="strict_v2"))


if __name__ == "__main__":
    unittest.main()
