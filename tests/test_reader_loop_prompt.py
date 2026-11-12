"""Exercise visible READ budgets through real multi-decision environments."""
from copy import deepcopy
import hashlib
import json
import unittest

from agmem.core import Completion
from agmem.env import MemoryEnvironment
from agmem.prompts import configured_reader_input_contract, reader_input_contract, reader_messages
from agmem.resume import validate_reader_input_contract
from tests.test_reader_evidence_state import MemoryStore, ScriptedReader, example, state


class ReaderLoopPromptTests(unittest.TestCase):
    def test_successful_reads_reenter_reader_with_updated_budget_and_evidence(self):
        for protocol in ("strict_v1", "strict_v2", "repair_v2"):
            with self.subTest(protocol=protocol):
                store, item = MemoryStore(), example()
                a, b = [record["id"] for record in store.memory(item.memory_id)]
                reader = ScriptedReader([{"ids": [a], "stop": False},
                                         {"ids": [b], "stop": False},
                                         {"ids": [], "stop": True}])
                previews = []
                env = MemoryEnvironment(store, "v", max_ids=1, max_calls=2,
                                        id_protocol=protocol, prompt_version="evidence_loop_v3",
                                        capacity=lambda messages, _: previews.append(deepcopy(messages)))
                row = env.run(item, reader, 10)
                values = [state(messages) for messages in reader.messages]
                self.assertEqual([v["read_calls_used"] for v in values], [0, 1, 2])
                self.assertEqual([v["read_calls_remaining"] for v in values], [2, 1, 0])
                self.assertEqual([v["max_ids_per_call"] for v in values], [1, 1, 1])
                self.assertEqual([v["read_ids"] for v in values], [[], [a], [a, b]])
                self.assertTrue(all("last_tool_error" not in v for v in values))
                self.assertEqual(previews, reader.messages[1:])
                self.assertEqual(row.n_read_calls, 2)
                self.assertEqual(row.termination_reason, "stop")
                system = reader.messages[0][0]["content"]
                self.assertIn("Select exactly 1 directory ID per READ call", system)
                self.assertIn("up to 2 READ calls", system)
                self.assertIn("then you will be called again", system)
                self.assertNotIn("private-gold", str(reader.messages))

    def test_budget_counts_failed_attempts_instead_of_number_of_read_ids(self):
        store, item = MemoryStore(), example()
        a = store.memory(item.memory_id)[0]["id"]
        reader = ScriptedReader([{"ids": ["unknown"], "stop": False},
                                 {"ids": [a], "stop": False},
                                 {"ids": [], "stop": True}])
        env = MemoryEnvironment(store, "v", max_ids=1, max_calls=3, id_protocol="strict_v2",
                                prompt_version="evidence_loop_v3")
        row = env.run(item, reader, 10)
        values = [state(messages) for messages in reader.messages]
        self.assertEqual([v["read_calls_used"] for v in values], [0, 1, 2])
        self.assertEqual([v["read_calls_remaining"] for v in values], [3, 2, 1])
        self.assertEqual([v["read_ids"] for v in values], [[], [], [a]])
        self.assertEqual(values[1]["last_tool_error"]["code"], "unknown_or_unauthorized_id")
        self.assertNotIn("last_tool_error", values[2])
        self.assertEqual(row.n_read_calls, 2)

    def test_parallel_rollouts_keep_budgets_independent(self):
        store, item = MemoryStore(), example()
        a, b = [record["id"] for record in store.memory(item.memory_id)]
        observed = []

        class Reader:
            def complete_batch(self, messages, seeds, *, max_batch_tokens):
                observed.extend((seed, state(message)) for seed, message in zip(seeds, messages))
                actions = {10: {"ids": [], "stop": True}, 20: {"ids": [a], "stop": False},
                           21: {"ids": [b], "stop": False}, 22: {"ids": [], "stop": True}}
                return [Completion(json.dumps(actions[seed])) for seed in seeds]

        env = MemoryEnvironment(store, "v", max_ids=1, max_calls=2, id_protocol="strict_v2",
                                prompt_version="evidence_loop_v3")
        rows = env.run_batch(item, Reader(), [10, 20])
        self.assertEqual([(seed, v["read_calls_remaining"]) for seed, v in observed],
                         [(10, 2), (20, 2), (21, 1), (22, 0)])
        self.assertEqual([row.n_read_calls for row in rows], [0, 2])

    def test_v3_contract_binds_budget_and_rejects_old_prompt_resume(self):
        config = {"reader": {"prompt_version": "evidence_loop_v3"},
                  "retrieval": {"max_ids_per_call": 1, "max_read_calls": 3, "id_protocol": "strict_v2"}}
        contract = configured_reader_input_contract(config)
        messages = reader_messages(example(), [], [], max_ids=1, max_calls=3,
                                   prompt_version="evidence_loop_v3", id_protocol="strict_v2")
        self.assertEqual(contract["reader_prompt_sha256"], hashlib.sha256(messages[0]["content"].encode()).hexdigest())
        self.assertEqual(contract["state_schema"], "evidence_loop_v3")
        old = deepcopy(config)
        old["reader"]["prompt_version"] = "evidence_ids_v2"
        with self.assertRaisesRegex(ValueError, "prompt_version"):
            validate_reader_input_contract({"config": old,
                                            "reader_input_contract": configured_reader_input_contract(old)}, config)
        changed = deepcopy(config)
        changed["retrieval"]["max_read_calls"] = 4
        with self.assertRaisesRegex(ValueError, "contract"):
            validate_reader_input_contract({"config": config, "reader_input_contract": contract}, changed)

    def test_previous_versions_ignore_new_budget_fields_and_preserve_contracts(self):
        for version in ("legacy_v1", "evidence_ids_v2"):
            with self.subTest(version=version):
                original = reader_messages(example(), [], [], max_ids=1, id_protocol="strict_v2",
                                           prompt_version=version)
                explicit = reader_messages(example(), [], [], max_ids=1, id_protocol="strict_v2",
                                           prompt_version=version, max_calls=3, read_calls_used=2)
                self.assertEqual(original, explicit)
                self.assertEqual(reader_input_contract(version, max_calls=3),
                                 reader_input_contract(version, max_calls=8))
        self.assertEqual(reader_input_contract("evidence_ids_v2", max_ids=1, id_protocol="strict_v2")
                         ["reader_prompt_sha256"],
                         "09537ea34aa2a4677ddbc4357c18bdc8046d405e86be8c7a834a43ebe11ceec2")


if __name__ == "__main__":
    unittest.main()
