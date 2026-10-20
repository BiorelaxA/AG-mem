from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from agmem.core import Example, InfrastructureError
from agmem.data import clean_history
from agmem.env import MemoryEnvironment
from agmem.pipeline import collect_group
from agmem.services import DirectoryBuilder
from agmem.store import Store


def history(texts):
    return clean_history({"haystack_sessions": [[{"role": "user", "content": text}] if text else [] for text in texts],
                          "haystack_session_ids": [f"session_{i}" for i in range(len(texts))]})


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class Writer:
    identity = {"model": "fixture", "model_revision": "v1", "base_url": "fixture"}

    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def complete(self, messages, max_tokens, cache_scope=None):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise InfrastructureError("fixture interruption")
        self.calls.append((messages, cache_scope))
        return '{"content":"Recorded facts."}'


WRITER_CONFIG = {"lookback_sessions": 5, "content_max_tokens": 96, "max_output_tokens": 192, "format_retries": 0}


class StoreTests(unittest.TestCase):
    def test_complete_histories_share_but_partial_overlap_keeps_whole_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            first = history(["北京", "Paris"])
            first_id = store.add_memory(first)
            self.assertEqual(store.add_memory(deepcopy(first)), first_id)
            second_id = store.add_memory(history(["北京", "Berlin"]))
            reversed_id = store.add_memory(history(["Paris", "北京"]))
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first_id, reversed_id)
            for index, query in enumerate(["Where did I live?", "When did I move?"]):
                store.add_example(Example(f"q{index}", "fixture", "train", first_id, query, gold_answers=["SECRET"]))
            store.close()
            rows = [json.loads(line) for line in (Path(temporary) / "memory_records.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["records"][0]["messages"], rows[1]["records"][0]["messages"])
            self.assertNotIn("SECRET", (Path(temporary) / "memory_records.jsonl").read_text())
            # Offset indexes are optional, including for multibyte UTF-8 histories.
            for path in Path(temporary).glob("*.index.json"):
                path.unlink()
            reopened = Store(temporary)
            self.assertEqual([row.memory_id for row in reopened.examples("train")], [first_id, first_id])
            self.assertIn("Berlin", reopened.record(reopened.memory(second_id)[1])["raw_text"])
            self.assertIn("北京", reopened.record(reopened.memory(first_id)[0])["raw_text"])
            self.assertEqual(len(reopened.memory_ids()), 3)
            reopened.close()

    def test_directory_resume_only_reuses_finished_entries_of_the_same_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            memory_id = store.add_memory(history(["first", "second", "third"]))
            writer = Writer(fail_after=1)
            builder = DirectoryBuilder(store, writer, WRITER_CONFIG, Tokenizer())
            with self.assertRaises(InfrastructureError):
                builder.build(memory_id)
            self.assertEqual(len((Path(temporary) / "directories.jsonl").read_text().splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "Directory missing"):
                store.directory(memory_id, builder.version)
            store.close()
            reopened = Store(temporary)
            writer = Writer()
            builder = DirectoryBuilder(reopened, writer, WRITER_CONFIG, Tokenizer())
            entries = builder.build(memory_id)
            self.assertEqual(len(entries), 3)
            self.assertEqual(len(writer.calls), 2)
            self.assertEqual([json.loads(messages[1]["content"])["TARGET_SESSION"]["order"]
                              for messages, scope in writer.calls], [1, 2])
            self.assertEqual(len((Path(temporary) / "directories.jsonl").read_text().splitlines()), 3)
            reopened.close()

    def test_jsonl_reload_drives_a_complete_eight_rollout_group(self):
        from agmem.core import Completion

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            memory_id = store.add_memory(history(["Paris", "", "Berlin"]))
            example = Example("q0", "fixture", "train", memory_id, "Where?", gold_answers=["Paris"])
            store.add_example(example)
            builder = DirectoryBuilder(store, Writer(), WRITER_CONFIG, Tokenizer())
            entries = builder.build(memory_id)
            store.close()
            store = Store(temporary)
            self.assertEqual(len(store.directory(memory_id, builder.version)), 3)
            self.assertEqual(store.directory(memory_id, builder.version)[1]["content"], "Empty session; no recorded messages.")

            class Policy:
                policy_version = 0

                def complete(self, messages, seed):
                    value = json.loads(messages[1]["content"])
                    action = {"ids": [], "stop": True} if value["retrieved_evidence"] else {"ids": [entries[0]["id"]], "stop": False}
                    return Completion(json.dumps(action))

            class Agent:
                def answer(self, example, evidence):
                    return "Paris", "fixture"

            group = collect_group({"seed": 42, "training": {"advantage_epsilon": 1e-6}, "reward": {"read_call_cost": 0.01}},
                                  MemoryEnvironment(store, builder.version), Policy(), Agent(), store.examples("train")[0], 0)
            self.assertEqual(len(group), 8)
            self.assertTrue(all(row.n_read_calls == 1 and row.reward == 0.99 for row in group))
            self.assertTrue(all("Paris" in row.evidence[0]["raw_text"] for row in group))
            self.assertTrue(all("Berlin" not in row.evidence[0]["raw_text"] for row in group))
            store.close()

    def test_directory_versions_remain_separate_after_reopen(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            memory_id = store.add_memory(history(["fact"]))
            builder = DirectoryBuilder(store, Writer(), WRITER_CONFIG, Tokenizer())
            entries = builder.build(memory_id)
            modified = deepcopy(entries)
            modified[0]["content"] = "A different summary."
            store.put_directory(memory_id, "v2", modified)
            store.close()
            store = Store(temporary)
            self.assertEqual(store.directory(memory_id, builder.version), entries)
            self.assertEqual(store.directory(memory_id, "v2"), modified)
            store.close()


if __name__ == "__main__":
    unittest.main()
