from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agmem.data import ADAPTER_VERSION, clean_history
from agmem.pipeline import open_prepared
from agmem.store import Store


class MemoryCacheTests(unittest.TestCase):
    def test_four_interleaved_histories_are_loaded_once_then_lru_evicted(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary, memory_cache_size=4)
            try:
                memory_ids = [store.add_memory(clean_history({"haystack_sessions": [
                    [{"role": "user", "content": f"fact {index}"}]]})) for index in range(5)]
                with patch.object(store.histories, "get", wraps=store.histories.get) as read_history:
                    references = [store.memory(memory_id)[0] for memory_id in memory_ids[:4]]
                    for _ in range(8):
                        for index, reference in enumerate(references):
                            self.assertIn(f"fact {index}", store.record(reference)["raw_text"])
                    self.assertEqual(read_history.call_count, 4)
                    store.memory(memory_ids[4])
                    self.assertEqual(len(store._memory_cache), 4)
                    self.assertNotIn(memory_ids[0], store._memory_cache)
                    self.assertEqual(read_history.call_count, 5)
                    store.record(references[1])
                    self.assertEqual(read_history.call_count, 5)
                    store.record(references[0])
                    self.assertEqual(read_history.call_count, 6)
                    self.assertEqual(len(store._memory_cache), 4)
                    self.assertNotIn(memory_ids[2], store._memory_cache)
            finally:
                store.close()

    def test_capacity_requires_a_positive_integer_and_default_remains_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in (0, -1, True, 1.5, "4", None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    Store(temporary, memory_cache_size=value)
            store = Store(temporary)
            self.assertEqual(store.memory_cache_size, 2)
            store.close()

    def test_open_prepared_matches_each_source_cache_to_candidate_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for source in ("first", "second"):
                store = Store(root / source)
                store.set_metadata("prepare_complete", True)
                store.set_metadata("prepare_identity", {"adapter_version": ADAPTER_VERSION,
                    "storage_format": Store.FORMAT, "sources": [source], "splits": ["train"]})
                store.close()
            config = {"data": {"layout": "by_source", "prepared_root": str(root),
                               "sources": ["first", "second"], "splits": ["train"]},
                      "execution": {"sampling_group_concurrency": 4}}
            store = open_prepared(config)
            try:
                self.assertEqual([child.memory_cache_size for child in store.stores.values()], [4, 4])
                self.assertEqual(store.runtime.memory_cache_size, 2)
            finally:
                store.close()
            config["execution"]["sampling_group_concurrency"] = 1
            store = open_prepared(config)
            try:
                self.assertEqual([child.memory_cache_size for child in store.stores.values()], [2, 2])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
