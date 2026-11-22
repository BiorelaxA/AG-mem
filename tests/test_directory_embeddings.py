"""Embedding request integrity, resumability, and concurrency without GPUs."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
import time
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem.core import InfrastructureError
from agmem.core import digest
from agmem.hierarchical.embedding import EmbeddingClient, embedding_settings


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def response(vectors, indices=None):
    indices = list(range(len(vectors))) if indices is None else indices
    return httpx.Response(200, json={"data": [{"index": i, "embedding": v}
                                               for i, v in zip(indices, vectors)]})


class DirectoryEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "embeddings.sqlite3"

    def client(self, transport, identity=None, cache=None, **settings):
        values = {"instruction": "Group topics.", "expected_dimensions": 2,
                  "max_input_tokens": 128, "max_batch_tokens": 256,
                  "network_retries": 0, **settings}
        client = EmbeddingClient("http://localhost:8000/v1", CharacterTokenizer(),
                                 cache_path=cache or self.cache,
                                 identity=identity or {"model_revision": "test-a", "tokenizer": "chars"},
                                 settings=values, transport=transport)
        self.addCleanup(client.close)
        return client

    def test_response_indices_restore_input_order_and_vectors_are_normalized(self):
        transport = Mock()
        transport.post.return_value = response([[0, 9], [3, 0]], [1, 0])
        client = self.client(transport)
        self.assertEqual(client.encode(["first", "second"]), [[1.0, 0.0], [0.0, 1.0]])
        request = transport.post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "agmem-embedding")
        self.assertEqual(request["encoding_format"], "float")
        self.assertNotIn("truncate_prompt_tokens", request)
        self.assertTrue(request["input"][0].endswith("first"))

    def test_bad_indices_dimensions_nan_and_empty_vectors_are_not_cached(self):
        bad = [response([[1, 0], [0, 1]], [0, 0]), response([[1, 0]], [4]),
               response([[1, 0, 0]]), response([[0, 0]]),
               httpx.Response(200, content=b'{"data":[{"index":0,"embedding":[NaN,1]}]}')]
        for index, result in enumerate(bad):
            with self.subTest(index=index):
                transport = Mock()
                transport.post.return_value = result
                client = self.client(transport, cache=Path(self.temp.name) / f"{index}.db")
                texts = ["a", "b"] if index == 0 else ["a"]
                with self.assertRaises(InfrastructureError):
                    client.encode(texts)
                self.assertFalse(client.pending)
                self.assertEqual(client.db.execute("select count(*) from vectors").fetchone()[0], 0)

    def test_long_inputs_cover_every_character_and_respect_both_batch_limits(self):
        seen = []
        def post(url, **kwargs):
            batch = kwargs["json"]["input"]
            seen.extend(batch)
            self.assertLessEqual(len(batch), 2)
            self.assertLessEqual(sum(len(s) + 2 for s in batch), 128)
            self.assertTrue(all(len(s) + 2 <= 64 for s in batch))
            return response([[3.0, 4.0] for _ in batch])
        transport = Mock()
        transport.post.side_effect = post
        client = self.client(transport, max_input_tokens=64, max_batch_tokens=128,
                             token_margin=2, batch_size=2)
        original = ("证据 🍎\nline with punctuation.  " * 12) + "FINAL-FACT"
        self.assertAlmostEqual(client.encode([original])[0][0], .6)
        pieces = [text.split("\nQuery: ", 1)[1] for text in seen]
        self.assertEqual("".join(pieces), original)
        self.assertGreater(len(pieces), 2)
        self.assertGreater(transport.post.call_count, 1)

    def test_long_input_pooling_is_one_message_vector(self):
        transport = Mock()
        transport.post.side_effect = lambda url, **kw: response(
            [[1, 0] if "AA" in s else [0, 1] for s in kw["json"]["input"]])
        client = self.client(transport, max_input_tokens=64, max_batch_tokens=128, token_margin=2)
        values = client.encode(["A" * 31 + "B" * 31])
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(math.hypot(*values[0]), 1.0)
        self.assertGreater(client.stats["chunks"], 1)

    def test_cache_survives_reopen_but_model_or_instruction_change_invalidates(self):
        transport = Mock()
        transport.post.return_value = response([[1, 0]])
        first = self.client(transport)
        first.encode(["source"])
        first.close()
        second = self.client(transport)
        self.assertEqual(second.encode(["source"]), [[1, 0]])
        self.assertEqual(transport.post.call_count, 1)
        second.close()
        changed = self.client(transport, identity={"model_revision": "test-b", "tokenizer": "chars"})
        changed.encode(["source"])
        changed.close()
        changed_prompt = self.client(transport, instruction="A different clustering instruction")
        changed_prompt.encode(["source"])
        self.assertEqual(transport.post.call_count, 3)

    def test_concurrent_same_key_coalesces_and_repeated_inputs_preserve_order(self):
        entered, release = Event(), Event()
        transport = Mock()
        def post(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return response([[1, 0] for _ in kwargs["json"]["input"]])
        transport.post.side_effect = post
        client = self.client(transport)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.encode, ["same", "same"])
            self.assertTrue(entered.wait(5))
            second = pool.submit(client.encode, ["same"])
            # The second caller must be able to claim/wait without acquiring a
            # network slot or holding the cache lock across the HTTP request.
            deadline = time.monotonic() + 5
            while client.stats["coalesced"] == 0 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(client.stats["coalesced"], 1)
            release.set()
            self.assertEqual(first.result(5), [[1, 0], [1, 0]])
            self.assertEqual(second.result(5), [[1, 0]])
        self.assertEqual(transport.post.call_count, 1)

    def test_network_concurrency_is_bounded_and_retries_release_slots(self):
        lock = Lock()
        active, maximum = 0, 0
        transport = Mock()
        def post(*args, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.01)
            with lock:
                active -= 1
            return response([[1, 0] for _ in kwargs["json"]["input"]])
        transport.post.side_effect = post
        client = self.client(transport, workers=2)
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(len(list(pool.map(lambda i: client.encode([f"unique-{i}"]), range(12)))), 12)
        self.assertLessEqual(maximum, 2)
        self.assertGreater(maximum, 1)

        retry_transport = Mock()
        retry_transport.post.side_effect = [httpx.Response(503), response([[1, 0]])]
        retry_client = self.client(retry_transport, cache=Path(self.temp.name) / "retry.db",
                                   workers=1, network_retries=1)
        def backoff(_):
            self.assertTrue(retry_client.slots.acquire(blocking=False))
            retry_client.slots.release()
        with patch("agmem.hierarchical.embedding.time.sleep", side_effect=backoff):
            self.assertEqual(retry_client.encode(["retry"]), [[1, 0]])

    def test_configuration_rejects_unsafe_urls_and_bad_budgets(self):
        for changes in ({"workers": 0}, {"network_retries": True}, {"expected_dimensions": 0},
                        {"max_input_tokens": 128, "max_batch_tokens": 64},
                        {"timeout_seconds": float("inf")}, {"context": 5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                embedding_settings(changes)
        for url in ("ftp://localhost", "http://user:secret@localhost/v1", "http://localhost/v1?key=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                EmbeddingClient(url, CharacterTokenizer(), cache_path=self.cache,
                                identity={"revision": "test"}, transport=Mock())

    def test_corrupt_cached_row_does_not_strand_other_requested_keys(self):
        transport = Mock()
        transport.post.return_value = response([[1, 0]])
        client = self.client(transport)
        key = digest({"namespace": client.namespace, "text": "corrupt"})
        with client.db:
            client.db.execute("INSERT INTO vectors VALUES (?,?)", (key, '[0,0]'))
        with self.assertRaisesRegex(ValueError, "cached embedding"):
            client.encode(["not-yet-cached", "corrupt"])
        self.assertFalse(client.pending)
        self.assertEqual(client.encode(["not-yet-cached"]), [[1, 0]])


if __name__ == "__main__":
    unittest.main()
