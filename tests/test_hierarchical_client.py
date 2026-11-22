"""Local Writer transport retries and credentials, without sockets or GPUs."""

import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem.core import InfrastructureError
from agmem.hierarchical.client import WriterPool, guidance_schema


class TraceStore:
    def __init__(self):
        self.rows = []

    def trace(self, row):
        self.rows.append(row)


def response(status=200, finish_reason="stop"):
    if status != 200:
        return httpx.Response(status, text="private-server-diagnostics")
    return httpx.Response(status, json={
        "choices": [{"message": {"content": '{"title":"source"}'}, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 12},
    })


class HierarchicalClientTests(unittest.TestCase):
    def make_pool(self, outcomes, **settings):
        transport = Mock()
        transport.post.side_effect = outcomes
        factory = patch("httpx.Client", return_value=transport)
        mocked_factory = factory.start()
        self.addCleanup(factory.stop)
        store = TraceStore()
        pool = WriterPool(["http://localhost:8000/v1"], 2, settings, store)
        self.addCleanup(pool.close)
        client = pool.for_session({"bank_id": "bank", "parent_id": "p1"})
        return pool, client, transport, store, mocked_factory

    def call(self, client):
        return client([{"role": "user", "content": "public history"}], 64, {"type": "object"}, "segment")

    def errors(self, store):
        return [row for row in store.rows if row["event"] == "writer_transport_error"]

    def test_429_retries_after_returning_slot_and_keeps_secrets_out_of_traces(self):
        pool, client, transport, store, factory = self.make_pool([response(429), response()], network_retries=1)
        with patch.dict(os.environ, {"AGMEM_API_KEY": "private-api-key", "HTTPS_PROXY": "http://proxy.invalid"}), \
                patch("agmem.hierarchical.client.time.sleep", side_effect=lambda _: self.assertEqual(pool.slots.qsize(), 2)) as sleep:
            result = self.call(client)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(transport.post.call_count, 2)
        self.assertEqual(pool.slots.qsize(), 2)
        sleep.assert_called_once_with(1)
        factory.assert_called_once_with(timeout=600, trust_env=False)
        for call in transport.post.call_args_list:
            self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer private-api-key")
        error, = self.errors(store)
        self.assertEqual((error["status_code"], error["retryable"], error["will_retry"]), (429, True, True))
        logged = json.dumps(store.rows)
        for private in ("private-api-key", "private-server-diagnostics", "Authorization", "proxy.invalid"):
            self.assertNotIn(private, logged)

    def test_401_fails_without_retry_and_returns_slot(self):
        pool, client, transport, store, _ = self.make_pool([response(401)], network_retries=3)
        with patch("agmem.hierarchical.client.time.sleep") as sleep, self.assertRaises(InfrastructureError):
            self.call(client)
        self.assertEqual(transport.post.call_count, 1)
        self.assertEqual(pool.slots.qsize(), 2)
        sleep.assert_not_called()
        error, = self.errors(store)
        self.assertEqual((error["status_code"], error["retryable"], error["will_retry"]), (401, False, False))
        self.assertNotIn("private-server-diagnostics", json.dumps(store.rows))

    def test_transport_error_exhaustion_records_null_status_and_remains_reusable(self):
        pool, client, transport, store, _ = self.make_pool([
            httpx.ConnectError("private-connection-detail"), httpx.ReadTimeout("private-timeout-detail"), response()
        ], network_retries=1)
        with patch("agmem.hierarchical.client.time.sleep"), self.assertRaises(InfrastructureError):
            self.call(client)
        self.assertEqual(pool.slots.qsize(), 2)
        self.assertEqual([(r["status_code"], r["retryable"], r["will_retry"]) for r in self.errors(store)],
                         [(None, True, True), (None, True, False)])
        self.assertEqual(self.call(client)["finish_reason"], "stop")
        self.assertEqual(pool.slots.qsize(), 2)
        self.assertNotIn("private-connection-detail", json.dumps(store.rows))
        self.assertNotIn("private-timeout-detail", json.dumps(store.rows))

    def test_length_is_returned_for_writer_validation_without_transport_retry(self):
        pool, client, transport, store, _ = self.make_pool([response(finish_reason="length")], network_retries=3)
        result = self.call(client)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(transport.post.call_count, 1)
        self.assertEqual(pool.slots.qsize(), 2)
        self.assertFalse(self.errors(store))
        self.assertFalse(any(row["event"] == "writer_validated" for row in store.rows))

    def test_decoder_projection_does_not_weaken_local_uniqueness_validation(self):
        from jsonschema import Draft202012Validator, ValidationError
        from agmem.hierarchical.writer import writer_schema, writer_settings
        schema = writer_schema(writer_settings())
        _, client, transport, _, _ = self.make_pool([response()])
        client([{"role": "user", "content": "history"}], 64, schema, "segment")
        sent = transport.post.call_args.kwargs["json"]["structured_outputs"]["json"]
        original_entities = schema["properties"]["children"]["items"]["properties"]["entities"]
        self.assertTrue(original_entities["uniqueItems"])
        self.assertNotIn("uniqueItems", sent["properties"]["children"]["items"]["properties"]["entities"])
        invalid = {"parent": {"title": "Session", "content": "Source"}, "children": [
            {"end_block": 1, "title": "Topic", "content": "Source", "entities": ["Alice", "Alice"]}]}
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(invalid)

    def test_installed_guidance_compiles_actual_writer_and_fixed_schemas(self):
        try:
            import llguidance
        except ImportError:
            self.skipTest("llguidance is supplied by the GPU serving environment")
        from agmem.hierarchical.writer import writer_schema, writer_settings
        for fixed in (None, [8, 16, 17]):
            with self.subTest(fixed=fixed):
                grammar = llguidance.LLMatcher.grammar_from_json_schema(
                    guidance_schema(writer_schema(writer_settings(), fixed)), defaults={"whitespace_flexible": True})
                self.assertFalse(llguidance.LLMatcher.validate_grammar(grammar))

    def test_invalid_configuration_is_rejected_before_creating_clients(self):
        valid = "http://localhost:8000/v1"
        cases = [([valid], {"network_retries": retries}) for retries in (-1, 1.0, True, "1")]
        cases += [([valid, bad], {}) for bad in ("http://user:secret@localhost/v1", "http://@localhost/v1",
                  "http://localhost/v1?key=secret", "http://localhost/v1#secret", "ftp://localhost/v1",
                  "http://localhost:99999/v1", "http://[bad/v1")]
        cases += [([], {}), (valid, {})]
        for endpoints, settings in cases:
            with self.subTest(endpoints=endpoints, settings=settings), patch("httpx.Client") as factory:
                with self.assertRaises(ValueError):
                    WriterPool(endpoints, 2, settings, TraceStore())
                factory.assert_not_called()

    def test_partial_client_construction_closes_previously_created_client(self):
        first = Mock()
        with patch("httpx.Client", side_effect=[first, RuntimeError("constructor failed")]):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                WriterPool(["http://localhost:8000/v1", "http://localhost:8001/v1"], 2, {}, TraceStore())
        first.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
