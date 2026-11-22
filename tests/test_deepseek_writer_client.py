"""DeepSeek directory transport tests: no network, paid generation, or GPU."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import format_datetime
import json
import os
from threading import Event, Lock
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem.core import InfrastructureError
from agmem.hierarchical.deepseek_client import DeepSeekWriterPool, schema_messages


class TraceStore:
    def __init__(self):
        self.rows = []

    def trace(self, row):
        self.rows.append(row)


class Clock:
    def __init__(self):
        self.now, self.sleeps = 100.0, []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def response(status=200, content='{"title":"Source"}', finish="stop", headers=None):
    if status != 200:
        return httpx.Response(status, headers=headers, text="secret-server-diagnostics private-api-key")
    return httpx.Response(200, json={
        "model": "deepseek-flash", "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "prompt_cache_hit_tokens": 10},
    })


class DeepSeekWriterClientTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-api-key"}))

    def make_pool(self, outcomes=None, *, handler=None, workers=2, endpoints=None, **settings):
        requests, clock, store = [], Clock(), TraceStore()
        pending = iter(outcomes or [])

        def respond(request):
            requests.append(request)
            item = handler(request) if handler else next(pending)
            if isinstance(item, Exception):
                raise item
            return item

        pool = DeepSeekWriterPool(endpoints or ["https://api.deepseek.com"], workers, settings, store,
                                  transport=httpx.MockTransport(respond), sleep=clock.sleep,
                                  monotonic=clock.monotonic, wall_time=lambda: 1_700_000_000)
        self.addCleanup(pool.close)
        client = pool.for_session({"bank_id": "bank", "parent_id": "parent"})
        return pool, client, requests, store, clock

    def call(self, client, messages=None, schema=None):
        return client(messages or [{"role": "user", "content": "Cluster these source blocks by topic."}],
                      64, schema or {"type": "object"}, "segment")

    def errors(self, store):
        return [row for row in store.rows if row["event"] == "writer_transport_error"]

    def test_payload_preserves_semantic_prompt_and_full_schema_without_vllm_options(self):
        _, client, requests, store, _ = self.make_pool([response()], seed=77, structured_outputs=True)
        messages = [{"role": "system", "content": "Existing semantic clustering instructions."},
                    {"role": "user", "content": "Source history."}]
        schema = {"type": "object", "properties": {"entities": {"type": "array", "uniqueItems": True}}}
        original = deepcopy((messages, schema))
        result = self.call(client, messages, schema)
        request, = requests
        payload = json.loads(request.content)
        self.assertEqual(set(payload), {"model", "messages", "max_tokens", "temperature", "thinking",
                                        "response_format", "stream"})
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["stream"])
        self.assertTrue(payload["messages"][0]["content"].startswith(messages[0]["content"] + "\n\n"))
        self.assertEqual(json.loads(payload["messages"][0]["content"].split("JSON Schema:\n")[1]), schema)
        self.assertEqual(payload["messages"][1:], messages[1:])
        self.assertEqual((messages, schema), original)
        self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer private-api-key")
        self.assertEqual(result["model"], "deepseek-flash")
        self.assertEqual(result["usage"]["prompt_cache_hit_tokens"], 10)
        self.assertNotIn("private-api-key", json.dumps(store.rows))
        self.assertNotIn("Authorization", json.dumps(store.rows))

    def test_schema_instruction_precedes_user_when_no_system_message(self):
        messages = [{"role": "user", "content": "unchanged"}]
        result = schema_messages(messages, {"type": "object"})
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("JSON", result[0]["content"])
        self.assertEqual(result[1:], messages)

    def test_429_honors_retry_after_and_preserves_response_metadata(self):
        pool, client, requests, store, clock = self.make_pool(
            [response(429, headers={"Retry-After": "7"}), response()], network_retries=1)
        result = self.call(client)
        self.assertEqual(len(requests), 2)
        self.assertEqual(clock.sleeps, [7])
        self.assertEqual(pool.slots.qsize(), 2)
        error, = self.errors(store)
        self.assertEqual((error["status_code"], error["retryable"], error["will_retry"]), (429, True, True))
        self.assertEqual(error["retry_delay_seconds"], 7)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(store.rows[-1]["response"], result)
        self.assertNotIn("secret-server-diagnostics", json.dumps(store.rows))

    def test_retry_after_supports_http_date_and_ignores_invalid_values(self):
        future = format_datetime(datetime.fromtimestamp(1_700_000_009, timezone.utc), usegmt=True)
        for value, expected in ((future, 9), ("junk", 1), ("nan", 1), ("inf", 1), ("-2", 1)):
            with self.subTest(value=value):
                _, client, _, _, clock = self.make_pool(
                    [response(503, headers={"Retry-After": value}), response()], network_retries=1)
                self.call(client)
                self.assertEqual(clock.sleeps, [expected])

    def test_network_and_5xx_use_bounded_exponential_backoff(self):
        pool, client, requests, store, clock = self.make_pool([
            httpx.ConnectError("private proxy credential"), httpx.ReadTimeout("private timeout detail"),
            response(500), response(503), response(),
        ], network_retries=4, retry_max_seconds=3)
        self.call(client)
        self.assertEqual(len(requests), 5)
        self.assertEqual(clock.sleeps, [1, 2, 3, 3])
        self.assertEqual(pool.slots.qsize(), 2)
        self.assertEqual([row["status_code"] for row in self.errors(store)], [None, None, 500, 503])
        for secret in ("private proxy credential", "private timeout detail", "private-api-key"):
            self.assertNotIn(secret, json.dumps(store.rows))

    def test_non_retryable_http_failures_are_redacted_and_release_slots(self):
        for status in (400, 401, 402, 403, 404, 408, 413, 422):
            with self.subTest(status=status):
                pool, client, requests, store, clock = self.make_pool([response(status)], network_retries=5)
                with self.assertRaisesRegex(InfrastructureError, f"HTTP {status}") as caught:
                    self.call(client)
                self.assertEqual(len(requests), 1)
                self.assertEqual(pool.slots.qsize(), 2)
                self.assertFalse(clock.sleeps)
                self.assertFalse(self.errors(store)[0]["will_retry"])
                self.assertFalse(self.errors(store)[0]["retryable"])
                logged = json.dumps(store.rows) + str(caught.exception)
                self.assertNotIn("private-api-key", logged)
                self.assertNotIn("secret-server-diagnostics", logged)

    def test_length_including_empty_content_returns_to_writer_without_retry(self):
        for content in (None, "", "   ", '{"parent":'):
            with self.subTest(content=content):
                pool, client, requests, store, clock = self.make_pool(
                    [response(content=content, finish="length")], network_retries=3)
                result = self.call(client)
                self.assertEqual(result["finish_reason"], "length")
                self.assertEqual(result["text"], content)
                self.assertEqual(len(requests), 1)
                self.assertEqual(pool.slots.qsize(), 2)
                self.assertFalse(self.errors(store))
                self.assertFalse(clock.sleeps)

    def test_empty_non_length_content_retries_then_raises_without_placeholder(self):
        for content in (None, "", " \n\t "):
            with self.subTest(content=content):
                pool, client, requests, store, _ = self.make_pool(
                    [response(content=content), response(content=content)], network_retries=1)
                with self.assertRaises(InfrastructureError):
                    self.call(client)
                self.assertEqual(len(requests), 2)
                self.assertEqual(pool.slots.qsize(), 2)
                self.assertEqual([row["will_retry"] for row in self.errors(store)], [True, False])
                self.assertFalse(any(row["event"] == "writer_validated" for row in store.rows))

    def test_empty_content_can_recover_on_retry(self):
        _, client, requests, _, _ = self.make_pool([response(content=" "), response()], network_retries=1)
        self.assertEqual(self.call(client)["text"], '{"title":"Source"}')
        self.assertEqual(len(requests), 2)

    def test_malformed_provider_payload_is_a_retryable_infrastructure_failure(self):
        malformed = [None, [], {}, {"choices": []}, {"choices": [None]},
                     {"choices": [{"message": None}]}, {"choices": [{"message": {"content": []}}]}]
        for payload in malformed:
            with self.subTest(payload=payload):
                _, client, requests, store, _ = self.make_pool(
                    [httpx.Response(200, json=payload), response()], network_retries=1)
                self.assertEqual(self.call(client)["finish_reason"], "stop")
                self.assertEqual(len(requests), 2)
                self.assertTrue(self.errors(store)[0]["will_retry"])

    def test_invalid_json_response_is_retried_without_logging_raw_body(self):
        _, client, requests, store, _ = self.make_pool(
            [httpx.Response(200, text="private-api-key secret-server-diagnostics"), response()], network_retries=1)
        self.call(client)
        self.assertEqual(len(requests), 2)
        self.assertNotIn("secret-server-diagnostics", json.dumps(store.rows))

    def test_json_semantics_stay_with_existing_writer_and_trace_accept_reject(self):
        from jsonschema import Draft202012Validator, ValidationError
        from agmem.hierarchical.writer import writer_schema, writer_settings

        schema = writer_schema(writer_settings())
        invalid = {"parent": {"title": "Session", "content": "Source"}, "children": [
            {"end_block": 1, "title": "Topic", "content": "Source", "entities": ["Alice", "Alice"]}]}
        _, client, _, store, _ = self.make_pool([response(content=json.dumps(invalid))])
        result = self.call(client, schema=schema)
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(json.loads(result["text"]))
        args = ([{"role": "user", "content": "history"}], 64, schema, "segment", result)
        client.reject(*args, reason="validation failed private-api-key")
        client.accept(*args)
        self.assertEqual([row["event"] for row in store.rows][-2:],
                         ["writer_validation_failed", "writer_validated"])
        self.assertEqual(store.rows[-1]["bank_id"], "bank")
        self.assertEqual(store.rows[-1]["parent_id"], "parent")
        self.assertIn("response_sha256", store.rows[-1])
        self.assertNotIn("private-api-key", json.dumps(store.rows))

    def test_provider_echoed_key_is_redacted_from_traces_but_not_result(self):
        _, client, _, store, _ = self.make_pool([response(content='{"echo":"private-api-key"}')])
        result = self.call(client)
        self.assertIn("private-api-key", result["text"])
        self.assertNotIn("private-api-key", json.dumps(store.rows))
        self.assertIn("[REDACTED]", json.dumps(store.rows))

    def test_shared_cooldown_staggers_different_sessions(self):
        pool, first, requests, _, clock = self.make_pool(
            [response(429, headers={"Retry-After": "5"}), response(), response()], network_retries=0)
        with self.assertRaises(InfrastructureError):
            self.call(first)
        second = pool.for_session({"bank_id": "bank", "parent_id": "second"})
        self.call(second)
        self.call(first)
        self.assertEqual(len(requests), 3)
        self.assertEqual(clock.sleeps[0], 5)
        self.assertAlmostEqual(clock.sleeps[1], 0.05)

    def test_another_429_extends_an_already_waiting_worker_cooldown(self):
        pool, client, _, _, clock = self.make_pool([response()])
        pool._defer(0, 5)
        original_sleep = clock.sleep

        def sleep_then_defer(duration):
            original_sleep(duration)
            if len(clock.sleeps) == 1:
                pool._defer(0, 7)

        pool._sleep = sleep_then_defer
        self.call(client)
        self.assertEqual(clock.sleeps, [5, 7])

    def test_workers_bound_inflight_requests_across_sessions_and_endpoints(self):
        lock, full, release = Lock(), Event(), Event()
        counters = {"active": 0, "maximum": 0}

        def handler(request):
            with lock:
                counters["active"] += 1
                counters["maximum"] = max(counters["maximum"], counters["active"])
                if counters["active"] == 3:
                    full.set()
            try:
                if not release.wait(5):
                    raise AssertionError("test request release timed out")
                return response()
            finally:
                with lock:
                    counters["active"] -= 1

        pool, _, requests, _, _ = self.make_pool(handler=handler, workers=3,
                                                endpoints=["https://one.invalid/v1/", "https://two.invalid"])
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(self.call, pool.for_session({"bank_id": "b", "parent_id": str(index)}))
                       for index in range(12)]
            try:
                self.assertTrue(full.wait(3))
                self.assertEqual(counters["maximum"], 3)
                self.assertEqual(len(requests), 3)
            finally:
                release.set()
            for future in futures:
                self.assertEqual(future.result(timeout=5)["finish_reason"], "stop")
        self.assertEqual(counters["maximum"], 3)
        self.assertEqual(len(requests), 12)
        self.assertEqual(pool.slots.qsize(), 3)
        self.assertIn("https://one.invalid/v1/chat/completions", {str(request.url) for request in requests})

    def test_client_limits_and_trust_env_are_explicit_and_configurable(self):
        for settings, expected in (({}, True), ({"trust_env": False}, False)):
            with self.subTest(settings=settings), patch("httpx.Client") as factory:
                pool = DeepSeekWriterPool(["https://api.deepseek.com"], 64, settings, TraceStore())
                self.addCleanup(pool.close)
                kwargs = factory.call_args.kwargs
                self.assertEqual(kwargs["trust_env"], expected)
                self.assertEqual(kwargs["limits"].max_connections, 64)
                self.assertEqual(kwargs["limits"].max_keepalive_connections, 64)
                self.assertEqual(kwargs["timeout"], 600)

    def test_configuration_and_endpoint_secrets_are_rejected_before_client_construction(self):
        valid = "https://api.deepseek.com"
        invalid_urls = ("https://user:secret@host/v1", "https://@host", "https://host?key=secret",
                        "https://host#secret", "https://host?", "https://host#", "ftp://host",
                        "https://host:99999", "https://[bad", "https://host:0", "https://ho st", "\nhttps://host")
        cases = [([value], 2, {}) for value in invalid_urls]
        cases += [([], 2, {}), (valid, 2, {}), ([valid], 0, {}), ([valid], True, {})]
        cases += [([valid], 2, {"network_retries": value}) for value in (-1, True, "1", 1.0)]
        cases += [([valid], 2, {key: value}) for key, value in (
            ("trust_env", "false"), ("trust_env", 1), ("model", ""), ("api_key_env", "secret key"),
            ("timeout_seconds", 0), ("retry_base_seconds", -1), ("retry_max_seconds", float("inf")),
            ("retry_spacing_seconds", 0),
        )]
        for endpoints, workers, settings in cases:
            with self.subTest(endpoints=endpoints, settings=settings), patch("httpx.Client") as factory:
                with self.assertRaises(ValueError) as caught:
                    DeepSeekWriterPool(endpoints, workers, settings, TraceStore())
                self.assertNotIn("user:secret", str(caught.exception))
                self.assertNotIn("key=secret", str(caught.exception))
                factory.assert_not_called()

    def test_missing_or_invalid_key_fails_before_http_client_construction(self):
        for key in ("", "private\nkey", " key "):
            with self.subTest(key=key), patch.dict(os.environ, {"DEEPSEEK_API_KEY": key}), \
                    patch("httpx.Client") as factory:
                with self.assertRaises(InfrastructureError):
                    DeepSeekWriterPool(["https://api.deepseek.com"], 2, {}, TraceStore())
                factory.assert_not_called()

    def test_custom_key_environment_and_model_are_supported(self):
        with patch.dict(os.environ, {"MY_DEEPSEEK_KEY": "custom-private-key"}):
            _, client, requests, store, _ = self.make_pool([response()], api_key_env="MY_DEEPSEEK_KEY",
                                                          model="deepseek-custom")
        self.call(client)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer custom-private-key")
        self.assertEqual(json.loads(requests[0].content)["model"], "deepseek-custom")
        self.assertNotIn("custom-private-key", json.dumps(store.rows))

    def test_partial_client_construction_closes_created_clients_and_redacts_failure(self):
        first = Mock()
        with patch("httpx.Client", side_effect=[first, RuntimeError("private proxy password")]):
            with self.assertRaises(InfrastructureError) as caught:
                DeepSeekWriterPool(["https://one.invalid", "https://two.invalid"], 2, {}, TraceStore())
        first.close.assert_called_once_with()
        self.assertNotIn("private proxy password", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
