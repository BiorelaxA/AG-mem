from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpcore
import httpx

from agmem.core import Example, InfrastructureError
from agmem.data import clean_history
from agmem.services import ChatService, DirectoryBuilder, TaskAgent
from agmem.store import Store


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, ids, **kwargs):
        return " ".join(ids)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.config = {"model": "fixture", "model_revision": "fixture-v1", "base_url": "http://fixture/v1", "retries": 0}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_loopback_service_bypasses_inherited_proxy_using_actual_endpoint(self):
        proxy_env = {name: "http://broken-proxy.invalid:59999" for name in
                     ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")}
        proxy_env.update(NO_PROXY="", no_proxy="")
        with patch.dict(os.environ, proxy_env), \
             patch.object(httpcore.HTTPProxy, "handle_request", side_effect=RuntimeError("proxy selected")) as proxy, \
             patch.object(httpcore.ConnectionPool, "handle_request",
                          side_effect=lambda request: httpcore.Response(200, content=b"{}")) as direct:
            for endpoint in ("http://127.0.0.1:50559/v1", "http://127.0.0.2/v1",
                             "http://localhost:50559/v1", "http://[::1]:50559/v1"):
                with self.subTest(endpoint=endpoint):
                    config = {**self.config, "request_base_url": endpoint}
                    service = ChatService(config, self.store, "task_agent")
                    try:
                        self.assertEqual(service.client.get(endpoint + "/models").status_code, 200)
                    finally:
                        service.client.close()
            self.assertEqual(direct.call_count, 4)
            proxy.assert_not_called()
            # Routing uses the actual deployment URL, not the cache identity URL.
            config = {**self.config, "base_url": "http://127.0.0.1/v1",
                      "request_base_url": "http://remote-service.invalid/v1"}
            service = ChatService(config, self.store, "task_agent")
            try:
                with self.assertRaisesRegex(RuntimeError, "proxy selected"):
                    service.client.get(config["request_base_url"] + "/models")
            finally:
                service.client.close()
            proxy.assert_called_once()

    def test_task_payload_and_deterministic_cache(self):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "A"}}]})
        service = ChatService(self.config, self.store, "task_agent")
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(respond))
        agent = TaskAgent(service)
        example = Example("fixture", "fixture", "train", "m", "Pick one", options={"A": "one", "B": "two"}, gold_answers=["A"], answer_mode="choice")
        result, key = agent.answer(example, [])
        self.assertEqual(result, "A")
        self.assertEqual(agent.answer(example, []), (result, key))
        self.assertEqual(len(requests), 1)
        self.assertFalse(requests[0]["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("gold_answers", json.dumps(requests))
        self.assertNotIn("correct_answer", json.dumps(requests))

    def test_http_failure_and_truncation_are_infrastructure_failures(self):
        service = ChatService(self.config, self.store, "task_agent")
        for response in [httpx.Response(503), httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "A"}}]})]:
            service.client.close()
            service.client = httpx.Client(transport=httpx.MockTransport(lambda request: response))
            with self.assertRaises(InfrastructureError):
                service.complete([{"role": "user", "content": "q"}], 10)
        self.assertEqual(len(self.store.completions.keys()), 0)

    def writer_fixture(self, responses):
        requests = []
        responses = iter(responses)
        def respond(request):
            requests.append(json.loads(request.content))
            response = next(responses)
            payload = response if isinstance(response, dict) else {
                "choices": [{"finish_reason": "stop", "message": {"content": response}}]}
            return httpx.Response(200, json=payload)
        service = ChatService(self.config, self.store, "writer")
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(service.client.close)
        config = {"lookback_sessions": 3, "content_max_tokens": 300}
        memory_id = self.store.add_memory(clean_history({
            "haystack_sessions": [[{"role": "user", "name": "Alice", "content": "Alice moved to Paris in 2023."}]],
            "haystack_session_ids": ["session0"],
        }))
        return DirectoryBuilder(self.store, service, config, WordTokenizer()), memory_id, requests

    def test_writer_plain_content_and_program_generated_metadata(self):
        summary = 'Alice said "hello"; moved to Paris in 2023.\nPrefers walking.'
        builder, memory_id, requests = self.writer_fixture([summary])
        entries = builder.build(memory_id)
        self.assertEqual(entries[0]["content"], summary)
        self.assertEqual(entries[0]["task_session"]["session_id"], "session0")
        self.assertEqual(entries[0]["time_order"]["order"], 0)
        self.assertEqual(entries[0]["actors_source"]["actors"], ["Alice"])
        self.assertEqual(requests[0]["max_tokens"], 300)
        prompt = requests[0]["messages"][0]["content"]
        self.assertIn("nonempty plain text", prompt)
        self.assertIn("at most 300 tokens", prompt)
        self.assertIn("Every summarized fact must be supported by TARGET_SESSION", prompt)
        saved = json.loads((self.store.root / "directories.jsonl").read_text())
        self.assertEqual(saved["content"], summary)
        self.assertEqual(saved["memory_id"], memory_id)
        self.assertEqual(builder.build(memory_id), entries)
        self.assertEqual(len(requests), 1)

    def test_writer_accepts_length_without_retry_and_caches_response(self):
        summary = " ".join(f"fact{i}" for i in range(300))
        response = {"choices": [{"finish_reason": "length", "message": {"content": summary}}],
                    "usage": {"prompt_tokens": 28060, "completion_tokens": 300}}
        builder, memory_id, requests = self.writer_fixture([response])
        builder.service.config = {**builder.service.config, "retries": 2}
        self.assertEqual(builder.build(memory_id)[0]["content"], summary)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(self.store.completions.keys()), 1)
        self.assertFalse((self.store.root / "service_failures.jsonl").exists())
        # Completion cache reuse also accepts the formerly truncated response.
        self.assertEqual(builder.service.complete(requests[0]["messages"], 300,
            cache_scope=f"{memory_id}/{builder.version}"), summary)
        self.assertEqual(len(requests), 1)

    def test_writer_overlong_response_is_cut_at_token_boundary_without_retry(self):
        summary = " ".join(f"fact{i}" for i in range(320))
        builder, memory_id, requests = self.writer_fixture([summary])
        entries = builder.build(memory_id)
        self.assertEqual(entries[0]["content"], " ".join(f"fact{i}" for i in range(300)))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 300)
        self.assertEqual(builder.build(memory_id), entries)
        self.assertEqual(len(requests), 1)

    def test_writer_empty_response_is_saved_pending_once_and_is_logged(self):
        for index, empty in enumerate((None, "", "   ")):
            response = {"choices": [{"finish_reason": "length", "message": {
                "content": empty, "reasoning_content": "Reasoning only"}}]}
            builder, memory_id, requests = self.writer_fixture([response])
            builder.service.config = {**builder.service.config, "retries": 2}
            entries = builder.build(memory_id)
            self.assertEqual(len(requests), 1)
            self.assertEqual(entries[0]["content"], "")
            self.assertEqual(entries[0]["writer_status"], "empty_pending")
            self.assertEqual(self.store.directory(memory_id, builder.version), entries)
            self.assertEqual(len(self.store.directory_entries.keys()), 1)
            self.assertEqual(len(self.store.completions.keys()), 0)
            failures = (self.store.root / "service_failures.jsonl").read_text().splitlines()
            self.assertEqual(len(failures), index + 1)
            self.assertEqual(json.loads(failures[-1])["response"], response)

    def test_writer_continues_after_empty_and_only_retries_pending_after_reopen(self):
        empty = {"choices": [{"finish_reason": "stop", "message": {"content": None}}]}
        builder, _, requests = self.writer_fixture([empty, "Second summary.", "Third summary."])
        memory_id = self.store.add_memory(clean_history({
            "haystack_sessions": [[{"role": "user", "content": text}] for text in ("first", "second", "third")],
            "haystack_session_ids": ["session0", "session1", "session2"],
        }))
        entries = builder.build(memory_id)
        self.assertEqual(len(requests), 3)
        self.assertEqual([entry["content"] for entry in entries], ["", "Second summary.", "Third summary."])
        self.assertEqual(entries[0]["writer_status"], "empty_pending")
        self.store.close()
        self.store = Store(self.store.root)
        self.assertEqual(self.store.directory(memory_id, builder.version)[0]["writer_status"], "empty_pending")
        retried = []
        def respond(request):
            retried.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "First summary."}}]})
        service = ChatService(self.config, self.store, "writer")
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(service.client.close)
        resumed = DirectoryBuilder(self.store, service, builder.config, WordTokenizer())
        completed = resumed.build(memory_id)
        self.assertEqual(resumed.version, builder.version)
        self.assertEqual(len(retried), 1)
        self.assertEqual(json.loads(retried[0]["messages"][1]["content"])["TARGET_SESSION"]["order"], 0)
        self.assertEqual([entry["content"] for entry in completed], ["First summary.", "Second summary.", "Third summary."])
        self.assertTrue(all("writer_status" not in entry for entry in completed))
        self.assertEqual(self.store.directory(memory_id, builder.version), completed)
        self.assertEqual(resumed.build(memory_id), completed)
        self.assertEqual(len(retried), 1)
        # Filling a placeholder also invalidates any in-memory directory cache.
        self.store.close()
        self.store = Store(self.store.root)
        self.assertEqual(self.store.directory(memory_id, builder.version), completed)

    def test_only_pending_content_can_be_replaced_and_source_metadata_stays_fixed(self):
        empty = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
        builder, memory_id, _ = self.writer_fixture([empty])
        pending = builder.build(memory_id)[0]
        altered = {**pending, "actors_source": {"actors": ["Somebody else"], "sources": []}}
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.put_directory_entry(memory_id, builder.version, altered)
        completed = {key: value for key, value in pending.items() if key != "writer_status"}
        completed["content"] = "Recovered summary."
        self.store.put_directory(memory_id, builder.version, [completed])
        self.assertEqual(self.store.directory(memory_id, builder.version), [completed])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.put_directory_entry(memory_id, builder.version, {**completed, "content": "Changed summary."})

    def test_writer_does_not_parse_json_or_retry_for_text_format(self):
        # The service body is content verbatim. JSON serialization belongs to Store.
        text = '{"text":"quotes and an unfinished phrase'
        builder, memory_id, requests = self.writer_fixture([text])
        self.assertEqual(builder.build(memory_id)[0]["content"], text)
        self.assertEqual(len(requests), 1)

    def test_writer_response_cache_is_scoped_to_complete_history(self):
        responses = ['Alice moved to Paris.'] * 3
        builder, first_id, requests = self.writer_fixture(responses)
        first = builder.build(first_id)
        records = clean_history({
            "haystack_sessions": [[{"role": "user", "name": "Alice", "content": "Alice moved to Paris in 2023."}],
                                  [{"role": "user", "content": "Alice moved to Berlin later."}]],
            "haystack_session_ids": ["session0", "session1"],
        })
        second_id = self.store.add_memory(records)
        self.assertNotEqual(first_id, second_id)
        builder.build(second_id)
        # Identical Writer input for the shared first session still requires a fresh call.
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertEqual(len(requests), 3)
        self.assertEqual(builder.build(first_id), first)
        self.assertEqual(self.store.add_memory(records), second_id)
        builder.build(second_id)
        self.assertEqual(len(requests), 3)
        rows = [json.loads(line) for line in (self.store.root / "directories.jsonl").read_text().splitlines()]
        self.assertEqual([row["memory_id"] for row in rows], [first_id, second_id, second_id])
        self.assertEqual([row["session_id"] for row in rows], ["session0", "session0", "session1"])
        self.assertEqual([row["order"] for row in rows], [0, 0, 1])
        self.assertFalse(list(self.store.root.glob("*.sqlite")))


if __name__ == "__main__":
    unittest.main()
