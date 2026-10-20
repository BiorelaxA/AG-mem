from pathlib import Path
import json
import tempfile
import unittest

import httpx

from agmem.core import Example, InfrastructureError
from agmem.data import clean_history
from agmem.services import ChatService, DirectoryBuilder, TaskAgent
from agmem.store import Store


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.config = {"model": "fixture", "model_revision": "fixture-v1", "base_url": "http://fixture/v1", "retries": 0}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

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

    def writer_fixture(self, responses, format_retries=2):
        requests = []
        responses = iter(responses)
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": next(responses)}}]})
        service = ChatService(self.config, self.store, "writer")
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(service.client.close)
        config = {"lookback_sessions": 5, "content_max_tokens": 200, "max_output_tokens": 384,
                  "format_retries": format_retries}
        memory_id = self.store.add_memory(clean_history({
            "haystack_sessions": [[{"role": "user", "content": "Alice moved to Paris in 2023."}]],
            "haystack_session_ids": ["session0"],
        }))
        return DirectoryBuilder(self.store, service, config, WordTokenizer()), memory_id, requests

    def test_writer_repairs_overlength_summary_and_reuses_valid_cache(self):
        oversized = json.dumps({"content": " ".join(["detail"] * 201)})
        valid_content = " ".join(["fact"] * 200)
        builder, memory_id, requests = self.writer_fixture([oversized, json.dumps({"content": valid_content})])
        entries = builder.build(memory_id)
        self.assertEqual(entries[0]["content"], valid_content)
        self.assertEqual(len(requests), 2)
        self.assertIn("at most 200 tokens", requests[0]["messages"][0]["content"])
        self.assertEqual(requests[0]["max_tokens"], 384)
        self.assertEqual(requests[1]["messages"][-2], {"role": "assistant", "content": oversized})
        self.assertIn("201 tokens", requests[1]["messages"][-1]["content"])
        self.assertIn("content_max_tokens=200", requests[1]["messages"][-1]["content"])
        self.assertIn("Shorten the summary", requests[1]["messages"][-1]["content"])
        self.assertEqual(self.store.directory_entry(memory_id, builder.version, entries[0]["id"])["content"], valid_content)
        self.assertEqual(builder.build(memory_id), entries)
        self.assertEqual(len(requests), 2)

    def test_writer_reports_exhausted_length_budget_without_publishing_directory(self):
        oversized = json.dumps({"content": " ".join(["detail"] * 201)})
        builder, memory_id, requests = self.writer_fixture([oversized] * 3)
        with self.assertRaisesRegex(ValueError, "after 3 attempts: Summary contains 201 tokens; content_max_tokens=200"):
            builder.build(memory_id)
        self.assertEqual(len(requests), 3)
        for request in requests[1:]:
            self.assertIn("content limit is 200 tokens", request["messages"][-1]["content"])
            self.assertNotIn("aim for", request["messages"][-1]["content"])
        self.assertEqual(len(self.store.directory_entries.keys()), 0)
        with self.assertRaisesRegex(ValueError, "Directory missing"):
            self.store.directory(memory_id, builder.version)

    def test_writer_repairs_invalid_json_schema(self):
        invalid = ["not JSON", "[]", '{"content": "fact", "extra": true}', '{"content": 3}',
                   '{"content": "  "}', '```json\n{"content": "fact"}\n```']
        builder, memory_id, requests = self.writer_fixture(
            invalid + ['{"content": "Alice moved to Paris in 2023."}'], format_retries=len(invalid))
        self.assertEqual(builder.build(memory_id)[0]["content"], "Alice moved to Paris in 2023.")
        self.assertEqual(len(requests), len(invalid) + 1)
        for index, response in enumerate(invalid):
            self.assertEqual(requests[index + 1]["messages"][-2]["content"], response)
            feedback = requests[index + 1]["messages"][-1]["content"]
            self.assertIn("Repair the JSON format", feedback)
            self.assertNotIn("Shorten the summary", feedback)

    def test_writer_response_cache_is_scoped_to_complete_history(self):
        responses = ['{"content": "Alice moved to Paris."}'] * 3
        builder, first_id, requests = self.writer_fixture(responses)
        first = builder.build(first_id)
        records = clean_history({
            "haystack_sessions": [[{"role": "user", "content": "Alice moved to Paris in 2023."}],
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
