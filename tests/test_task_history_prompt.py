"""Versioned history-answer prompts shared by local and remote Task agents."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agmem.core import Example, InfrastructureError, canonical, digest
from agmem.pipeline import make_task_agent
from agmem.prompts import HISTORY_ANSWER_TEMPLATE, TASK_SYSTEM, task_input_contract, task_messages
from agmem.services import ChatService, TaskAgent
from agmem.store import Store


class HistoryTaskPromptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name) / "store")
        self.addCleanup(self.store.close)
        self.example = Example("locomo/test/0", "locomo", "test", "memory", "When did Ada visit?",
                               gold_answers=["SECRET_GOLD"], judge_metadata={"secret": "SCORING_ONLY"})
        self.evidence = [{"id": "READER_ID", "time": "2023-05-25 15:00", "raw_text": "Ada: I visited yesterday.",
                          "content": "DIRECTORY_SUMMARY", "private_field": "INTERNAL"},
                         {"id": "SECOND_ID", "time": "2023-06-01", "raw_text": "Bob: Hello."}]
        self.service_config = {"model": "fixture-qwen", "model_revision": "fixture-v1",
                               "base_url": "http://fixture/v1", "retries": 0}

    def service(self, replies, *, remote=False, retries=0):
        calls = []
        replies = iter(replies)
        config = {**self.service_config, "retries": retries}
        if remote:
            config.update(model="gpt-5-mini", model_revision="gpt-5-mini", request_profile="openai_chat",
                          api_key_env="OPENAI_API_KEY", send_seed=False, temperature=None,
                          max_tokens_parameter="max_completion_tokens")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "mock-key"}):
            service = ChatService(config, self.store, "task_agent")
        service.client.close()
        def respond(request):
            calls.append(json.loads(request.content))
            text, finish = next(replies)
            return httpx.Response(200, json={"choices": [{"message": {"content": text},
                                                         "finish_reason": finish}],
                                             "usage": {"completion_tokens": 12}})
        service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(service.client.close)
        return service, calls

    def test_exact_user_template_renders_only_original_chats_and_question(self):
        original = deepcopy(self.evidence)
        expected = (
            "I will give you several history chats between you and a user. Please answer the question based on "
            "the relevant chat history. Provide your final answer wrapped in <answer> ... </answer>.\n\n"
            "History Chats: Session 1\nDate: 2023-05-25 15:00\nAda: I visited yesterday.\n\n"
            "Session 2\nDate: 2023-06-01\nBob: Hello.\n\n"
            "Question: When did Ada visit?\n\n"
            "Please provide your final answer wrapped in <answer> ... </answer>.")
        messages = task_messages(self.example, self.evidence, "history_answer_v1")
        self.assertEqual(messages, [{"role": "user", "content": expected}])
        for forbidden in ("SECRET_GOLD", "SCORING_ONLY", "READER_ID", "SECOND_ID", "DIRECTORY_SUMMARY",
                          "INTERNAL", "give both the relative", "corresponding\nabsolute date"):
            self.assertNotIn(forbidden, expected)
        self.assertEqual(self.evidence, original)

    def test_contract_versions_preserve_legacy_and_bind_new_rendering_and_parser(self):
        legacy = {"task_context_version": "question_date_v1",
                  "task_prompt_sha256": "2a8298b4e30f51ba472c8fa443319b20d84ea86e303607d8212b7884f57a955f"}
        self.assertEqual(task_input_contract(), legacy)
        self.assertEqual(task_input_contract("legacy_v1"), legacy)
        self.assertEqual(task_messages(self.example, self.evidence), [
            {"role": "system", "content": TASK_SYSTEM},
            {"role": "user", "content": canonical({**self.example.public_task(), "retrieved_evidence": self.evidence})}])
        current = task_input_contract("history_answer_v1")
        self.assertEqual(current["task_prompt_sha256"], digest(HISTORY_ANSWER_TEMPLATE))
        self.assertEqual(current["task_prompt_version"], "history_answer_v1")
        self.assertEqual(current["history_rendering"], "session_time_raw_text_v1")
        self.assertEqual(current["answer_parser"], "answer_tag_v1")
        self.assertEqual(current["completion_policy"], "require_stop_v1")
        with self.assertRaises(ValueError):
            task_input_contract("unknown")
        with self.assertRaises(ValueError):
            task_messages(self.example, [], "unknown")

    def test_public_observation_and_options_survive_without_scoring_fields(self):
        example = deepcopy(self.example)
        example.observation = "Current date: 2023-06-01"
        example.options = {"A": "May", "B": "June"}
        prompt = task_messages(example, [], "history_answer_v1")[0]["content"]
        self.assertIn("Public observation: Current date: 2023-06-01", prompt)
        self.assertIn('Options: {"A":"May","B":"June"}', prompt)
        self.assertNotIn("SECRET_GOLD", prompt)
        with self.assertRaisesRegex(ValueError, "raw_text"):
            task_messages(example, [{"content": "Summary is not original history"}], "history_answer_v1")

    def test_both_services_use_identical_messages_and_score_only_tagged_answer(self):
        raw = "<answer> 24 May 2023 </answer>"
        payloads = []
        for remote in (False, True):
            service, calls = self.service([(raw, "stop")], remote=remote)
            agent = TaskAgent(service, 2048, prompt_version="history_answer_v1")
            metadata = {}
            with patch.dict(os.environ, {"OPENAI_API_KEY": "mock-key"}):
                answer, input_hash = agent.answer(self.example, self.evidence, metadata=metadata)
                again = {}
                self.assertEqual(agent.answer(self.example, self.evidence, metadata=again), (answer, input_hash))
            self.assertEqual(answer, "24 May 2023")
            self.assertEqual(metadata["raw_text"], raw)
            self.assertEqual(metadata["answer_parser"], "answer_tag_v1")
            self.assertEqual(metadata["finish_reason"], "stop")
            self.assertFalse(metadata["truncated"])
            self.assertEqual(again, metadata)
            self.assertEqual(len(calls), 1)
            self.assertEqual(input_hash, digest(calls[0]["messages"]))
            payloads.append(calls[0]["messages"])
        self.assertEqual(payloads[0], payloads[1])

    def test_bad_tags_retry_before_caching_then_preserve_valid_answer(self):
        service, calls = self.service([("missing tags", "stop"), ("<answer>May</answer>", "stop")], retries=1)
        agent = TaskAgent(service, prompt_version="history_answer_v1")
        with patch("agmem.services.time.sleep"), redirect_stdout(io.StringIO()):
            self.assertEqual(agent.answer(self.example, self.evidence)[0], "May")
        self.assertEqual(len(calls), 2)
        self.assertEqual(agent.answer(self.example, self.evidence)[0], "May")
        self.assertEqual(len(calls), 2)

    def test_strict_completion_rejects_length_even_when_tags_are_complete(self):
        service, calls = self.service([("<answer>May</answer>", "length")])
        agent = TaskAgent(service, prompt_version="history_answer_v1")
        with redirect_stdout(io.StringIO()), self.assertRaises(InfrastructureError):
            agent.answer(self.example, self.evidence)
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.store.completions.keys()), [])

    def test_strict_completion_cache_is_separate_from_legacy_length_cache(self):
        raw = "<answer>May</answer>"
        service, calls = self.service([(raw, "length"), (raw, "stop")])
        messages = task_messages(self.example, self.evidence, "history_answer_v1")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(service.complete(messages, 512), raw)
        self.assertEqual(TaskAgent(service, prompt_version="history_answer_v1").answer(
            self.example, self.evidence)[0], "May")
        self.assertEqual(len(calls), 2)

    def test_factory_passes_explicit_prompt_version_and_preserves_default(self):
        for version in (None, "history_answer_v1"):
            config = {"task_agent": {"max_output_tokens": 512}}
            if version is not None:
                config["task_agent"]["prompt_version"] = version
            with patch("agmem.services.resolve_task_service", return_value=self.service_config):
                agent = make_task_agent(config, self.store)
            self.addCleanup(agent.service.client.close)
            self.assertEqual(agent.prompt_version, version or "legacy_v1")


if __name__ == "__main__":
    unittest.main()
