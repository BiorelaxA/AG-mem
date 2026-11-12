"""Independent remote Task endpoints, without model weights or network calls."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import io
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem.core import Example, InfrastructureError, Trajectory, digest
from agmem.pipeline import scored_rollout
from agmem.services import ChatService, TASK_OUTPUT_POLICY, TaskAgent, resolve_task_service
from agmem.store import Store


class RemoteTaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.env = patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://provider.invalid/v1",
            "OPENAI_API_KEY": "fixture-secret-never-log",
        })
        self.env.start()
        self.remote = {
            "request_profile": "openai_chat", "model": "gpt-5-mini",
            "model_revision": "provider-deployment-v1", "base_url_env": "OPENAI_BASE_URL",
            "api_key_env": "OPENAI_API_KEY", "temperature": None,
            "max_tokens_parameter": "max_completion_tokens", "retries": 0,
        }
        self.local = {"model": "agmem-27b", "base_url": "http://127.0.0.1:8000/v1",
                      "request_base_url": "http://127.0.0.1:50001/v1", "api_key_env": "AGMEM_API_KEY"}

    def tearDown(self):
        self.env.stop()
        self.store.close()
        self.temp.cleanup()

    def service(self, config, handler):
        service = ChatService(config, self.store, "task_agent")
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(service.client.close)
        return service

    @staticmethod
    def response(answer="answer"):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                    "message": {"content": answer}}],
                                        "usage": {"prompt_tokens": 12, "completion_tokens": 2}})

    def test_task_override_is_independent_and_does_not_mutate_writer_judge(self):
        config = {"models": {"task_path": "local-27b"}, "service": deepcopy(self.local),
                  "task_agent": {"service": deepcopy(self.remote)}}
        original = deepcopy(config)
        with patch("agmem.policy.model_fingerprint", side_effect=AssertionError("Remote is not a local model")):
            resolved = resolve_task_service(config)
        self.assertEqual(config, original)
        self.assertEqual(resolved["base_url"], "https://provider.invalid/v1")
        self.assertEqual(resolved["model"], "gpt-5-mini")
        self.assertEqual(resolved["model_revision"], "provider-deployment-v1")
        self.assertNotIn("request_base_url", resolved)
        self.assertNotIn(os.environ["OPENAI_API_KEY"], json.dumps(resolved))

    def test_legacy_service_resolution_keeps_local_fingerprint_and_fields(self):
        config = {"models": {"task_path": "local-27b"}, "service": deepcopy(self.local), "task_agent": {}}
        original = deepcopy(config)
        with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "local-revision"}) as fingerprint:
            resolved = resolve_task_service(config)
        self.assertEqual(resolved, {**self.local, "model_revision": "local-revision"})
        fingerprint.assert_called_once_with("local-27b")
        self.assertEqual(config, original)

    def test_standard_payload_auth_and_task_metadata(self):
        calls = []
        def handle(request):
            calls.append(request)
            return self.response()
        service = self.service(self.remote, handle)
        metadata = {}
        messages = [{"role": "user", "content": "Question"}]
        self.assertEqual(service.complete(messages, 512, seed=9, metadata=metadata), "answer")
        self.assertEqual(str(calls[0].url), "https://provider.invalid/v1/chat/completions")
        self.assertEqual(calls[0].headers["Authorization"], "Bearer fixture-secret-never-log")
        self.assertEqual(json.loads(calls[0].content), {
            "model": "gpt-5-mini", "messages": messages, "max_completion_tokens": 512,
        })
        self.assertEqual(metadata["max_tokens"], 512)
        self.assertFalse(metadata["truncated"])
        self.assertNotIn(os.environ["OPENAI_API_KEY"], json.dumps(service.identity))
        self.assertNotIn(os.environ["OPENAI_API_KEY"], json.dumps(service.config))
        self.assertEqual(service.complete(messages, 512, seed=99), "answer")
        self.assertEqual(len(calls), 1)  # The omitted seed is not part of the request.

    def test_explicit_seed_temperature_and_legacy_token_limit_parameter(self):
        calls = []
        config = {**self.remote, "send_seed": True, "temperature": 0.0, "max_tokens_parameter": "max_tokens"}
        def handle(request):
            calls.append(json.loads(request.content))
            return self.response()
        service = self.service(config, handle)
        service.complete([{"role": "user", "content": "q"}], 100, seed=17)
        self.assertEqual(calls[0]["seed"], 17)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertEqual(calls[0]["max_tokens"], 100)
        self.assertNotIn("max_completion_tokens", calls[0])
        self.assertNotIn("chat_template_kwargs", calls[0])

    def test_legacy_request_and_cache_key_are_unchanged(self):
        config = {**self.local, "model_revision": "local-revision"}
        messages = [{"role": "user", "content": "q"}]
        request = {"model": "agmem-27b", "messages": messages, "temperature": 0.0,
                   "max_tokens": 10, "seed": 42, "chat_template_kwargs": {"enable_thinking": False}}
        identity = {key: config[key] for key in ("model", "model_revision", "base_url")}
        key = digest({"identity": identity, "role": "task_agent", "request": request,
                      "cache_scope": None, "response_protocol": TASK_OUTPUT_POLICY})
        self.store.put_completion(key, {"protocol": TASK_OUTPUT_POLICY, "text": "cached local answer",
                                       "finish_reason": "stop", "truncated": False, "usage": None,
                                       "max_tokens": 10})
        handler = Mock(side_effect=AssertionError("Existing local cache must be reusable"))
        service = self.service(config, handler)
        self.assertEqual(service.identity, identity)
        self.assertEqual(service.complete(messages, 10), "cached local answer")
        handler.assert_not_called()

    def test_actual_remote_endpoint_model_and_request_parameters_separate_cache(self):
        calls = []
        def handle(request):
            calls.append(request)
            return self.response(str(len(calls)))
        base = {k: v for k, v in self.remote.items() if k != "base_url_env"}
        configs = [
            {**base, "base_url": "https://identity.invalid/v1", "request_base_url": "https://a.invalid/v1"},
            {**base, "base_url": "https://identity.invalid/v1", "request_base_url": "https://b.invalid/v1"},
            {**base, "base_url": "https://a.invalid/v1", "model": "other-model"},
            {**base, "base_url": "https://a.invalid/v1", "max_tokens_parameter": "max_tokens"},
        ]
        services = [self.service(config, handle) for config in configs]
        messages = [{"role": "user", "content": "same q"}]
        self.assertEqual([s.complete(messages, 20) for s in services], ["1", "2", "3", "4"])
        self.assertEqual(services[0].identity["base_url"], "https://a.invalid/v1")
        self.assertEqual(services[0].complete(messages, 20), "1")
        self.assertEqual(len(calls), 4)

    def test_missing_environment_or_remote_revision_fails_before_client_creation(self):
        for name in ["OPENAI_BASE_URL", "OPENAI_API_KEY"]:
            with self.subTest(name=name), patch.dict(os.environ, {name: ""}), patch("httpx.Client") as client:
                with self.assertRaisesRegex(ValueError, name):
                    ChatService(self.remote, self.store, "task_agent")
                client.assert_not_called()
        for revision in [None, "", "REQUIRED_REMOTE_VERSION"]:
            with self.subTest(revision=revision), patch("httpx.Client") as client:
                with self.assertRaisesRegex(ValueError, "model_revision"):
                    ChatService({**self.remote, "model_revision": revision}, self.store, "task_agent")
                client.assert_not_called()

    def test_remote_failures_never_receive_a_task_score(self):
        example = Example("locomo/test/0", "locomo", "test", "memory", "q", gold_answers=["gold"])
        responses = [httpx.Response(429), httpx.Response(500), httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": ""}}]})]
        for index, response in enumerate(responses):
            with self.subTest(index=index):
                service = self.service({**self.remote, "model_revision": f"failure-{index}"}, lambda request: response)
                agent = TaskAgent(service)
                env = SimpleNamespace(run=Mock(return_value=Trajectory(example.sample_id, 42, "memory")))
                scorer = Mock()
                with self.assertRaises(InfrastructureError):
                    scored_rollout({}, env, SimpleNamespace(policy_version=20), agent, example, 42, scorer=scorer)
                scorer.score.assert_not_called()
        self.assertEqual(len(self.store.completions.keys()), 0)

    def test_credentials_are_redacted_from_remote_failure_logs(self):
        secret = os.environ["OPENAI_API_KEY"]
        def fail(request):
            raise RuntimeError("provider echoed credential " + secret)
        service = self.service(self.remote, fail)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(InfrastructureError) as raised:
            service.complete([{"role": "user", "content": "q"}], 10)
        self.assertNotIn(secret, output.getvalue())
        self.assertNotIn(secret, (self.store.root / "service_failures.jsonl").read_text())
        self.assertIn("[REDACTED]", (self.store.root / "service_failures.jsonl").read_text())
        self.assertTrue(raised.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
