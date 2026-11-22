"""Remote Writer completion/provenance contracts without network or GPU access."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier, Event
import unittest
from unittest.mock import patch
from urllib.error import URLError

from agmem.core import InfrastructureError
from agmem.remote_task_runner import COMPLETION_PROTOCOL
from agmem.remote_writer import create_writer_service, writer_identity
from agmem.store import Store


def completion(text="Subject visited Paris; visit was in May.", budget=2048, finish="stop"):
    return {"protocol": COMPLETION_PROTOCOL, "text": text, "max_tokens": budget,
            "finish_reason": finish, "truncated": finish == "length", "usage": {"completion_tokens": 20}}


class RemoteWriterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root / "store")
        self.addCleanup(lambda: self.store.close())
        self.config = {"models": {"task_path": "unused-local-model"},
                       "service": {"base_url": "http://localhost/v1", "model": "qwen"},
                       "writer_service": {"request_profile": "openai_chat", "model": "gpt-5-mini",
                                          "model_revision": "gpt-5-mini", "base_url": "https://fixture.invalid/v1",
                                          "summary_tokenizer_revision": "qwen-tokenizer-fixture", "retries": 1}}
        env = patch.dict(os.environ, {"OPENAI_BASE_URL": "", "OPENAI_API_KEY": "fixture-secret"})
        env.start()
        self.addCleanup(env.stop)
        self.messages = [{"role": "system", "content": "Writer prompt"},
                         {"role": "user", "content": "private original session"}]

    def service(self):
        return create_writer_service(self.config, self.store)

    def test_local_identity_and_factory_preserve_legacy_runtime_dependencies(self):
        config = deepcopy(self.config)
        config.pop("writer_service")
        with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "local-fingerprint"}), \
             patch("agmem.services.ChatService") as service:
            identity = writer_identity(config)
            self.assertEqual(identity, {"base_url": "http://localhost/v1", "model": "qwen",
                                        "model_revision": "local-fingerprint"})
            self.assertIs(create_writer_service(config, self.store), service.return_value)
            service.assert_called_once_with(identity, self.store, "writer")
        self.assertNotIn("model_revision", config["service"])

    def test_identity_needs_no_key_but_generation_requires_one(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            identity = writer_identity(self.config)
            with self.assertRaisesRegex(InfrastructureError, "OPENAI_API_KEY"):
                self.service()
        self.assertEqual(identity["base_url"], "https://fixture.invalid/v1")
        self.assertNotIn("api_key_env", identity)
        self.assertIsNone(identity["request_options"]["temperature"])
        self.assertEqual(identity["summary_tokenizer"], {"revision": "qwen-tokenizer-fixture"})
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://other.invalid/v1/"}):
            changed = writer_identity(self.config)
        self.assertNotEqual(identity, changed)
        self.assertEqual(changed["base_url"], "https://other.invalid/v1")

    def test_remote_tokenizer_identity_hashes_files_without_model_weights(self):
        model = self.root / "tokenizer"
        model.mkdir()
        (model / "tokenizer.json").write_text('{"version":"1"}')
        (model / "tokenizer_config.json").write_text('{}')
        config = deepcopy(self.config)
        config["models"]["task_path"] = str(model)
        config["writer_service"].pop("summary_tokenizer_revision")
        with patch("agmem.policy.model_fingerprint", side_effect=AssertionError("No weights required")):
            identity = writer_identity(config)
            (model / "tokenizer.json").write_text('{"version":"2"}')
            changed = writer_identity(config)
        self.assertNotEqual(identity, changed)
        self.assertEqual(set(identity["summary_tokenizer"]["files"]), {"tokenizer.json", "tokenizer_config.json"})

    def test_generation_policy_changes_identity_but_workers_retries_do_not(self):
        identity = writer_identity(self.config)
        config = deepcopy(self.config)
        config["execution"] = {"writer_workers": 8}
        config["writer_service"].update(retries=7, timeout_seconds=20)
        self.assertEqual(identity, writer_identity(config))
        config["writer_service"]["max_output_tokens"] = 4096
        self.assertNotEqual(identity, writer_identity(config))
        config["writer_service"]["base_url"] = "https://user:secret@fixture.invalid/v1"
        with self.assertRaisesRegex(ValueError, "credentials"):
            writer_identity(config)

    def test_summary_limit_is_separate_and_success_survives_store_reopen(self):
        service = self.service()
        with patch("agmem.remote_writer._complete", return_value=completion()) as request, redirect_stdout(io.StringIO()):
            result = service.complete(self.messages, 300, cache_scope="history/version")
        args = request.call_args.args
        self.assertEqual(args[0]["messages"], self.messages)
        self.assertEqual(args[1]["max_output_tokens"], 2048)
        self.assertEqual(args[7], 16384)
        self.assertIsNone(args[1]["temperature"])
        self.assertFalse(args[1]["send_seed"])
        self.store.close()
        self.store = Store(self.root / "store")
        with patch("agmem.remote_writer._complete", side_effect=AssertionError("Use persisted summary")):
            self.assertEqual(self.service().complete(self.messages, 300, cache_scope="history/version"), result)
        self.assertEqual(len(self.store.completions.keys()), 1)
        events = (self.store.root / "remote_writer_events.jsonl").read_text()
        self.assertIn("remote_writer_session_completed", events)
        self.assertNotIn("private original session", events)
        self.assertNotIn("fixture-secret", events)

    def test_nonempty_length_expands_without_caching_partial_summary(self):
        with patch("agmem.remote_writer._complete", side_effect=[completion("partial", finish="length"),
                   completion("complete", budget=8192)]) as request, redirect_stdout(io.StringIO()):
            self.assertEqual(self.service().complete(self.messages, 300), "complete")
        self.assertEqual([call.args[1]["max_output_tokens"] for call in request.call_args_list], [2048, 8192])
        self.assertEqual(len(self.store.completions.keys()), 1)
        row = self.store.completion(next(iter(self.store.completions.keys()))[0])
        self.assertEqual(row["completion"]["finish_reason"], "stop")
        self.assertEqual(row["completion"]["text"], "complete")

    def test_real_transport_empty_length_and_network_retries_expand_independently(self):
        calls = []
        sequence = iter([URLError("private error fixture-secret"),
                         {"finish_reason": "length", "message": {"content": ""}},
                         URLError("another private error"),
                         {"finish_reason": "stop", "message": {"content": "Complete summary."}}])

        class Opener:
            def open(inner, request, timeout):
                payload = json.loads(request.data)
                calls.append(payload)
                value = next(sequence)
                if isinstance(value, Exception):
                    raise value
                return io.BytesIO(json.dumps({"choices": [value], "usage": {"completion_tokens": 20}}).encode())

        with patch("agmem.remote_task_runner.build_opener", return_value=Opener()), \
             patch("agmem.remote_task_runner.time.sleep"), redirect_stdout(io.StringIO()):
            self.assertEqual(self.service().complete(self.messages, 300), "Complete summary.")
        self.assertEqual([call["max_completion_tokens"] for call in calls], [2048, 2048, 8192, 8192])
        self.assertTrue(all("temperature" not in call and "seed" not in call for call in calls))
        logs = (self.store.root / "remote_writer_events.jsonl").read_text()
        self.assertNotIn("private error", logs)
        self.assertNotIn("fixture-secret", logs)
        self.assertNotIn("private original session", logs)

    def test_failed_or_length_capped_results_never_enter_cache(self):
        self.config["writer_service"].update(max_completion_tokens=2048, retries=0)
        for result in (completion("partial", finish="length"), completion("refusal", finish="content_filter")):
            with patch("agmem.remote_writer._complete", return_value=result), redirect_stdout(io.StringIO()):
                with self.assertRaises(InfrastructureError):
                    self.service().complete(self.messages, 300)
        with patch("agmem.remote_writer._complete", side_effect=RuntimeError("fixture-secret")):
            with self.assertRaises(InfrastructureError) as failure:
                self.service().complete(self.messages, 300)
        self.assertNotIn("fixture-secret", str(failure.exception))
        self.assertEqual(len(self.store.completions.keys()), 0)

    def test_corrupt_cached_finish_reason_is_rejected(self):
        with patch("agmem.remote_writer._complete", return_value=completion()), redirect_stdout(io.StringIO()):
            service = self.service()
            service.complete(self.messages, 300)
        key = next(iter(self.store.completions.keys()))[0]
        record = self.store.completion(key)
        record["completion"].update(finish_reason="length", truncated=True)
        self.store.put_completion(key, record)
        with self.assertRaisesRegex(InfrastructureError, "incomplete"):
            service.complete(self.messages, 300)

    def test_same_concurrent_request_is_shared_without_serializing_other_requests(self):
        service = self.service()
        entered, release, barrier = Event(), Event(), Barrier(4)

        def generate(*args):
            entered.set()
            self.assertTrue(release.wait(5))
            return completion()

        def ask(_):
            barrier.wait(5)
            return service.complete(self.messages, 300)

        with patch("agmem.remote_writer._complete", side_effect=generate) as request, redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(ask, index) for index in range(4)]
                self.assertTrue(entered.wait(5))
                release.set()
                results = [future.result() for future in futures]
        self.assertEqual(request.call_count, 1)
        self.assertEqual(results, [completion()["text"]] * 4)
        self.assertFalse(service._inflight)


if __name__ == "__main__":
    unittest.main()
