"""Portable exported-evidence answering: no real provider or GPU required."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

from agmem.core import Example, digest as core_digest
from agmem.prompts import task_input_contract, task_messages
from agmem import remote_task_runner as runner


def fixture_bundle(count=2):
    requests = []
    for index in range(count):
        messages = task_messages(Example(sample_id=f"q{index}", source="locomo", split="test",
                                         memory_id="m", query=f"Question {index}", gold_answers=["DO_NOT_EXPORT"]),
                                 [{"id": "s1", "raw_text": "Historical evidence 中文"}])
        requests.append({"sample_id": f"q{index}", "messages": messages, "task_input_hash": runner.digest(messages)})
    return {"manifest": {"protocol": runner.REQUEST_PROTOCOL,
                         "sample_ids": [row["sample_id"] for row in requests],
                         "requests_sha256": runner.digest(requests),
                         "retrieval_manifest_sha256": "a" * 64, "source_task_manifest_sha256": "b" * 64,
                         "task_input_contract": task_input_contract(), "checkpoint": "best", "dataset": "locomo",
                         "request_settings": {"model": "gpt-5-mini", "model_revision": "gpt-5-mini",
                                              "request_profile": "openai_chat", "send_seed": False,
                                              "temperature": None, "max_tokens_parameter": "max_completion_tokens",
                                              "max_output_tokens": 512, "seed": 42}}, "requests": requests}


def completion(text="answer", finish_reason="stop"):
    return {"protocol": runner.COMPLETION_PROTOCOL, "text": text, "finish_reason": finish_reason,
            "truncated": finish_reason == "length", "usage": {"completion_tokens": 2}, "max_tokens": 512}


@contextmanager
def server(handler):
    # This execution sandbox cannot create sockets, including loopback ones.
    # Exercise real urllib Request objects at the transport boundary instead.
    class FakeOpener:
        def open(self, request, timeout):
            result, status = handler(json.loads(request.data), dict(request.header_items()),
                                     urlsplit(request.full_url).path)
            if status != 200:
                raise AssertionError("Unexpected fixture status")
            return io.BytesIO(json.dumps(result).encode())

    with patch.object(runner, "build_opener", return_value=FakeOpener()):
        yield "https://provider.invalid/v1"


class RemoteTaskRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = fixture_bundle()
        self.requests = self.root / "requests.json.gz"
        self.output = self.root / "responses"
        self.write_bundle()
        self.env = patch.dict(os.environ, {"OPENAI_BASE_URL": "https://provider.invalid/v1/",
                                          "OPENAI_API_KEY": "secret-never-persist"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)

    def write_bundle(self):
        with gzip.open(self.requests, "wt", encoding="utf-8") as stream:
            json.dump(self.bundle, stream, ensure_ascii=False)

    def run_fake(self, answer=None, **kwargs):
        answer = answer or completion()
        with patch.object(runner, "_complete", return_value=answer) as mock, redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output, **kwargs)
        return mock

    def test_hash_compatible_with_repository_and_no_gold_export(self):
        self.assertEqual(runner.digest(self.bundle), core_digest(self.bundle))
        self.assertEqual(runner.load_bundle(self.requests), self.bundle)
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(self.bundle))

    def test_script_is_standalone_with_isolated_python(self):
        copied = self.root / "remote_task_runner.py"
        shutil.copyfile(runner.__file__, copied)
        result = subprocess.run([sys.executable, "-I", str(copied), "--help"], cwd=self.root,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--requests", result.stdout)

    def test_public_payload_and_exact_authenticated_http_request(self):
        received = []

        def handle(payload, headers, path):
            received.append((payload, headers, path))
            return {"choices": [{"message": {"content": "中文 answer"}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 512}}, 200

        with server(handle) as endpoint, patch.dict(os.environ, {"OPENAI_BASE_URL": endpoint}), redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output, workers=2, retries=0)
        self.assertEqual(len(received), 2)
        payloads = [item[0] for item in received]
        for request in self.bundle["requests"]:
            self.assertIn({"model": "gpt-5-mini", "messages": request["messages"],
                           "max_completion_tokens": 512}, payloads)
        self.assertTrue(all(item[1]["Authorization"] == "Bearer secret-never-persist" for item in received))
        self.assertTrue(all(item[2] == "/v1/chat/completions" for item in received))
        manifest, rows = runner.load_response_records(self.bundle, self.output)
        self.assertTrue(all(row["completion"]["truncated"] for row in rows.values()))
        self.assertEqual(manifest["task"]["identity"]["base_url"], endpoint)
        for path in self.output.rglob("*.json"):
            self.assertNotIn("secret-never-persist", path.read_text())

    def test_resume_completed_answers_without_new_requests(self):
        self.run_fake()
        with patch.object(runner, "_complete", side_effect=AssertionError("Must use saved answers")), redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output)
        _, rows = runner.load_response_records(self.bundle, self.output)
        self.assertEqual(list(rows), ["q0", "q1"])

    def test_bounded_concurrency_and_drain_after_failure(self):
        self.bundle = fixture_bundle(6)
        self.write_bundle()
        active = peak = 0
        called = []
        lock = threading.Lock()

        def fake(request, *args):
            nonlocal active, peak
            with lock:
                called.append(request["sample_id"])
                active += 1
                peak = max(peak, active)
            if request["sample_id"] == "q0":
                time.sleep(.02)
                with lock:
                    active -= 1
                raise RuntimeError("deliberate failure")
            time.sleep(.08)
            with lock:
                active -= 1
            return completion()

        with patch.object(runner, "_complete", side_effect=fake), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                runner.run(self.requests, self.output, workers=2, retries=0)
        self.assertEqual(set(called), {"q0", "q1"})
        self.assertEqual(peak, 2)
        self.assertFalse((self.output / "complete.json").exists())
        _, saved = runner.load_response_records(self.bundle, self.output, require_complete=False)
        self.assertEqual(list(saved), ["q1"])
        mock = self.run_fake(workers=2)
        self.assertEqual(mock.call_count, 5)
        self.assertEqual(len(runner.load_response_records(self.bundle, self.output)[1]), 6)

    def test_tampered_requests_hash_or_input_hash_rejected(self):
        for mutate in (lambda b: b["requests"][0].update(task_input_hash="c" * 64),
                       lambda b: b["manifest"].update(requests_sha256="c" * 64),
                       lambda b: b["manifest"].update(sample_ids=["q0", "q0"])):
            bad = deepcopy(self.bundle)
            mutate(bad)
            with self.assertRaises(ValueError):
                runner.validate_bundle(bad)

    def test_gold_or_changed_system_prompt_rejected_even_with_new_hash(self):
        bad = deepcopy(self.bundle)
        public = json.loads(bad["requests"][0]["messages"][1]["content"])
        public["gold_answers"] = ["secret"]
        bad["requests"][0]["messages"][1]["content"] = json.dumps(public)
        bad["requests"][0]["task_input_hash"] = runner.digest(bad["requests"][0]["messages"])
        bad["manifest"]["requests_sha256"] = runner.digest(bad["requests"])
        with self.assertRaisesRegex(ValueError, "only the public"):
            runner.validate_bundle(bad)
        bad = deepcopy(self.bundle)
        bad["requests"][0]["messages"][0]["content"] = "changed prompt"
        bad["requests"][0]["task_input_hash"] = runner.digest(bad["requests"][0]["messages"])
        bad["manifest"]["requests_sha256"] = runner.digest(bad["requests"])
        with self.assertRaisesRegex(ValueError, "system prompt"):
            runner.validate_bundle(bad)

    def test_response_pairing_and_foreign_manifest_rejected(self):
        self.run_fake()
        altered = deepcopy(self.bundle)
        altered["manifest"]["checkpoint"] = "final"
        with self.assertRaisesRegex(ValueError, "match"):
            runner.load_response_records(altered, self.output)
        path = self.output / "episodes" / "00000000.json"
        row = json.loads(path.read_text())
        row["task_input_hash"] = "c" * 64
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.load_response_records(self.bundle, self.output)

    def test_changed_endpoint_cannot_reuse_answers(self):
        self.run_fake()
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://another.invalid/v1"}):
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                runner.run(self.requests, self.output)

    def test_complete_marker_and_unknown_episodes_are_validated(self):
        self.run_fake()
        marker = self.output / "complete.json"
        marker.unlink()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            runner.load_response_records(self.bundle, self.output)
        self.assertEqual(len(runner.load_response_records(self.bundle, self.output, require_complete=False)[1]), 2)
        (self.output / "episodes" / "00000099.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            runner.load_response_records(self.bundle, self.output, require_complete=False)

    def test_empty_answer_retried_and_no_completion_created(self):
        result = {"choices": [{"message": {"content": " "}, "finish_reason": "length"}], "usage": None}
        mocked_response = Mock()
        mocked_response.__enter__ = Mock(return_value=io.BytesIO(json.dumps(result).encode()))
        mocked_response.__exit__ = Mock(return_value=False)
        # Return a fresh stream on each retry.
        mocked_response.__enter__.side_effect = lambda: io.BytesIO(json.dumps(result).encode())
        opener = Mock()
        opener.open.return_value = mocked_response
        with patch.object(runner, "build_opener", return_value=opener), patch.object(runner.time, "sleep"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "no task score assigned"):
                runner.run(self.requests, self.output, workers=1, retries=1)
        self.assertEqual(opener.open.call_count, 2)
        self.assertEqual(list((self.output / "episodes").glob("*.json")), [])
        self.assertFalse((self.output / "complete.json").exists())

    def test_provider_errors_do_not_expose_secrets(self):
        opener = Mock()
        opener.open.side_effect = RuntimeError("echoed secret-never-persist with private prompt")
        out, err = io.StringIO(), io.StringIO()
        with patch.object(runner, "build_opener", return_value=opener), redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(RuntimeError) as caught:
                runner.run(self.requests, self.output, workers=1, retries=0)
        self.assertNotIn("secret-never-persist", str(caught.exception) + out.getvalue() + err.getvalue())

    def test_offline_import_never_needs_credentials(self):
        self.run_fake()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(len(runner.load_response_records(self.bundle, self.output)[1]), 2)

    def test_explicit_request_options_are_respected(self):
        settings = self.bundle["manifest"]["request_settings"]
        settings.update(send_seed=True, seed=7, temperature=.2, max_tokens_parameter="max_tokens")
        received = []

        def handle(payload, *args):
            received.append(payload)
            return {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}, 200

        with server(handle) as endpoint:
            runner._complete(self.bundle["requests"][0], settings, endpoint, "key", 5, 0)
        self.assertEqual(received[0]["seed"], 7)
        self.assertEqual(received[0]["temperature"], .2)
        self.assertEqual(received[0]["max_tokens"], 512)
        self.assertNotIn("max_completion_tokens", received[0])

    def test_endpoint_credentials_and_invalid_completion_rejected(self):
        for endpoint in ("file:///tmp/service", "https://user:pass@host/v1", "https://host/v1?key=secret", "https://host/#secret"):
            with self.assertRaises(ValueError):
                runner._endpoint(endpoint)
        for bad in (completion(""), {**completion(), "truncated": True}, {**completion(), "max_tokens": 1024}):
            with self.assertRaises(ValueError):
                runner.validate_completion(bad, 512)

    def test_response_directory_lock_prevents_second_writer(self):
        self.output.mkdir()
        with runner._output_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another runner"):
                runner.run(self.requests, self.output)


if __name__ == "__main__":
    unittest.main()
