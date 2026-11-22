"""Request failures and slow in-flight work remain diagnosable without private text."""
from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from agmem import remote_task_runner as runner
from tests.test_remote_task_runner import completion, fixture_bundle


class RemoteTaskDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = self.root / "requests.json.gz"
        self.output = self.root / "responses"
        self.bundle = fixture_bundle(1)
        self.write_bundle()
        self.env = patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://provider.invalid/v1",
            "OPENAI_API_KEY": "secret-never-persist",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_bundle(self):
        with gzip.open(self.requests, "wt", encoding="utf-8") as stream:
            json.dump(self.bundle, stream)

    def events(self):
        path = self.output / "events.jsonl"
        if not path.exists():
            return []
        # A concurrent writer can still be publishing the last line.
        lines = path.read_text().splitlines(keepends=True)
        return [json.loads(line) for line in lines if line.endswith("\n")]

    def assert_private_text_absent(self, console, *private):
        rendered = console.getvalue() + (self.output / "events.jsonl").read_text()
        for value in ("secret-never-persist", *private):
            self.assertNotIn(value, rendered)

    def response_opener(self, result):
        opener = Mock()
        opener.open.side_effect = lambda *a, **kw: io.BytesIO(json.dumps(result).encode())
        return opener

    def test_empty_length_response_reports_budget_and_retries_without_private_payload(self):
        result = {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 512, "completion_tokens_details": {
                "reasoning_tokens": 512}, "private": "usage-must-not-be-dumped"},
        }
        opener, console = self.response_opener(result), io.StringIO()
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"), redirect_stdout(console):
            with self.assertRaisesRegex(RuntimeError, "empty_text"):
                runner.run(self.requests, self.output, workers=1, retries=1)
        failures = [e for e in self.events() if e["event"] == "remote_task_attempt_failed"]
        self.assertEqual(len(failures), 2)
        self.assertEqual([e["attempt"] for e in failures], [1, 2])
        self.assertEqual([e["will_retry"] for e in failures], [True, False])
        for event in failures:
            self.assertEqual(event["sample_id"], "q0")
            self.assertEqual(event["reason"], "empty_text")
            self.assertEqual(event["error_type"], "ValueError")
            self.assertEqual(event["finish_reason"], "length")
            self.assertEqual(event["content_chars"], 0)
            self.assertEqual(event["completion_tokens"], 512)
            self.assertEqual(event["reasoning_tokens"], 512)
        self.assertFalse((self.output / "complete.json").exists())
        self.assertEqual(list((self.output / "episodes").glob("*.json")), [])
        self.assert_private_text_absent(console, "usage-must-not-be-dumped")

    def test_invalid_usage_is_distinguished_without_logging_the_answer(self):
        private_answer = "private-answer-content-must-not-be-logged"
        result = {"choices": [{"message": {"content": private_answer}, "finish_reason": "stop"}],
                  "usage": "malformed-usage-containing-secret-never-persist"}
        console = io.StringIO()
        with patch.object(runner, "build_opener", return_value=self.response_opener(result)), \
             redirect_stdout(console):
            with self.assertRaisesRegex(RuntimeError, "invalid_usage"):
                runner.run(self.requests, self.output, workers=1, retries=0)
        failure = next(e for e in self.events() if e["event"] == "remote_task_attempt_failed")
        self.assertEqual(failure["reason"], "invalid_usage")
        self.assertEqual(failure["error_type"], "ValueError")
        self.assertEqual(failure["content_chars"], len(private_answer))
        self.assert_private_text_absent(console, private_answer, "malformed-usage")

    def test_http_status_is_logged_without_provider_error_body_or_key(self):
        private = "provider-body-secret-never-persist"
        error = HTTPError("https://provider.invalid/v1/chat/completions", 429,
                          private, {}, io.BytesIO(private.encode()))
        opener = Mock()
        opener.open.side_effect = error
        console = io.StringIO()
        with patch.object(runner, "build_opener", return_value=opener), redirect_stdout(console):
            with self.assertRaisesRegex(RuntimeError, "HTTPError"):
                runner.run(self.requests, self.output, workers=1, retries=0)
        failure = next(e for e in self.events() if e["event"] == "remote_task_attempt_failed")
        self.assertEqual(failure["status_code"], 429)
        self.assertEqual(failure["error_type"], "HTTPError")
        self.assert_private_text_absent(console, private)

    def test_first_failure_is_reported_before_other_requests_drain_and_resume_reuses_success(self):
        self.bundle = fixture_bundle(3)
        self.write_bundle()
        slow_started = threading.Event()
        seen_failure = []
        calls = []

        def answer(request, *args, **kwargs):
            calls.append(request["sample_id"])
            if request["sample_id"] == "q0":
                if not slow_started.wait(3):
                    raise AssertionError("Second worker did not start")
                raise RuntimeError("Request failed")
            if request["sample_id"] != "q1":
                raise AssertionError("New work must stop after a terminal failure")
            slow_started.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                failure = next((e for e in self.events() if e["event"] == "remote_task_failed"), None)
                if failure is not None:
                    seen_failure.append(failure)
                    return completion("preserved success")
                time.sleep(.01)
            raise AssertionError("Failure was hidden until in-flight work ended")

        with patch.object(runner, "_complete", side_effect=answer), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Request failed"):
                runner.run(self.requests, self.output, workers=2, retries=0)
        self.assertEqual(set(calls), {"q0", "q1"})
        self.assertEqual(len(seen_failure), 1)
        self.assertEqual(seen_failure[0]["sample_id"], "q0")
        self.assertGreaterEqual(seen_failure[0]["draining_requests"], 1)
        self.assertFalse((self.output / "complete.json").exists())
        _, rows = runner.load_response_records(self.bundle, self.output, require_complete=False)
        self.assertEqual(set(rows), {"q1"})
        with patch.object(runner, "_complete", return_value=completion("new success")) as again, \
             redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output, workers=2, retries=0)
        self.assertEqual(again.call_count, 2)
        _, rows = runner.load_response_records(self.bundle, self.output)
        self.assertEqual(rows["q1"]["completion"]["text"], "preserved success")

    def test_heartbeat_exposes_pending_sample_without_waiting_for_a_response(self):
        observed = []

        def answer(request, *args, **kwargs):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                waiting = next((e for e in self.events() if e["event"] == "remote_task_waiting"), None)
                if waiting is not None:
                    observed.append(waiting)
                    return completion()
                time.sleep(.01)
            raise AssertionError("No heartbeat while a request was pending")

        with patch.object(runner, "WAIT_REPORT_SECONDS", .01), \
             patch.object(runner, "_complete", side_effect=answer), redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output, workers=1, retries=0)
        self.assertTrue(observed)
        self.assertIn("q0", json.dumps(observed[0]))
        self.assertTrue((self.output / "complete.json").exists())

    def test_success_diagnostics_do_not_change_payload_or_trigger_requests_on_resume(self):
        private_answer = "answer body intentionally absent from diagnostic events"
        result = {"choices": [{"message": {"content": private_answer}, "finish_reason": "stop"}],
                  "usage": {"completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 2}}}
        opener, console = self.response_opener(result), io.StringIO()
        with patch.object(runner, "build_opener", return_value=opener), redirect_stdout(console):
            runner.run(self.requests, self.output, workers=1, retries=0)
        successes = [e for e in self.events() if e["event"] == "remote_task_attempt_succeeded"]
        self.assertEqual(len(successes), 1)
        self.assertEqual(successes[0]["sample_id"], "q0")
        self.assertEqual(successes[0]["completion_tokens"], 9)
        _, rows = runner.load_response_records(self.bundle, self.output)
        self.assertEqual(rows["q0"]["completion"]["text"], private_answer)
        self.assert_private_text_absent(console, private_answer)
        with patch.object(runner, "_complete", side_effect=AssertionError("No new request expected")), \
             redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output, workers=1, retries=0)


if __name__ == "__main__":
    unittest.main()
