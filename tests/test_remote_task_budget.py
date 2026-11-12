"""Recover only empty, length-limited answers while retaining paired evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from agmem import remote_task_runner as runner
from tests.test_remote_task_runner import completion, fixture_bundle


class RemoteTaskBudgetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = self.root / "requests.json.gz"
        self.responses = self.root / "responses"
        self.bundle = fixture_bundle()
        self.settings = self.bundle["manifest"]["request_settings"]
        self.endpoint = "https://provider.invalid/v1"
        with gzip.open(self.requests, "wt") as stream:
            json.dump(self.bundle, stream)
        env = patch.dict(os.environ, {"OPENAI_BASE_URL": self.endpoint,
                                      "OPENAI_API_KEY": "secret-budget-key"})
        env.start()
        self.addCleanup(env.stop)

    @staticmethod
    def response(text=None, finish="length", usage=None):
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": usage if usage is not None else {"completion_tokens": 512,
                    "completion_tokens_details": {"reasoning_tokens": 512}}}

    def opener(self, replies):
        payloads = []
        replies = iter(replies)
        def open_response(request, timeout):
            payloads.append(json.loads(request.data))
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return io.BytesIO(json.dumps(reply).encode())
        opener = Mock()
        opener.open.side_effect = open_response
        return opener, payloads

    def call(self, replies, retries=3, cap=16384, settings=None):
        opener, payloads = self.opener(replies)
        events = []
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"):
            value = runner._complete(self.bundle["requests"][0], settings or self.settings,
                                     self.endpoint, "secret-budget-key", 10, retries,
                                     lambda event, **fields: events.append({"event": event, **fields}), cap)
        return value, payloads, events

    def legacy_partial(self):
        with patch.object(runner, "_complete", side_effect=[completion("saved answer"), RuntimeError("stop")]), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                runner.run(self.requests, self.responses, workers=1, retries=0)

    def test_empty_length_answer_retries_with_actual_budget_and_identical_input(self):
        answer, payloads, events = self.call([
            self.response(), self.response("usable answer", "stop", {"completion_tokens": 850})])
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512, 2048])
        self.assertEqual(answer["max_tokens"], 2048)
        self.assertEqual(answer["text"], "usable answer")
        self.assertEqual(payloads[0]["messages"], payloads[1]["messages"])
        self.assertEqual(payloads[0]["messages"], self.bundle["requests"][0]["messages"])
        failed = next(e for e in events if e["event"] == "remote_task_attempt_failed")
        self.assertEqual((failed["reason"], failed["finish_reason"], failed["max_tokens"],
                          failed["next_max_tokens"]), ("empty_text", "length", 512, 2048))
        self.assertEqual(failed["reasoning_tokens"], 512)

    def test_nonempty_truncated_answer_retains_existing_policy(self):
        answer, payloads, _ = self.call([self.response("partial answer", "length")])
        self.assertEqual(len(payloads), 1)
        self.assertTrue(answer["truncated"])
        self.assertEqual(answer["max_tokens"], 512)

    def test_other_validation_failures_do_not_raise_token_budget(self):
        for response in [self.response("", "stop"), self.response(["not a string"], "length"),
                         self.response("answer", "stop", usage="malformed usage")]:
            with self.subTest(response=response):
                opener, payloads = self.opener([response] * 4)
                with patch.object(runner, "build_opener", return_value=opener), \
                     patch.object(runner.time, "sleep"):
                    with self.assertRaises(RuntimeError):
                        runner._complete(self.bundle["requests"][0], self.settings,
                                         self.endpoint, "key", 10, 3, None, 16384)
                self.assertEqual([p["max_completion_tokens"] for p in payloads], [512] * 4)

    def test_token_cap_is_bounded_and_empty_results_never_count_as_answers(self):
        opener, payloads = self.opener([self.response()] * 6)
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "empty answer at token limit 3000"):
                runner._complete(self.bundle["requests"][0], self.settings,
                                 self.endpoint, "key", 10, 3, None, 3000)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512, 2048] + [3000] * 4)

    def test_empty_answer_after_last_transport_retry_still_expands_budget(self):
        # Regression for the actual failure: attempts 1-3 could not connect,
        # attempt 4 returned HTTP success but used all 512 tokens on reasoning.
        # That response must still receive a fifth attempt with more tokens.
        answer, payloads, events = self.call([
            URLError("temporary connection failure"),
            URLError("temporary connection failure"),
            URLError("temporary connection failure"),
            self.response(), self.response("recovered answer", "stop")])
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512] * 4 + [2048])
        self.assertEqual((answer["text"], answer["max_tokens"]), ("recovered answer", 2048))
        failed = [e for e in events if e["event"] == "remote_task_attempt_failed"]
        self.assertEqual([e["attempt"] for e in failed], [1, 2, 3, 4])
        self.assertEqual(failed[-1]["retry_kind"], "increase_token_budget")
        self.assertTrue(failed[-1]["will_retry"])
        self.assertEqual(failed[-1]["next_max_tokens"], 2048)
        fifth = next(e for e in events if e["event"] == "remote_task_attempt_started" and e["attempt"] == 5)
        self.assertEqual(fifth["retries_used_at_budget"], 0)
        self.assertEqual(fifth["budget_expansions"], 1)

    def test_expanded_budget_has_its_own_transport_retry_allowance(self):
        failures = lambda: [URLError("temporary connection failure") for _ in range(3)]
        answer, payloads, events = self.call([
            *failures(), self.response(), *failures(), self.response("recovered answer", "stop")])
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512] * 4 + [2048] * 4)
        self.assertEqual(answer["max_tokens"], 2048)
        failures_at_2048 = [e for e in events if e["event"] == "remote_task_attempt_failed"
                            and e["max_tokens"] == 2048]
        self.assertEqual(len(failures_at_2048), 3)
        self.assertTrue(all(e["will_retry"] for e in failures_at_2048))
        self.assertTrue(all(e["retry_kind"] == "retry_current_budget" for e in failures_at_2048))
        self.assertEqual(events[-1]["attempt"], 8)

    def test_zero_ordinary_retries_still_allows_all_budget_expansions(self):
        answer, payloads, events = self.call([
            self.response(), self.response(), self.response(), self.response("answer", "stop")], retries=0)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512, 2048, 8192, 16384])
        self.assertEqual(answer["max_tokens"], 16384)
        failed = [e for e in events if e["event"] == "remote_task_attempt_failed"]
        self.assertEqual(len(failed), 3)
        self.assertTrue(all(e["will_retry"] for e in failed))
        self.assertTrue(all(e["retry_kind"] == "increase_token_budget" for e in failed))

    def test_transport_failure_exhaustion_does_not_increase_budget(self):
        opener, payloads = self.opener([URLError("temporary connection failure") for _ in range(4)])
        events = []
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "URLError"):
                runner._complete(self.bundle["requests"][0], self.settings,
                                 self.endpoint, "key", 10, 3,
                                 lambda event, **fields: events.append({"event": event, **fields}), 16384)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512] * 4)
        failed = [e for e in events if e["event"] == "remote_task_attempt_failed"]
        self.assertFalse(failed[-1]["will_retry"])
        self.assertIsNone(failed[-1]["next_max_tokens"])

    def test_combined_retry_paths_have_finite_request_bound(self):
        replies = []
        for _ in range(3):
            replies.extend(URLError("temporary connection failure") for _ in range(3))
            replies.append(self.response())
        replies.extend(URLError("temporary connection failure") for _ in range(4))
        opener, payloads = self.opener(replies)
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "URLError"):
                runner._complete(self.bundle["requests"][0], self.settings,
                                 self.endpoint, "key", 10, 3, None, 16384)
        self.assertEqual([p["max_completion_tokens"] for p in payloads],
                         [512] * 4 + [2048] * 4 + [8192] * 4 + [16384] * 4)

    def test_nonreasoning_max_tokens_requests_do_not_enable_policy(self):
        settings = {**self.settings, "max_tokens_parameter": "max_tokens"}
        opener, payloads = self.opener([self.response()] * 2)
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"):
            with self.assertRaises(RuntimeError):
                runner._complete(self.bundle["requests"][0], settings,
                                 self.endpoint, "key", 10, 1, None, 16384)
        self.assertEqual([p["max_tokens"] for p in payloads], [512, 512])
        bundle = deepcopy(self.bundle)
        bundle["manifest"]["request_settings"] = settings
        self.assertEqual(runner.response_manifest(bundle, self.endpoint, 16384)["protocol"],
                         runner.RESPONSE_PROTOCOL)

    def test_partial_legacy_upgrade_preserves_successes_and_resumes_only_missing(self):
        self.legacy_partial()
        legacy_path = self.responses / "episodes/00000000.json"
        legacy_bytes = legacy_path.read_bytes()
        legacy_manifest = json.loads((self.responses / "manifest.json").read_text())
        opener, payloads = self.opener([self.response(), self.response("new answer", "stop")])
        with patch.object(runner, "build_opener", return_value=opener), redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.responses, workers=1, empty_answer_max_tokens=16384)
        self.assertEqual(legacy_path.read_bytes(), legacy_bytes)
        manifest, rows = runner.load_response_records(self.bundle, self.responses)
        self.assertEqual(manifest["protocol"], runner.ADAPTIVE_RESPONSE_PROTOCOL)
        self.assertEqual(manifest["legacy_manifest_sha256"], runner.digest(legacy_manifest))
        self.assertEqual(manifest["task"]["request_options"]["empty_answer_retry"]["max_completion_tokens"], 16384)
        self.assertEqual(rows["q0"]["completion"]["max_tokens"], 512)
        self.assertEqual(rows["q1"]["completion"]["max_tokens"], 2048)
        self.assertTrue(all(p["messages"] == self.bundle["requests"][1]["messages"] for p in payloads))
        # Omitted library setting adopts the saved adaptive policy on resume.
        with patch.object(runner, "_complete", side_effect=AssertionError("Must resume saved rows")), \
             redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.responses)

    def test_atomic_upgrade_can_read_old_complete_marker_and_republish_new_signature(self):
        with patch.object(runner, "_complete", return_value=completion()), redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.responses)
        old_marker = json.loads((self.responses / "complete.json").read_text())
        updated_manifest = runner.response_manifest(self.bundle, self.endpoint, 16384)
        runner._atomic_json(self.responses / "manifest.json", updated_manifest)
        # Simulate interruption just after the only manifest write: all old
        # episodes and the old completion marker must still be readable.
        manifest, rows = runner.load_response_records(self.bundle, self.responses)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(old_marker["manifest_sha256"], runner.digest(manifest))
        with patch.object(runner, "_complete", side_effect=AssertionError("No repeat requests")), \
             redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.responses)
        self.assertEqual(json.loads((self.responses / "complete.json").read_text())["manifest_sha256"],
                         runner.digest(updated_manifest))

    def test_foreign_endpoint_and_changed_policy_cannot_reuse_answers(self):
        self.legacy_partial()
        original = (self.responses / "manifest.json").read_bytes()
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://other.invalid/v1"}), \
             patch.object(runner, "_complete", side_effect=AssertionError("No API call")):
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                runner.run(self.requests, self.responses, empty_answer_max_tokens=16384)
        self.assertEqual((self.responses / "manifest.json").read_bytes(), original)
        manifest = runner.response_manifest(self.bundle, self.endpoint, 16384)
        runner._atomic_json(self.responses / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            runner.run(self.requests, self.responses, empty_answer_max_tokens=8192)

    def test_import_rejects_false_legacy_budget_and_unsupported_policy(self):
        self.legacy_partial()
        manifest = runner.response_manifest(self.bundle, self.endpoint, 16384)
        runner._atomic_json(self.responses / "manifest.json", manifest)
        path = self.responses / "episodes/00000000.json"
        original = json.loads(path.read_text())
        bad = deepcopy(original)
        bad["completion"]["max_tokens"] = 2048
        runner._atomic_json(path, bad)
        with self.assertRaisesRegex(ValueError, "budget"):
            runner.load_response_records(self.bundle, self.responses, require_complete=False)
        bad["manifest_sha256"] = runner.digest(manifest)
        bad["completion"]["max_tokens"] = 1024
        runner._atomic_json(path, bad)
        with self.assertRaisesRegex(ValueError, "budget"):
            runner.load_response_records(self.bundle, self.responses, require_complete=False)
        runner._atomic_json(path, original)
        manifest["task"]["request_options"]["empty_answer_retry"]["multiplier"] = 2
        runner._atomic_json(self.responses / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "protocol"):
            runner.load_response_records(self.bundle, self.responses, require_complete=False)

    def test_cli_enables_adaptive_budget_and_invalid_caps_fail_before_output(self):
        with patch.object(runner, "run") as run, \
             patch("sys.argv", ["runner.py", "--requests", str(self.requests), "--output", str(self.responses)]):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(run.call_args.args[-1], 16384)
        for invalid in (True, 511, -1, "16384"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "empty-answer-max-tokens"):
                runner.run(self.requests, self.responses, empty_answer_max_tokens=invalid)
        self.assertFalse(self.responses.exists())


if __name__ == "__main__":
    unittest.main()
