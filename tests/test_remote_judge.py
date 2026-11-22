"""Frozen-prompt remote judging with strict labels; never calls a real provider."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
import errno
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch

from agmem.core import Example, InfrastructureError
from agmem.judges import load_judge_templates, prompt_for
from agmem.remote_judge import RemoteJudgeScorer
from agmem.remote_task_runner import COMPLETION_PROTOCOL


def completion(text='{"label":"CORRECT"}', *, finish="stop", budget=2048, usage=None):
    return {"protocol": COMPLETION_PROTOCOL, "text": text, "finish_reason": finish,
            "truncated": finish == "length", "usage": usage, "max_tokens": budget}


def failed_transport(*args, transport_type="gaierror", error_type="URLError",
                     reason="response_or_transport_error", status_code=None, budget=None, number=None):
    request, settings, _, _, _, _, report, _ = args
    report("remote_task_attempt_failed", sample_id=request["sample_id"],
           error_type=error_type, transport_error_type=transport_type, transport_errno=number,
           reason=reason, status_code=status_code, will_retry=False,
           max_tokens=budget or settings["max_output_tokens"])
    raise RuntimeError("secret-do-not-persist raw provider error https://private.invalid")


class RemoteJudgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.remote = {"model": "gpt-5-mini", "model_revision": "gpt-5-mini",
                       "request_profile": "openai_chat", "base_url_env": "OPENAI_BASE_URL",
                       "api_key_env": "OPENAI_API_KEY", "send_seed": False,
                       "temperature": None, "max_tokens_parameter": "max_completion_tokens",
                       "timeout_seconds": 2, "retries": 2}
        path = Path("configs/protocols/format05_training_judge.json")
        snapshot = json.loads(path.read_text())
        self.judge_config = {"template_file": str(path), "expected_template_sha256": snapshot["template_sha256"],
                             "max_output_tokens": 128, "seed": 42}
        self.example = Example("q0", "locomo", "test", "m", "What colors?", gold_answers=["blue and red"])
        env = patch.dict(os.environ, {"OPENAI_BASE_URL": "https://provider.invalid/v1/",
                                      "OPENAI_API_KEY": "secret-do-not-persist"})
        env.start()
        self.addCleanup(env.stop)

    def judge(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return RemoteJudgeScorer(self.remote, self.judge_config, self.root / "diagnostics", **kwargs)

    def events(self):
        path = self.root / "diagnostics/events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_exact_frozen_prompt_and_honest_remote_provenance(self):
        judge = self.judge()
        with patch("agmem.remote_judge._complete", return_value=completion()) as call:
            self.assertEqual(judge.score(self.example, "red and blue"), 1.0)
        expected, _ = prompt_for(self.example, "blue and red", "red and blue", "locomo_refined",
                                 templates=load_judge_templates(self.judge_config)[0])
        self.assertEqual(call.call_args.args[0]["messages"], [{"role": "user", "content": expected}])
        settings = call.call_args.args[1]
        self.assertEqual(settings["max_output_tokens"], 2048)
        self.assertIsNone(settings["temperature"])
        self.assertFalse(settings["send_seed"])
        provenance = judge.provenance()
        self.assertEqual(provenance["template_sha256"], self.judge_config["expected_template_sha256"])
        self.assertEqual(provenance["model"]["model"], "gpt-5-mini")
        self.assertEqual(provenance["gold_aggregation"], "any_complete_reference_v1")
        self.assertNotIn("enable_thinking", provenance)
        self.assertIsNone(provenance["seed"])
        self.assertEqual(provenance["max_output_tokens"], 2048)

    def test_complete_references_are_alternatives_not_split_facts(self):
        example = deepcopy(self.example)
        example.gold_answers = ["blue and red", "red and blue", "red and blue"]
        judge = self.judge()
        with patch("agmem.remote_judge._complete", side_effect=[completion('{"label":"WRONG"}'), completion()]) as call:
            self.assertEqual(judge.score(example, "red and blue"), 1)
        self.assertEqual(call.call_count, 2)
        for expected, invocation in zip(example.gold_answers, call.call_args_list):
            self.assertIn("Gold answer: " + expected + "\n", invocation.args[0]["messages"][0]["content"])
        with patch("agmem.remote_judge._complete", side_effect=AssertionError("Use cached labels")):
            self.assertEqual(judge.score(example, "red and blue"), 1)

    def test_wrong_is_only_a_valid_wrong_label_and_invalid_labels_fail(self):
        judge = self.judge()
        with patch("agmem.remote_judge._complete", return_value=completion('{"label":"WRONG"}')):
            self.assertEqual(judge.score(self.example, "green"), 0)
        with patch("agmem.remote_judge._complete", return_value=completion("maybe")) as call, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(InfrastructureError, "invalid_label"):
                judge.score(self.example, "different prediction")
        self.assertEqual(call.call_count, 3)
        calls = list((self.root / "diagnostics/calls").glob("*.json"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].read_text())["label"], "WRONG")

    def test_valid_json_ended_by_length_is_retried_not_counted(self):
        judge = self.judge()
        with patch("agmem.remote_judge._complete", side_effect=[completion(finish="length"),
                                                               completion(budget=8192)]) as call, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(judge.score(self.example, "blue and red"), 1)
        self.assertEqual([item.args[1]["max_output_tokens"] for item in call.call_args_list], [2048, 8192])
        self.assertEqual(self.events()[-1]["reason"], "length")
        record = json.loads(next((self.root / "diagnostics/calls").glob("*.json")).read_text())
        self.assertEqual(record["completion"]["max_tokens"], 8192)
        self.assertFalse(record["completion"]["truncated"])

    def test_empty_length_recovery_uses_shared_transport_and_never_truncates_label(self):
        payloads = []
        replies = iter([
            {"choices": [{"message": {"content": None}, "finish_reason": "length"}],
             "usage": {"completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": 2048}}},
            {"choices": [{"message": {"content": '{"label":"CORRECT"}'}, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 2500}},
        ])
        opener = Mock()
        def request(req, timeout):
            payloads.append(json.loads(req.data))
            return io.BytesIO(json.dumps(next(replies)).encode())
        opener.open.side_effect = request
        judge = self.judge()
        with patch("agmem.remote_task_runner.build_opener", return_value=opener), redirect_stdout(io.StringIO()):
            self.assertEqual(judge.score(self.example, "blue and red"), 1)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [2048, 8192])
        self.assertNotIn("seed", payloads[0])
        self.assertNotIn("temperature", payloads[0])
        self.assertEqual(payloads[0]["messages"], payloads[1]["messages"])

    def test_capped_truncated_judge_never_becomes_wrong(self):
        judge = self.judge(max_completion_tokens=2048)
        with patch("agmem.remote_judge._complete", return_value=completion(finish="length")) as call, \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(InfrastructureError, "length"):
                judge.score(self.example, "blue")
        self.assertEqual(call.call_count, 3)
        self.assertFalse((self.root / "diagnostics/calls").exists())

    def test_same_call_cache_is_thread_safe_and_bound_to_input_and_provenance(self):
        judge = self.judge()
        def remote(*args):
            time.sleep(.02)
            return completion()
        with patch("agmem.remote_judge._complete", side_effect=remote) as call:
            with ThreadPoolExecutor(max_workers=2) as pool:
                scores = list(pool.map(lambda _: judge.score(self.example, "blue"), range(2)))
        self.assertEqual(scores, [1, 1])
        self.assertEqual(call.call_count, 1)
        restored = self.judge()
        with patch("agmem.remote_judge._complete", side_effect=AssertionError("cached")):
            self.assertEqual(restored.score(self.example, "blue"), 1)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.judge(max_output_tokens=4096)
        path = next((self.root / "diagnostics/calls").glob("*.json"))
        value = json.loads(path.read_text())
        value["input_sha256"] = "incorrect"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "provenance"):
            restored.score(self.example, "blue")

    def test_diagnostics_do_not_persist_secrets_or_raw_invalid_labels(self):
        judge = self.judge()
        with patch("agmem.remote_judge._complete", return_value=completion(
                "secret-do-not-persist echoed provider data")), redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(InfrastructureError) as caught:
                judge.score(self.example, "blue")
        combined = out.getvalue() + str(caught.exception)
        for path in (self.root / "diagnostics").rglob("*"):
            if path.is_file():
                combined += path.read_text()
        self.assertNotIn("secret-do-not-persist", combined)
        self.assertNotIn("echoed provider data", combined)
        with patch("agmem.remote_judge._complete", return_value=completion(
                usage={"debug": "secret-do-not-persist"})):
            self.assertEqual(judge.score(self.example, "blue"), 1)
        self.assertNotIn("secret-do-not-persist", next((self.root / "diagnostics/calls").glob("*.json")).read_text())

    def test_training_template_mismatch_and_bad_references_fail_before_api(self):
        self.judge_config["expected_template_sha256"] = {"bad": "hash"}
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.judge()
        del self.judge_config["expected_template_sha256"]
        judge = self.judge()
        self.example.gold_answers = []
        with patch("agmem.remote_judge._complete", side_effect=AssertionError("No API call")):
            with self.assertRaisesRegex(ValueError, "reference"):
                judge.score(self.example, "blue")

    def test_longmemeval_uses_official_branch_and_strict_yes_no_parser(self):
        judge = self.judge()
        example = Example("lme1", "split_longmemeval_s", "test", "m", "When?", gold_answers=["unknown"],
                          judge_metadata={"question_id": "lme1_abs", "question_type": "temporal-reasoning"})
        with patch("agmem.remote_judge._complete", return_value=completion("yes")) as call:
            self.assertEqual(judge.score(example, "not enough information"), 1)
        self.assertIn("unanswerable question", call.call_args.args[0]["messages"][0]["content"])

    def test_network_failure_recovers_after_cooldown_without_changing_ordinary_retries(self):
        self.remote["retries"] = 3
        judge = self.judge()
        attempts = 0

        def remote(*args):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                failed_transport(*args)
            return completion()

        with patch("agmem.remote_judge._complete", side_effect=remote) as request, \
             patch("agmem.remote_judge.sleep") as sleep, redirect_stdout(io.StringIO()):
            self.assertEqual(judge.score(self.example, "blue"), 1)
        self.assertEqual(request.call_count, 3)
        self.assertEqual([c.args[5] for c in request.call_args_list], [3, 3, 3])
        self.assertEqual(sleep.call_args_list, [call(30.0), call(60.0)])
        recovery = [event for event in self.events() if event["event"] == "remote_judge_network_recovery"]
        self.assertEqual([event["recovery_round"] for event in recovery], [1, 2])
        self.assertEqual(recovery[0]["transport_error_type"], "gaierror")
        self.assertEqual(recovery[0]["sample_id"], "q0")

    def test_network_recovery_is_bounded_and_failure_never_becomes_wrong(self):
        judge = self.judge()
        with patch("agmem.remote_judge._complete", side_effect=failed_transport) as request, \
             patch("agmem.remote_judge.sleep") as sleep, redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(InfrastructureError, "q0/reference/0.*gaierror") as caught:
                judge.score(self.example, "blue")
        self.assertEqual(request.call_count, 7)
        self.assertEqual(sleep.call_args_list, [call(30.0)] + [call(60.0)] * 5)
        self.assertFalse((self.root / "diagnostics/calls").exists())
        diagnostics = output.getvalue() + str(caught.exception)
        diagnostics += (self.root / "diagnostics/events.jsonl").read_text()
        self.assertNotIn("secret-do-not-persist", diagnostics)
        self.assertNotIn("private.invalid", diagnostics)

    def test_http_certificate_label_and_length_failures_do_not_enter_network_recovery(self):
        judge = self.judge()
        failures = [
            {"error_type": "HTTPError", "transport_type": "str", "status_code": code}
            for code in (401, 403, 429, 500)
        ] + [
            {"transport_type": "SSLCertVerificationError"},
            {"transport_type": "CertificateError"},
            {"error_type": "ValueError", "transport_type": None,
             "reason": "empty_text", "status_code": 200},
            {"error_type": "ValueError", "transport_type": None, "reason": "invalid_usage"},
            {"transport_type": "str"},
        ]
        for fields in failures:
            with self.subTest(fields=fields):
                with patch("agmem.remote_judge._complete", side_effect=lambda *args: failed_transport(
                        *args, **fields)) as request, patch("agmem.remote_judge.sleep") as sleep, \
                     redirect_stdout(io.StringIO()):
                    with self.assertRaises(InfrastructureError):
                        judge.score(self.example, "blue")
                self.assertEqual(request.call_count, 1)
                sleep.assert_not_called()
        self.assertFalse((self.root / "diagnostics/calls").exists())

    def test_network_recovery_preserves_expanded_completion_budget(self):
        judge = self.judge()
        calls = 0

        def remote(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                failed_transport(*args, transport_type="SSLError", budget=8192)
            return completion(budget=8192)

        with patch("agmem.remote_judge._complete", side_effect=remote) as request, \
             patch("agmem.remote_judge.sleep"), redirect_stdout(io.StringIO()):
            self.assertEqual(judge.score(self.example, "blue"), 1)
        self.assertEqual([c.args[1]["max_output_tokens"] for c in request.call_args_list], [2048, 8192])
        self.assertEqual(self.events()[-1]["max_tokens"], 8192)

    def test_network_options_are_runtime_only_and_existing_cache_is_reused(self):
        original = self.judge(network_retries=0)
        with patch("agmem.remote_judge._complete", return_value=completion()):
            self.assertEqual(original.score(self.example, "blue"), 1)
        manifest_path = self.root / "diagnostics/manifest.json"
        original_manifest = manifest_path.read_bytes()
        resumed = self.judge(network_retries=4, network_retry_delay=12)
        self.assertEqual(original.provenance(), resumed.provenance())
        self.assertEqual(original_manifest, manifest_path.read_bytes())
        with patch("agmem.remote_judge._complete") as request, patch("agmem.remote_judge.sleep") as sleep:
            self.assertEqual(resumed.score(self.example, "blue"), 1)
        request.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(self.events()[-1]["network_retries"], 4)
        self.assertEqual(self.events()[-1]["network_retry_delay_seconds"], 12)

    def test_network_recovery_can_be_disabled_and_interrupts_propagate(self):
        judge = self.judge(network_retries=0)
        with patch("agmem.remote_judge._complete", side_effect=failed_transport) as request, \
             patch("agmem.remote_judge.sleep") as sleep, redirect_stdout(io.StringIO()):
            with self.assertRaises(InfrastructureError):
                judge.score(self.example, "blue")
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()
        judge = self.judge()
        with patch("agmem.remote_judge._complete", side_effect=failed_transport), \
             patch("agmem.remote_judge.sleep", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                judge.score(self.example, "blue")
        with patch("agmem.remote_judge._complete", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                judge.score(self.example, "blue")

    def test_network_failure_classification_uses_safe_metadata(self):
        judge = self.judge()
        cases = [
            {"error_type": "TimeoutError", "transport_type": None},
            {"transport_type": "SSLError"},
            {"transport_type": "SSLEOFError"},
            {"transport_type": "ConnectionResetError"},
            {"transport_type": "OSError", "number": errno.ENETUNREACH},
        ]
        for index, fields in enumerate(cases):
            with self.subTest(fields=fields):
                calls = 0

                def remote(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        failed_transport(*args, **fields)
                    return completion()

                with patch("agmem.remote_judge._complete", side_effect=remote), \
                     patch("agmem.remote_judge.sleep") as sleep, redirect_stdout(io.StringIO()):
                    self.assertEqual(judge.score(self.example, f"prediction {index}"), 1)
                self.assertEqual(calls, 2)
                sleep.assert_called_once_with(30.0)

    def test_network_recovery_settings_are_validated(self):
        for kwargs in ({"network_retries": -1}, {"network_retries": True}, {"network_retries": 1.5},
                       {"network_retry_delay": 0}, {"network_retry_delay": -1},
                       {"network_retry_delay": float("nan")}, {"network_retry_delay": float("inf")},
                       {"network_retry_delay": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.judge(**kwargs)


if __name__ == "__main__":
    unittest.main()
