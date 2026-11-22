"""Portable history-chat exports and complete tagged answers, without API calls."""
from contextlib import redirect_stdout
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
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from agmem.core import Example
from agmem.prompts import HISTORY_ANSWER_TEMPLATE, task_input_contract, task_messages
from agmem import remote_task_runner as runner
from tests.test_remote_task_runner import fixture_bundle


def tagged_bundle(count=2):
    bundle = fixture_bundle(count)
    requests = []
    for index in range(count):
        example = Example(f"q{index}", "locomo", "test", "m", f"Question {index}",
                          gold_answers=["SECRET_GOLD"])
        messages = task_messages(example, [{"time": "25 May 2022", "raw_text": "Alice: I adopted a dog."}],
                                 "history_answer_v1")
        requests.append({"sample_id": example.sample_id, "messages": messages,
                         "task_input_hash": runner.digest(messages)})
    bundle["manifest"].update(protocol=runner.TAGGED_REQUEST_PROTOCOL, source_kind="retrieval_reprompt",
                              source_task_manifest_sha256=None,
                              task_input_contract=task_input_contract("history_answer_v1"),
                              requests_sha256=runner.digest(requests))
    bundle["requests"] = requests
    return bundle


class TaggedRemoteTaskTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = tagged_bundle(1)
        self.requests = self.root / "requests.json.gz"
        with gzip.open(self.requests, "wt", encoding="utf-8") as stream:
            json.dump(self.bundle, stream)
        self.output = self.root / "responses"
        env = patch.dict(os.environ, {"OPENAI_BASE_URL": "https://provider.invalid/v1",
                                      "OPENAI_API_KEY": "secret-never-persist"})
        env.start()
        self.addCleanup(env.stop)

    @staticmethod
    def response(text="<answer>25 May 2022</answer>", finish="stop"):
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"completion_tokens": 32}}

    def opener(self, replies):
        replies = iter(replies)
        payloads = []
        def request(req, timeout):
            payloads.append(json.loads(req.data))
            result = next(replies)
            if isinstance(result, Exception):
                raise result
            return io.BytesIO(json.dumps(result).encode())
        opener = Mock()
        opener.open.side_effect = request
        return opener, payloads

    def run_replies(self, replies, **kwargs):
        opener, payloads = self.opener(replies)
        with patch.object(runner, "build_opener", return_value=opener), \
             patch.object(runner.time, "sleep"), redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output, workers=1, **kwargs)
        return payloads

    def test_single_user_template_contract_and_no_gold_are_exact(self):
        self.assertEqual(runner.HISTORY_ANSWER_TEMPLATE, HISTORY_ANSWER_TEMPLATE)
        self.assertEqual(runner._tagged_contract(), task_input_contract("history_answer_v1"))
        self.assertEqual(runner.load_bundle(self.requests), self.bundle)
        messages = self.bundle["requests"][0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user"])
        self.assertIn("History Chats: Session 1\nDate: 25 May 2022\nAlice:", messages[0]["content"])
        self.assertNotIn("SECRET_GOLD", json.dumps(self.bundle))

    def test_altered_contract_template_source_or_hash_is_rejected(self):
        changes = [lambda b: b["manifest"]["task_input_contract"].update(task_prompt_version="unknown"),
                   lambda b: b["manifest"]["task_input_contract"].update(task_prompt_sha256="a" * 64),
                   lambda b: b["manifest"]["task_input_contract"].update(answer_parser="loose"),
                   lambda b: b["manifest"].update(source_task_manifest_sha256="b" * 64),
                   lambda b: b["manifest"].update(source_kind="old_answers"),
                   lambda b: b["requests"][0]["messages"][0].update(role="system"),
                   lambda b: b["requests"][0]["messages"][0].update(content="Please answer."),
                   lambda b: b["requests"][0].update(task_input_hash="c" * 64)]
        for change in changes:
            changed = deepcopy(self.bundle)
            change(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_bundle(changed)
        changed = deepcopy(self.bundle)
        changed["manifest"]["protocol"] = runner.REQUEST_PROTOCOL
        changed["manifest"]["source_task_manifest_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "prompt version"):
            runner.validate_bundle(changed)

    def test_parser_accepts_only_one_nonempty_complete_block(self):
        self.assertEqual(runner.extract_tagged_answer("Thought\n<answer>  25 May\n2022 </answer>\nDone"),
                         "25 May\n2022")
        for text in ("plain", "<answer></answer>", "<answer>  </answer>", "<answer>blue",
                     "blue</answer>", "<answer>a</answer><answer>b</answer>",
                     "<answer><answer>b</answer></answer>", "<answer>a</answer><answer",
                     "<ANSWER>a</ANSWER>", "<answer score='1'>blue</answer>", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.extract_tagged_answer(text)

    def test_complete_response_is_saved_raw_and_resume_does_not_call_provider(self):
        raw = "Thinking briefly.\n<answer>25 May 2022</answer>"
        payloads = self.run_replies([self.response(raw)], empty_answer_max_tokens=16384)
        self.assertEqual(payloads[0]["messages"], self.bundle["requests"][0]["messages"])
        manifest, rows = runner.load_response_records(self.bundle, self.output)
        self.assertEqual(manifest["protocol"], runner.TAGGED_RESPONSE_PROTOCOL)
        self.assertNotIn("legacy_manifest_sha256", manifest)
        options = manifest["task"]["request_options"]
        self.assertEqual(options["answer_parser"], "answer_tag_v1")
        self.assertEqual(options["answer_completion_policy"]["trigger"], "finish_reason_length")
        self.assertEqual(rows["q0"]["completion"]["text"], raw)
        self.assertFalse(rows["q0"]["completion"]["truncated"])
        before = (self.output / "episodes/00000000.json").read_bytes()
        with patch.object(runner, "_complete", side_effect=AssertionError("must use saved answers")), \
             redirect_stdout(io.StringIO()):
            runner.run(self.requests, self.output)
        self.assertEqual((self.output / "episodes/00000000.json").read_bytes(), before)
        for path in self.output.rglob("*.json*"):
            self.assertNotIn("secret-never-persist", path.read_text())

    def test_nonempty_length_and_empty_length_grow_until_complete(self):
        payloads = self.run_replies([self.response("<answer>25 May 2022</answer>", "length"),
                                    self.response(None, "length"), self.response()],
                                   retries=0, empty_answer_max_tokens=16384)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512, 2048, 8192])
        _, rows = runner.load_response_records(self.bundle, self.output)
        self.assertEqual(rows["q0"]["completion"]["max_tokens"], 8192)
        self.assertEqual(rows["q0"]["completion"]["finish_reason"], "stop")

    def test_transport_retries_preserve_chance_to_expand_tagged_answer(self):
        payloads = self.run_replies([URLError("network"), self.response("<answer>partial", "length"),
                                    URLError("network"), self.response()],
                                   retries=1, empty_answer_max_tokens=16384)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512, 512, 2048, 2048])

    def test_invalid_tags_retry_without_growing_budget(self):
        payloads = self.run_replies([self.response("no tags"), self.response()], retries=1,
                                   empty_answer_max_tokens=16384)
        self.assertEqual([p["max_completion_tokens"] for p in payloads], [512, 512])
        events = [json.loads(s) for s in (self.output / "events.jsonl").read_text().splitlines()]
        failed = [e for e in events if e["event"] == "remote_task_attempt_failed"]
        self.assertEqual(failed[0]["reason"], "invalid_answer_tags")

    def test_exhausted_tags_or_length_never_publish_an_answer(self):
        for text, finish, reason in (("plain", "stop", "invalid_answer_tags"),
                                     ("<answer>blue</answer>", "length", "truncated_answer"),
                                     ("<answer>blue</answer>", "content_filter", "incomplete_answer")):
            with self.subTest(finish=finish):
                if self.output.exists():
                    shutil.rmtree(self.output)
                opener, payloads = self.opener([self.response(text, finish)] * 2)
                with patch.object(runner, "build_opener", return_value=opener), \
                     patch.object(runner.time, "sleep"), redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, reason):
                        runner.run(self.requests, self.output, workers=1, retries=1, empty_answer_max_tokens=512)
                self.assertEqual(len(payloads), 2)
                self.assertFalse((self.output / "complete.json").exists())
                self.assertEqual(list((self.output / "episodes").glob("*.json")), [])

    def test_import_rejects_truncated_or_malformed_saved_answer(self):
        self.run_replies([self.response()], empty_answer_max_tokens=16384)
        path = self.output / "episodes/00000000.json"
        original = json.loads(path.read_text())
        for changes in ({"finish_reason": "length", "truncated": True}, {"text": "plain"},
                        {"max_tokens": 1024}):
            changed = deepcopy(original)
            changed["completion"].update(changes)
            runner._atomic_json(path, changed)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.load_response_records(self.bundle, self.output)
        runner._atomic_json(path, original)
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["task"]["request_options"]["answer_completion_policy"]["require_finish_reason"] = "length"
        runner._atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "protocol"):
            runner.load_response_records(self.bundle, self.output)

    def test_legacy_answers_cannot_be_reused_for_new_prompt(self):
        legacy = fixture_bundle(1)
        runner._atomic_json(self.output / "manifest.json", runner.response_manifest(legacy, "https://provider.invalid/v1"))
        with patch.object(runner, "_complete", side_effect=AssertionError("No API")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                runner.run(self.requests, self.output)

    def test_tagged_runner_remains_standalone(self):
        copied = self.root / "standalone_runner.py"
        shutil.copyfile(runner.__file__, copied)
        result = subprocess.run([sys.executable, "-I", "-c",
                                 "import runpy,sys; m=runpy.run_path(sys.argv[1],run_name='runner'); "
                                 "m['load_bundle'](sys.argv[2])", str(copied), str(self.requests)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
