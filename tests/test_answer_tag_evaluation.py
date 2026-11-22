"""Re-answer frozen evidence with tagged answers, preserving legacy artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import yaml

from agmem.core import atomic_json, digest
from agmem.evaluate_checkpoints import EpisodeJournal, load_suite
from agmem.evaluation_offline import export_task_requests
from agmem.judge_gpt_responses import load_inputs
from agmem.prompts import configured_reader_input_contract, task_input_contract
from agmem.remote_task_runner import load_bundle, response_manifest
from tests import test_gpt_judge_responses as legacy_fixture


class AnswerTagEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy_fixture.GPTJudgeResponsesTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.legacy_suite, self.legacy_config = deepcopy(f.suite), deepcopy(f.config)
        self.new_root = f.root / "answer_tags/results"
        f.suite.update(task_prompt_version="history_answer_v1",
                       retrieval_reuse_root=str(f.original), output_root=str(self.new_root))
        f.config["task_agent"].update(prompt_version="history_answer_v1", max_output_tokens=2048)
        self.raw_answers = {"0": "Explanation outside the answer.\n<answer>blue</answer>",
                            "1": "<answer> red </answer>"}
        for label in ("best", "final"):
            source = f.original / label / "locomo/retrieval"
            journal = EpisodeJournal(source, json.loads((source / "manifest.json").read_text()))
            target = self.new_root / label / "locomo"
            with patch("agmem.pipeline.announce"):
                bundle_path = export_task_requests(f.config, f.examples, journal, target,
                                                   f.remote, label, "locomo")
            bundle = load_bundle(bundle_path)
            manifest = response_manifest(bundle, "https://provider.example/v1", 16384)
            self.assertEqual(bundle["manifest"]["protocol"], "checkpoint_task_requests_v2")
            self.assertEqual(manifest["protocol"], "checkpoint_task_responses_v3")
            responses = target / "offline/responses"
            atomic_json(responses / "manifest.json", manifest)
            for index, request in enumerate(bundle["requests"]):
                atomic_json(responses / "episodes" / f"{index:08d}.json", {
                    "manifest_sha256": digest(manifest), "sample_id": request["sample_id"],
                    "task_input_hash": request["task_input_hash"],
                    "completion": {"protocol": "score_nonempty_truncated_v1",
                                   "text": self.raw_answers[request["sample_id"]],
                                   "finish_reason": "stop", "truncated": False,
                                   "usage": {"completion_tokens": 12}, "max_tokens": 2048},
                })
            atomic_json(responses / "complete.json", {"manifest_sha256": digest(manifest), "n": 2})

    def test_reprompt_without_new_qwen_answers_judges_inner_text_and_preserves_raw_output(self):
        f = self.fixture
        original = f.snapshot_original()
        self.assertFalse((self.new_root / "best/locomo/answers/local").exists())
        output = f.invoke()
        self.assertEqual(output, f.root / "answer_tags/gpt_judge_results")
        self.assertCountEqual(f.calls, [("0", "blue"), ("1", "red")] * 2)
        for label in ("best", "final"):
            target = output / label / "locomo/answers/remote"
            manifest = json.loads((target / "manifest.json").read_text())
            report = json.loads((target / "metrics.json").read_text())
            self.assertEqual(manifest["task_input_contract"], task_input_contract("history_answer_v1"))
            self.assertEqual(report["task_input_contract"], manifest["task_input_contract"])
            self.assertEqual(report["pooled"]["n"], 2)
            self.assertEqual(report["pooled"]["judge_accuracy"], .5)
            self.assertEqual(report["pooled"]["answer_f1"], .5)
            options = manifest["task"]["request_options"]
            self.assertEqual(options["answer_parser"], "answer_tag_v1")
            self.assertEqual(options["answer_completion_policy"]["require_finish_reason"], "stop")
            self.assertEqual(options["answer_completion_policy"]["max_completion_tokens"], 16384)
            journal = EpisodeJournal(target, manifest)
            for sid, answer in (("0", "blue"), ("1", "red")):
                row = journal.read(sid)
                self.assertEqual(row["prediction"], answer)
                self.assertEqual(row["task_completion"]["raw_text"], self.raw_answers[sid])
                self.assertEqual(row["task_completion"]["answer_parser"], "answer_tag_v1")
            self.assertTrue((target / "complete.json").exists())
        f.calls.clear()
        self.assertEqual(f.invoke(), output)
        self.assertEqual(f.calls, [])
        self.assertEqual(f.snapshot_original(), original)

    def test_check_only_needs_no_credentials_or_local_task_service(self):
        f = self.fixture
        original = f.snapshot_original()
        checked = f.invoke(check_only=True)
        self.assertEqual([(r["label"], r["n"]) for r in checked], [("best", 2), ("final", 2)])
        self.assertEqual(f.instances, [])
        self.assertEqual(f.snapshot_original(), original)
        self.assertFalse((f.root / "answer_tags/gpt_judge_results").exists())

    def test_legacy_and_tagged_bundles_cannot_be_interchanged(self):
        f = self.fixture
        old = f.original / "best/locomo/offline/requests.json.gz"
        new = self.new_root / "best/locomo/offline/requests.json.gz"
        old_bytes, new_bytes = old.read_bytes(), new.read_bytes()
        new.write_bytes(old_bytes)
        with self.assertRaisesRegex(ValueError, "Task prompt"):
            f.invoke(check_only=True)
        new.write_bytes(new_bytes)
        old.write_bytes(new_bytes)
        with self.assertRaisesRegex(ValueError, "Task prompt"):
            load_inputs(self.legacy_suite, "best", "locomo")
        old.write_bytes(old_bytes)
        self.assertEqual(f.instances, [])

    def test_old_response_records_and_bad_tagged_answers_fail_before_judge(self):
        f = self.fixture
        imported = f.root / "imported_legacy"
        for label in ("best", "final"):
            shutil.copytree(f.original / label / "locomo/offline/responses", imported / label / "locomo")
        with self.assertRaises(ValueError):
            f.invoke(responses_root=str(imported))
        path = self.new_root / "final/locomo/offline/responses/episodes/00000000.json"
        original = json.loads(path.read_text())
        for change in ("missing_tags", "length"):
            row = deepcopy(original)
            if change == "missing_tags":
                row["completion"]["text"] = "blue"
            else:
                row["completion"].update(finish_reason="length", truncated=True)
            atomic_json(path, row)
            with self.subTest(change=change), self.assertRaises(ValueError):
                f.invoke()
        self.assertEqual(f.instances, [])

    def test_judge_output_cannot_overlap_either_original_or_new_answers(self):
        f = self.fixture
        for root in (f.original, f.original / "judge", f.root,
                     self.new_root, self.new_root / "judge", self.new_root.parent):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "separate"):
                f.invoke(output_root=str(root))
        self.assertEqual(f.instances, [])

    def saved_suite(self):
        f = self.fixture
        config = deepcopy(self.legacy_config)
        config.update(data={"input_root": "unchanged-training-input"}, service={"base_url": "http://training/v1"})
        config["task_agent"]["service"] = {"model": "training-service-override"}
        saved = {"config": config, "reader_input_contract": configured_reader_input_contract(config),
                 "task_input_contract": task_input_contract(), "scoring_protocol": deepcopy(f.saved["scoring_protocol"])}
        run = f.root / "training"
        atomic_json(run / "run.json", saved)
        suite = {**deepcopy(f.suite), "training_run": str(run), "input_root": str(f.root / "input"),
                 "local_base_url": "http://127.0.0.1:8000/v1", "task_max_output_tokens": 2048,
                 "judge_template_file": config["reward"]["judge"]["template_file"]}
        path = f.root / "new_suite.yaml"
        path.write_text(yaml.safe_dump(suite))
        return path, suite, saved

    def test_load_suite_explicit_prompt_override_preserves_saved_training_configuration(self):
        path, _, saved = self.saved_suite()
        _, current, observed_saved = load_suite(path)
        self.assertEqual(current["task_agent"]["prompt_version"], "history_answer_v1")
        self.assertEqual(current["task_agent"]["max_output_tokens"], 2048)
        self.assertNotIn("service", current["task_agent"])
        self.assertEqual(observed_saved, saved)
        self.assertNotIn("prompt_version", observed_saved["config"]["task_agent"])
        self.assertEqual(observed_saved["config"]["task_agent"]["max_output_tokens"], 512)
        self.assertEqual(current["reward"]["judge"]["expected_template_sha256"],
                         self.fixture.hashes)

    def test_load_suite_rejects_unknown_prompt_or_overlapping_outputs(self):
        path, suite, _ = self.saved_suite()
        for version in ("history_answer_typo", "", None):
            changed = {**suite, "task_prompt_version": version}
            path.write_text(yaml.safe_dump(changed))
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "prompt_version"):
                load_suite(path)
        source = Path(suite["retrieval_reuse_root"])
        for output in (source, source / "new", source.parent):
            path.write_text(yaml.safe_dump({**suite, "output_root": str(output)}))
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                load_suite(path)


if __name__ == "__main__":
    unittest.main()
