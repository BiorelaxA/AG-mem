"""Judge exported Task answers without regenerating evidence or using a GPU/API."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agmem.core import Example, Trajectory, atomic_json, digest
from agmem.evaluate_checkpoints import EpisodeJournal
from agmem.evaluation_offline import export_task_requests
from agmem.judge_gpt_responses import run
from agmem.judges import load_judge_templates
from agmem.prompts import task_input_contract, task_messages
from agmem.remote_task_runner import load_bundle, response_manifest


class GPTJudgeResponsesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / "results"
        self.prepared = self.root / "prepared"
        template = Path(__file__).resolve().parents[1] / "configs/protocols/format05_training_judge.json"
        self.config = {
            "reader": {"prompt_version": "evidence_loop_v3"},
            "retrieval": {"max_ids_per_call": 1, "max_read_calls": 8, "id_protocol": "strict_v2"},
            "task_agent": {"max_output_tokens": 512},
            "reward": {"formula": "judge_cost_v2", "read_call_cost": .01,
                       "format_penalty": .5, "format_penalty_mode": "per_id",
                       "likelihood": {"mode": "off"}, "judge": {"template_file": str(template)}},
        }
        _, _, self.hashes = load_judge_templates(self.config["reward"]["judge"])
        self.config["reward"]["judge"]["expected_template_sha256"] = deepcopy(self.hashes)
        self.saved = {"scoring_protocol": {"judge": {"template_sha256": deepcopy(self.hashes)}}}
        self.remote = {
            "model": "gpt-5-mini", "model_revision": "gpt-5-mini", "request_profile": "openai_chat",
            "send_seed": False, "temperature": None, "max_tokens_parameter": "max_completion_tokens",
            "api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL",
        }
        self.suite = {
            "output_root": str(self.original), "prepared_root": str(self.prepared),
            "datasets": ["locomo"], "checkpoints": {"best": "checkpoint20", "final": "checkpoint32"},
            "remote_task": self.remote,
        }
        self.examples = [Example(str(i), "locomo", "test", "history", f"question {i}",
                                 gold_answers=["blue"]) for i in range(2)]
        self.example_file = self.prepared / "locomo/test_examples.jsonl"
        self.example_file.parent.mkdir(parents=True)
        self.example_file.write_text("".join(json.dumps(asdict(e)) + "\n" for e in self.examples))
        for label in self.suite["checkpoints"]:
            self.create_source(label)
        self.calls = []
        self.instances = []

    def create_source(self, label):
        target = self.original / label / "locomo"
        sample_ids = [e.sample_id for e in self.examples]
        retrieval = EpisodeJournal(target / "retrieval", {
            "protocol": "fixture", "sample_ids": sample_ids,
            "examples_sha256": digest([asdict(e) for e in self.examples]),
        })
        local = EpisodeJournal(target / "answers/local", {
            "protocol": "fixture", "sample_ids": sample_ids,
            "retrieval_manifest_sha256": retrieval.signature,
            "task_input_contract": task_input_contract(), "task_max_output_tokens": 512,
        })
        for example in self.examples:
            evidence = [{"id": "session1", "raw_text": f"blue unicode 中文 {example.sample_id}"}]
            row = Trajectory(example.sample_id, 42, example.memory_id, evidence=evidence,
                             n_read_calls=1, termination_reason="stop")
            retrieval.write(row)
            row.prediction = "QWEN_LOCAL_ANSWER"
            row.task_input_hash = digest(task_messages(example, evidence))
            local.write(row)
        retrieval.finish()
        local.finish()
        with patch("agmem.pipeline.announce"):
            bundle_path = export_task_requests(self.config, self.examples, retrieval, target,
                                               self.remote, label, "locomo")
        bundle = load_bundle(bundle_path)
        response_root = target / "offline/responses"
        manifest = response_manifest(bundle, "https://provider.example/v1")
        atomic_json(response_root / "manifest.json", manifest)
        for i, request in enumerate(bundle["requests"]):
            atomic_json(response_root / "episodes" / f"{i:08d}.json", {
                "manifest_sha256": digest(manifest), "sample_id": request["sample_id"],
                "task_input_hash": request["task_input_hash"],
                "completion": {"protocol": "score_nonempty_truncated_v1", "text": "blue" if i == 0 else "red",
                               "finish_reason": "stop", "truncated": False,
                               "usage": {"completion_tokens": 1}, "max_tokens": 512},
            })
        atomic_json(response_root / "complete.json", {"manifest_sha256": digest(manifest), "n": 2})
        # Existing Qwen Judge output must never be replaced by the GPT Judge run.
        atomic_json(target / "answers/remote/existing_qwen_result.json", {"judge": "qwen", "score": 1})

    def args(self, **overrides):
        values = dict(config="fixture.yaml", datasets=None, checkpoints=None, responses_root=None,
                      output_root=None, workers=2, timeout=120, retries=3, check_only=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def fake_judge(self, remote, config, diagnostics_root, **execution):
        self.assertEqual(remote["model"], "gpt-5-mini")
        self.assertEqual(config, self.config["reward"]["judge"])
        self.assertEqual(load_judge_templates(config)[2], self.hashes)
        def score(example, prediction):
            self.calls.append((example.sample_id, prediction))
            return float(prediction == "blue")
        judge = SimpleNamespace(
            provenance=lambda: {"model": "gpt-5-mini", "template_sha256": deepcopy(self.hashes)},
            score=score, close=Mock(), execution=execution)
        self.instances.append(judge)
        return judge

    def invoke(self, **overrides):
        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {}, clear=True))
            stack.enter_context(patch("agmem.judge_gpt_responses.load_suite",
                                      return_value=(self.suite, self.config, self.saved)))
            stack.enter_context(patch("agmem.remote_judge.RemoteJudgeScorer", side_effect=self.fake_judge))
            for name in ("make_task_agent", "make_reward_scorer", "checkpoint_model"):
                stack.enter_context(patch("agmem.evaluate_checkpoints." + name,
                                          side_effect=AssertionError("No model/Task/Reader service allowed")))
            stack.enter_context(patch("agmem.judge_gpt_responses.announce"))
            stack.enter_context(patch("agmem.evaluate_checkpoints.announce"))
            return run(self.args(**overrides))

    def snapshot_original(self):
        return {str(p.relative_to(self.original)): p.read_bytes()
                for p in self.original.rglob("*") if p.is_file()}

    def test_two_checkpoints_use_saved_predictions_separate_outputs_and_resume(self):
        original = self.snapshot_original()
        output = self.invoke()
        self.assertEqual(output, self.root / "gpt_judge_results")
        self.assertCountEqual(self.calls, [("0", "blue"), ("1", "red")] * 2)
        for label in ("best", "final"):
            target = output / label / "locomo"
            metrics = json.loads((target / "answers/remote/metrics.json").read_text())
            self.assertEqual(metrics["pooled"]["n"], 2)
            self.assertEqual(metrics["pooled"]["judge_accuracy"], .5)
            self.assertAlmostEqual(metrics["pooled"]["reward"], -.01)
            source = json.loads((target / "source_manifest.json").read_text())
            self.assertEqual(source["training_judge_template_sha256"], self.hashes)
            journal = EpisodeJournal(target / "answers/remote", json.loads(
                (target / "answers/remote/manifest.json").read_text()))
            self.assertEqual(journal.read("0")["prediction"], "blue")
            self.assertEqual(journal.read("1")["prediction"], "red")
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual({r["checkpoint"] for r in summary["results"]}, {"best", "final"})
        self.calls.clear()
        self.assertEqual(self.invoke(), output)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.snapshot_original(), original)
        self.assertTrue(all(j.close.called for j in self.instances))

    def test_check_only_verifies_complete_answers_and_exact_prompt_without_credentials(self):
        items = self.invoke(check_only=True)
        self.assertEqual([item["label"] for item in items], ["best", "final"])
        for item in items:
            self.assertEqual((item["n"], item["empty"], item["truncated"]), (2, 0, 0))
            self.assertEqual(item["finish_reasons"], {"stop": 2})
        self.assertEqual(self.instances, [])
        self.assertFalse((self.root / "gpt_judge_results").exists())
        self.assertEqual(self.hashes["locomo_refined"],
                         "fd44ba60ff07d28f4b44057a96a80cfbb45e2a7973dab7aae8051c7aabad06ea")

    def test_network_recovery_settings_reach_judge_and_allow_resume(self):
        output = self.invoke(network_retries=2, network_retry_delay=10)
        self.assertEqual(self.instances[-1].execution,
                         {"network_retries": 2, "network_retry_delay": 10})
        source = (output / "best/locomo/source_manifest.json").read_bytes()
        self.calls.clear()
        self.invoke(network_retries=8, network_retry_delay=45)
        self.assertEqual(self.instances[-1].execution,
                         {"network_retries": 8, "network_retry_delay": 45})
        self.assertEqual(self.calls, [])
        self.assertEqual((output / "best/locomo/source_manifest.json").read_bytes(), source)

    def test_invalid_network_settings_fail_before_judge(self):
        for setting in ({"network_retries": -1}, {"network_retries": True},
                        {"network_retry_delay": 0}, {"network_retry_delay": float("nan")}):
            with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, "network-"):
                self.invoke(**setting)
        self.assertEqual(self.instances, [])

    def test_invalid_answers_in_later_checkpoint_fail_before_any_paid_call(self):
        path = self.original / "final/locomo/offline/responses/episodes/00000001.json"
        original = path.read_bytes()
        for failure in ("missing", "foreign_hash", "empty"):
            with self.subTest(failure=failure):
                row = json.loads(original)
                if failure == "missing":
                    path.unlink()
                else:
                    if failure == "foreign_hash":
                        row["task_input_hash"] = "foreign"
                    else:
                        row["completion"]["text"] = ""
                    atomic_json(path, row)
                with self.assertRaises(ValueError):
                    self.invoke()
                self.assertEqual(self.instances, [])
                path.write_bytes(original)

    def test_modified_question_or_retrieval_is_rejected_before_judging(self):
        original = self.example_file.read_text()
        changed = [asdict(e) for e in self.examples]
        changed[0]["gold_answers"] = ["changed reference"]
        self.example_file.write_text("".join(json.dumps(e) + "\n" for e in changed))
        with self.assertRaisesRegex(ValueError, "questions/gold answers"):
            self.invoke()
        self.example_file.write_text(original)
        path = self.original / "final/locomo/retrieval/episodes/00000001.json.gz"
        with gzip.open(path, "rt") as stream:
            row = json.load(stream)
        row["trajectory"]["evidence"][0]["raw_text"] = "different evidence"
        with gzip.open(path, "wt") as stream:
            json.dump(row, stream)
        with self.assertRaisesRegex(ValueError, "evidence does not match"):
            self.invoke()
        self.assertEqual(self.instances, [])

    def test_output_must_not_overlap_source_results(self):
        for path in (self.original, self.original / "gpt_judge", self.root):
            with self.subTest(path=str(path)), self.assertRaisesRegex(ValueError, "separate"):
                self.invoke(output_root=str(path))
        self.assertEqual(self.instances, [])

    def test_prompt_mismatch_and_changed_resumed_inputs_are_rejected(self):
        self.saved["scoring_protocol"]["judge"]["template_sha256"] = {"locomo_refined": "different"}
        with self.assertRaisesRegex(ValueError, "saved training prompt"):
            self.invoke()
        self.assertEqual(self.instances, [])
        self.saved["scoring_protocol"]["judge"]["template_sha256"] = deepcopy(self.hashes)
        self.invoke(checkpoints=["best"])
        self.calls.clear()
        path = self.original / "best/locomo/offline/responses/episodes/00000001.json"
        row = json.loads(path.read_text())
        row["completion"]["text"] = "modified saved answer"
        atomic_json(path, row)
        with self.assertRaisesRegex(ValueError, "inputs/protocol changed"):
            self.invoke(checkpoints=["best"])
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
