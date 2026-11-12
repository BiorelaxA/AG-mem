"""Export fresh tagged Task requests from immutable completed Reader evidence."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agmem.core import Example, Trajectory, atomic_json, digest
from agmem.evaluate_checkpoints import EpisodeJournal
from agmem.evaluation_offline import SavedTaskAnswers, export_task_requests, judge_remote_answers
from agmem.prompts import task_input_contract, task_messages
from agmem.remote_task_runner import COMPLETION_PROTOCOL, load_bundle, response_manifest
from agmem.rewards import score_trajectory


class RepromptExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "old_results/best/locomo"
        self.target = self.root / "new_results/best/locomo"
        self.examples = [Example(f"locomo/test/{i}", "locomo", "test", "history", f"question {i}",
                                 gold_answers=["HIDDEN_GOLD"], judge_metadata={"private": "SCORING_ONLY"})
                         for i in range(2)]
        self.config = {
            "reader": {"prompt_version": "evidence_loop_v3"},
            "retrieval": {"max_ids_per_call": 1, "max_read_calls": 8, "id_protocol": "strict_v2"},
            "task_agent": {"prompt_version": "history_answer_v1", "max_output_tokens": 2048},
            "reward": {"formula": "judge_cost_v2", "read_call_cost": .01, "format_penalty": .5,
                       "format_penalty_mode": "per_id", "text_outcome": "qa_f1_v1", "choice_outcome": "label_f1"},
        }
        self.remote = {"model": "gpt-5-mini", "model_revision": "gpt-5-mini", "request_profile": "openai_chat",
                       "send_seed": False, "temperature": None, "max_tokens_parameter": "max_completion_tokens"}
        self.retrieval = EpisodeJournal(self.source / "retrieval", {
            "protocol": "fixture", "sample_ids": [e.sample_id for e in self.examples],
            "examples_sha256": digest([asdict(e) for e in self.examples])})
        for example in self.examples:
            self.retrieval.write(Trajectory(example.sample_id, 42, example.memory_id,
                n_read_calls=1, termination_reason="stop", evidence=[{
                    "id": "r_1", "time": "2023-01-02", "raw_text": "The original chat says blue.",
                    "content": "DIRECTORY_SUMMARY_NOT_REQUIRED"}]))
        self.retrieval.finish()
        self.calls = []
        self.scorer = SimpleNamespace(provenance=lambda: {"judge": "frozen-training"}, score=self.score,
                                      close=lambda: None)

    def score(self, example, row):
        self.calls.append((example.sample_id, row.prediction))
        judge = SimpleNamespace(score=lambda e, prediction: float(prediction == "blue"))
        return score_trajectory(example, row, self.config["reward"], judge=judge, max_ids=1)

    def export(self, **overrides):
        values = dict(config=self.config, examples=self.examples, retrieval=self.retrieval,
                      target=self.target, remote=self.remote, checkpoint="best", dataset="locomo")
        values.update(overrides)
        with patch("agmem.pipeline.announce"):
            return export_task_requests(**values)

    def responses(self, bundle_path):
        bundle = load_bundle(bundle_path)
        output = self.root / "returned"
        manifest = response_manifest(bundle, "https://provider.invalid/v1", 16384)
        atomic_json(output / "manifest.json", manifest)
        for i, request in enumerate(bundle["requests"]):
            atomic_json(output / "episodes" / f"{i:08d}.json", {
                "manifest_sha256": digest(manifest), "sample_id": request["sample_id"],
                "task_input_hash": request["task_input_hash"],
                "completion": {"protocol": COMPLETION_PROTOCOL,
                    "text": "An explanation before the final answer.\n<answer> blue </answer>",
                    "finish_reason": "stop", "truncated": False,
                    "usage": {"completion_tokens": 20}, "max_tokens": 2048}})
        atomic_json(output / "complete.json", {"manifest_sha256": digest(manifest), "n": 2})
        return output

    def judge(self, bundle, responses):
        with patch.dict("os.environ", {}, clear=True), \
             patch("agmem.evaluate_checkpoints.make_task_agent", side_effect=AssertionError("No Task/Reader")), \
             patch("agmem.evaluate_checkpoints.make_reward_scorer", side_effect=AssertionError("Injected Judge")), \
             patch("agmem.evaluate_checkpoints.announce"):
            return judge_remote_answers(self.config, None, self.examples, self.retrieval, self.target,
                                        bundle, responses, self.scorer.provenance(), workers=2,
                                        reward_scorer=self.scorer)

    def source_bytes(self):
        return {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}

    def test_export_without_local_answers_binds_retrieval_without_leaking_gold(self):
        original = self.source_bytes()
        with patch.dict("os.environ", {}, clear=True):
            path = self.export()
        bundle = load_bundle(path)
        self.assertFalse((self.source / "answers/local").exists())
        self.assertFalse((self.target / "answers/local").exists())
        self.assertEqual(bundle["manifest"]["protocol"], "checkpoint_task_requests_v2")
        self.assertEqual(bundle["manifest"]["source_kind"], "retrieval_reprompt")
        self.assertIsNone(bundle["manifest"]["source_task_manifest_sha256"])
        self.assertEqual(bundle["manifest"]["retrieval_manifest_sha256"], self.retrieval.signature)
        self.assertEqual(bundle["manifest"]["task_input_contract"], task_input_contract("history_answer_v1"))
        for request in bundle["requests"]:
            self.assertEqual([m["role"] for m in request["messages"]], ["user"])
            self.assertIn("2023-01-02", request["messages"][0]["content"])
            self.assertIn("The original chat says blue.", request["messages"][0]["content"])
        text = json.dumps(bundle)
        for private in ("HIDDEN_GOLD", "SCORING_ONLY", "api_key", "DIRECTORY_SUMMARY_NOT_REQUIRED"):
            self.assertNotIn(private, text)
        before = path.read_bytes()
        self.assertEqual(self.export(), path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.source_bytes(), original)

    def test_export_rejects_changed_questions_gold_dataset_or_settings(self):
        path = self.export()
        before = path.read_bytes()
        for key, value in (("query", "changed"), ("gold_answers", ["changed"]), ("source", "foreign")):
            examples = deepcopy(self.examples)
            setattr(examples[0], key, value)
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "questions/gold/dataset"):
                self.export(examples=examples)
        remote = {**self.remote, "model_revision": "different"}
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.export(remote=remote)
        self.assertEqual(path.read_bytes(), before)

    def test_export_requires_complete_source_and_separate_target(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            self.export(target=self.source)
        self.assertFalse((self.source / "offline").exists())
        (self.source / "retrieval/complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse((self.target / "offline").exists())

    def test_tagged_round_trip_judges_only_answer_and_retains_raw_text_with_resume(self):
        path = self.export()
        responses = self.responses(path)
        original = self.source_bytes()
        report = self.judge(path, responses)
        self.assertEqual(report["pooled"]["judge_accuracy"], 1)
        self.assertCountEqual(self.calls, [(e.sample_id, "blue") for e in self.examples])
        output = self.target / "answers/remote"
        journal = EpisodeJournal(output, json.loads((output / "manifest.json").read_text()))
        self.assertEqual(json.loads((output / "manifest.json").read_text())["task_input_contract"],
                         task_input_contract("history_answer_v1"))
        for example in self.examples:
            row = journal.read(example.sample_id)
            self.assertEqual(row["prediction"], "blue")
            self.assertEqual(row["task_completion"]["answer_parser"], "answer_tag_v1")
            self.assertIn("<answer> blue </answer>", row["task_completion"]["raw_text"])
            self.assertIn("explanation", row["task_completion"]["raw_text"])
        self.calls.clear()
        self.assertEqual(self.judge(path, responses), report)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.source_bytes(), original)

    def test_changed_imported_raw_text_or_evidence_rejected_before_judge(self):
        path = self.export()
        responses = self.responses(path)
        self.judge(path, responses)
        self.calls.clear()
        answer_path = responses / "episodes/00000000.json"
        original = answer_path.read_bytes()
        answer = json.loads(original)
        answer["completion"]["text"] = "A different explanation. <answer>blue</answer>"
        atomic_json(answer_path, answer)
        with self.assertRaisesRegex(ValueError, "Already-scored"):
            self.judge(path, responses)
        self.assertEqual(self.calls, [])
        answer_path.write_bytes(original)
        episode_path = self.source / "retrieval/episodes/00000001.json.gz"
        with gzip.open(episode_path, "rt") as stream:
            episode = json.load(stream)
        episode["trajectory"]["evidence"][0]["raw_text"] = "A modified history"
        with gzip.open(episode_path, "wt") as stream:
            json.dump(episode, stream)
        with self.assertRaisesRegex(ValueError, "evidence mismatch"):
            self.judge(path, responses)
        self.assertEqual(self.calls, [])

    def test_bad_tags_or_length_rejected_before_any_judge_call(self):
        path = self.export()
        responses = self.responses(path)
        answer_path = responses / "episodes/00000001.json"
        original = answer_path.read_bytes()
        for text, finish in (("plain answer", "stop"), ("<answer></answer>", "stop"),
                             ("<answer>blue</answer><answer>red</answer>", "stop"),
                             ("<answer>blue</answer>", "length")):
            with self.subTest(text=text, finish=finish):
                answer = json.loads(original)
                answer["completion"].update(text=text, finish_reason=finish, truncated=finish == "length")
                atomic_json(answer_path, answer)
                with self.assertRaises(ValueError):
                    self.judge(path, responses)
                self.assertEqual(self.calls, [])

    def test_legacy_saved_answers_keep_original_text_and_reject_new_contract_mix(self):
        example = self.examples[0]
        evidence = self.retrieval.read(example.sample_id)["evidence"]
        text = "<answer>blue</answer>"
        record = {"task_input_hash": digest(task_messages(example, evidence)),
                  "completion": {"protocol": COMPLETION_PROTOCOL, "text": text,
                      "finish_reason": "stop", "truncated": False, "usage": None, "max_tokens": 512}}
        agent = SavedTaskAnswers({"task": {"identity": {}, "request_options": {}}}, {example.sample_id: record})
        metadata = {}
        self.assertEqual(agent.answer(example, evidence, metadata=metadata)[0], text)
        self.assertNotIn("raw_text", metadata)
        self.assertNotIn("answer_parser", metadata)
        path = self.export()
        responses = self.responses(path)
        self.config["task_agent"]["prompt_version"] = "legacy_v1"
        with self.assertRaisesRegex(ValueError, "another evaluation protocol"):
            self.judge(path, responses)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
