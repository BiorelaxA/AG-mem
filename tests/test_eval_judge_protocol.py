"""Evaluation can reproduce the actual training Judge without reverting new prompts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agmem.core import Example
from agmem.judge_prompts import LOCOMO_REFINED, LONGMEMEVAL_TEMPLATES
from agmem.judges import JudgeScorer, load_judge_templates, template_hashes
from agmem.store import Store


SNAPSHOT = Path(__file__).resolve().parents[1] / "configs/protocols/format05_training_judge.json"
TRAINING_LOCOMO_SHA = "fd44ba60ff07d28f4b44057a96a80cfbb45e2a7973dab7aae8051c7aabad06ea"


class FrozenJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.service_config = {"model": "agmem-27b", "model_revision": "fixture-v1",
                               "base_url": "http://127.0.0.1:8000/v1", "retries": 0}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def scorer(self, judge_config):
        requests = []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            result = '"CORRECT"' if "Core principle" in body["messages"][0]["content"] else None
            text = '{"label":' + result + '}' if result else "yes"
            return httpx.Response(200, json={"choices": [
                {"finish_reason": "stop", "message": {"content": text}}]})
        scorer = JudgeScorer(self.service_config, self.store, judge_config)
        scorer.service.client.close()
        scorer.service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(scorer.close)
        return scorer, requests

    def test_snapshot_matches_training_hash_and_preserves_original_provenance(self):
        templates, upstream, hashes = load_judge_templates({"template_file": str(SNAPSHOT)})
        self.assertEqual(hashes["locomo_refined"], TRAINING_LOCOMO_SHA)
        self.assertEqual(hashes, template_hashes(templates))
        self.assertEqual(templates["longmemeval_official"], LONGMEMEVAL_TEMPLATES)
        self.assertNotIn("local_adaptation", upstream["locomo_refined"])
        self.assertIn("strict granularity", templates["locomo_refined"])

    def test_frozen_locomo_exactly_matches_user_requested_prompt(self):
        requested = (Path(__file__).parent / "fixtures/locomo_refined_requested.txt").read_text()
        templates, _, _ = load_judge_templates({"template_file": str(SNAPSHOT)})
        actual = templates["locomo_refined"].format(
            question="{question}", gold_answer="{gold_answer}", generated_answer="{generated_answer}")
        self.assertEqual(actual.strip(), requested.strip())

    def test_explicit_frozen_scorer_does_not_change_default_templates_or_cache(self):
        current_before = LOCOMO_REFINED
        frozen, old_requests = self.scorer({"template_file": str(SNAPSHOT)})
        current, new_requests = self.scorer({})
        example = Example("locomo/test/0", "locomo", "test", "m", "When?",
                          gold_answers=["July 2023"])
        for scorer in (frozen, current):
            self.assertEqual(scorer.score(example, "July 2023"), 1)
        self.assertEqual(LOCOMO_REFINED, current_before)
        self.assertIn("strict granularity", old_requests[0]["messages"][0]["content"])
        self.assertEqual(new_requests[0]["messages"][0]["content"], LOCOMO_REFINED.format(
            question="When?", gold_answer="July 2023", generated_answer="July 2023"))
        self.assertNotEqual(frozen.cache_scope, current.cache_scope)
        self.assertEqual(current.provenance()["template_sha256"]["locomo_refined"],
                         hashlib.sha256(LOCOMO_REFINED.encode()).hexdigest())

    def test_frozen_longmemeval_routing_including_abstention(self):
        frozen, requests = self.scorer({"template_file": str(SNAPSHOT)})
        for kind in LONGMEMEVAL_TEMPLATES:
            if kind == "abstention":
                continue
            for abstention in (False, True):
                example = Example(f"split_longmemeval_s/test/{kind}/{abstention}",
                                  "split_longmemeval_s", "test", "m", f"Question {kind}?",
                                  gold_answers=["Reference"], judge_metadata={
                                      "question_type": kind,
                                      "question_id": "q_abs" if abstention else "q"})
                self.assertEqual(frozen.score(example, "Prediction"), 1)
                branch = "abstention" if abstention else kind
                expected = LONGMEMEVAL_TEMPLATES[branch].format(example.query, "Reference", "Prediction")
                self.assertEqual(requests[-1]["messages"][0]["content"], expected)

    def test_expected_hash_mismatch_fails_before_constructing_service(self):
        _, _, expected = load_judge_templates({"template_file": str(SNAPSHOT)})
        bad = deepcopy(expected)
        bad["locomo_refined"] = "0" * 64
        with patch("agmem.judges.ChatService") as service:
            with self.assertRaisesRegex(ValueError, "required evaluation protocol"):
                JudgeScorer(self.service_config, self.store, {
                    "template_file": str(SNAPSHOT), "expected_template_sha256": bad})
            service.assert_not_called()
        scorer, _ = self.scorer({"template_file": str(SNAPSHOT), "expected_template_sha256": expected})
        self.assertEqual(scorer.provenance()["template_sha256"], expected)

    def test_changed_snapshot_contents_cannot_keep_its_old_manifest(self):
        value = json.loads(SNAPSHOT.read_text())
        value["templates"]["locomo_refined"] += " changed"
        path = Path(self.temp.name) / "changed.json"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "content SHA-256"):
            load_judge_templates({"template_file": str(path)})

    def test_incomplete_snapshot_is_rejected(self):
        value = json.loads(SNAPSHOT.read_text())
        del value["templates"]["longmemeval_official"]["abstention"]
        path = Path(self.temp.name) / "incomplete.json"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "complete LoCoMo and LongMemEval"):
            load_judge_templates({"template_file": str(path)})


if __name__ == "__main__":
    unittest.main()
