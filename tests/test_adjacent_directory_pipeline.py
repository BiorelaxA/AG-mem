"""Real segmentation, two-stage Writer and persistent source binding without GPUs."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from agmem import build_adjacent_directories as pipeline
from agmem.core import InfrastructureError, canonical
from agmem.hierarchical.embedding import embedding_settings
from agmem.hierarchical.reader import DirectoryReader
from agmem.hierarchical.segmentation import segmentation_settings
from agmem.hierarchical.writer import writer_settings


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(canonical(messages)) + [1, 2]


class Encoder:
    def __init__(self, forbidden=False, gray=False):
        self.forbidden, self.gray, self.calls, self.closed = forbidden, gray, [], False

    def factory(self, *args, **kwargs):
        return self

    def encode(self, texts):
        if self.forbidden:
            raise AssertionError("A committed or cached segmentation must not re-embed")
        self.calls.append(list(texts))
        return [[.65, (.5775 ** .5)] if self.gray and "BETA" in t else
                [0, 1] if "BETA" in t else [1, 0] for t in texts]

    def close(self):
        self.closed = True


class Pool:
    def __init__(self, *, forbidden=False, fail_parent=False, fail_review=False):
        self.forbidden, self.fail_parent, self.fail_review = forbidden, fail_parent, fail_review
        self.calls, self.closed, self.lock = [], False, Lock()

    def factory(self, *args, **kwargs):
        return self

    def for_session(self, session):
        def complete(messages, max_tokens, schema, stage):
            if self.forbidden:
                raise AssertionError("Completed sessions must resume without model calls")
            payload = json.loads(messages[-1]["content"])
            with self.lock:
                self.calls.append((session["bank_id"], session["parent_id"], stage, payload))
            if self.fail_parent and stage == "adjacent_parent":
                raise InfrastructureError("Synthetic parent service failure")
            if stage.startswith("adjacent_boundary"):
                if self.fail_review:
                    raise InfrastructureError("Synthetic review service failure")
                value = {"decision": "merge", "reason": "The request is clarified.",
                         "evidence_blocks": [payload["left"]["end_block"], payload["right"]["start_block"]]}
            else:
                value = {"title": "Source topics", "content": "The reported source facts."}
                if stage.startswith("adjacent_child"):
                    value["entities"] = []
            return {"finish_reason": "stop", "text": canonical(value), "usage": {}}
        return complete

    def close(self):
        self.closed = True


class AdjacentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.output = self.home / "output"
        self.tokenizer = Tokenizer()

    @staticmethod
    def item(qid, bodies):
        return {"question_id": qid, "question": "PRIVATE QUERY", "answer": "PRIVATE GOLD",
                "haystack_sessions": [[{"role": "user" if i % 2 == 0 else "assistant",
                                         "content": t, "has_answer": True}
                                        for i, t in enumerate(body)] for body in bodies],
                "haystack_session_ids": [f"answer_hidden_{i}" for i in range(len(bodies))],
                "haystack_dates": ["2024-01-01"] * len(bodies)}

    def recipe(self, rows, **segment_changes):
        source = self.home / "source.json"
        source.write_text(json.dumps(rows))
        config = {"protocol": pipeline.PROTOCOL, "model_path": "fake-writer", "output_root": str(self.output),
                  "workers": 3, "retry_fallback": True, "service": {},
                  "writer": writer_settings({"context_tokens": 8192, "output_tokens": 64,
                                              "max_output_tokens": 256, "safety_margin": 16}),
                  "segmentation": segmentation_settings({"context_weight": 0, **segment_changes}),
                  "embedding": {"model_id": "Qwen/Qwen3-Embedding-4B", "model_path": "fake-encoder",
                                **embedding_settings()},
                  "datasets": [{"name": "split_longmemeval_s", "path": str(source), "split": "test"}]}
        identity = {"protocol": pipeline.PROTOCOL, "writer": {"model": "fixture", "model_revision": "fixed"},
                    "tokenizer_id": "fixture-tokenizer", "embedding": {"ready": True, "model_revision": "fixed"},
                    "segmentation": config["segmentation"], "prompt_version": pipeline.PROMPT_VERSION}
        return config, identity

    def args(self, **changes):
        return Namespace(**{"config": "unused", "workers": None, "datasets": None, "splits": ["test"],
                            "base_urls": ["http://writer1/v1", "http://writer2/v1"], "base_url": None,
                            "embedding_base_url": "http://embedding/v1", "dry_run": False,
                            "limit_sessions": None, **changes})

    def run_pipeline(self, recipe, pool=None, encoder=None, **args):
        pool, encoder = pool or Pool(), encoder or Encoder()
        with patch.object(pipeline, "load_recipe", return_value=deepcopy(recipe)), patch.object(pipeline, "announce"):
            return pipeline.run(self.args(**args), tokenizer=self.tokenizer, embedding_tokenizer=self.tokenizer,
                                pool_factory=pool.factory, embedding_factory=encoder.factory)

    def rows(self, name):
        return [json.loads(line) for line in (self.output / name).read_text().splitlines()]

    def test_children_first_exact_coverage_cross_bank_cache_and_resume(self):
        history = [["ALPHA question", "ALPHA answer", "BETA question", "BETA answer"]]
        recipe = self.recipe([self.item("q1", history), self.item("q2", history)])
        pool, encoder = Pool(), Encoder()
        report = self.run_pipeline(recipe, pool, encoder)
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(report["execution"]["cache_hits"], 1)
        self.assertEqual([c[2] for c in pool.calls], ["adjacent_child", "adjacent_child", "adjacent_parent"])
        self.assertEqual(len(encoder.calls), 1)
        self.assertTrue(pool.closed and encoder.closed)
        directories = self.rows("directories.jsonl")
        for directory in directories:
            self.assertEqual(directory["build"]["prompt_version"], pipeline.PROMPT_VERSION)
            self.assertEqual([(c["source"]["start_block"], c["source"]["end_block"])
                              for c in directory["children"]], [(1, 2), (3, 4)])
        self.assertNotEqual(directories[0]["source"]["snapshot_id"], directories[1]["source"]["snapshot_id"])
        model_inputs = canonical(pool.calls) + canonical(encoder.calls)
        for hidden in ("PRIVATE QUERY", "PRIVATE GOLD", "answer_hidden", "has_answer"):
            self.assertNotIn(hidden, model_inputs)
        parent = pool.calls[-1][3]
        self.assertEqual([b["index"] for b in parent["blocks"]], [1, 2, 3, 4])
        self.assertEqual(len(parent["children"]), 2)
        saved = (self.output / "directories.jsonl").read_bytes()
        report = self.run_pipeline(recipe, Pool(forbidden=True), Encoder(forbidden=True))
        self.assertEqual(report["execution"]["sessions_resumed"], 2)
        self.assertEqual((self.output / "directories.jsonl").read_bytes(), saved)

    def test_parent_failure_keeps_boundary_cache_and_retry_does_not_reembed(self):
        recipe = self.recipe([self.item("q", [["ALPHA fact", "BETA fact"]])])
        with self.assertRaises(InfrastructureError):
            self.run_pipeline(recipe, Pool(fail_parent=True))
        report = json.loads((self.output / "build_report.json").read_text())
        self.assertEqual(report["states"]["failed"], 1)
        retry = self.run_pipeline(recipe, Pool(), Encoder(forbidden=True))
        self.assertEqual(retry["states"]["complete"], 1)
        self.assertEqual(retry["execution"]["segment_cache_hits"], 1)

    def test_failed_review_is_fallback_and_retried_instead_of_caching_bad_boundary(self):
        recipe = self.recipe([self.item("q", [["ALPHA request", "BETA clarification"]])])
        report = self.run_pipeline(recipe, Pool(fail_review=True), Encoder(gray=True))
        self.assertEqual(report["states"]["fallback"], 1)
        self.assertEqual(len(self.rows("directories.jsonl")[0]["children"]), 2)
        pool, encoder = Pool(), Encoder(gray=True)
        report = self.run_pipeline(recipe, pool, encoder)
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual(len(self.rows("directories.jsonl")[0]["children"]), 1)
        self.assertTrue(encoder.calls)
        self.assertTrue(any(c[2] == "adjacent_boundary" for c in pool.calls))

    def test_limited_smoke_run_resumes_remaining_sessions(self):
        recipe = self.recipe([self.item("q", [["ALPHA a"], ["BETA b"], ["ALPHA c"]])])
        report = self.run_pipeline(recipe, limit_sessions=1)
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual(report["execution"]["run_status"], "limited")
        report = self.run_pipeline(recipe)
        self.assertEqual(report["states"]["complete"], 3)
        self.assertEqual(report["execution"]["sessions_resumed"], 1)

    def test_changed_identity_is_rejected_before_services_are_created(self):
        recipe = self.recipe([self.item("q", [["ALPHA a"]])])
        self.run_pipeline(recipe)
        changed = deepcopy(recipe)
        changed[1]["segmentation"]["high_threshold"] = .99
        pool = Pool(forbidden=True)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_pipeline(changed, pool, Encoder(forbidden=True))
        self.assertFalse(pool.calls)

    def test_new_directories_work_with_existing_reader_visibility_rules(self):
        recipe = self.recipe([self.item("q", [["ALPHA a", "ALPHA b", "BETA c", "BETA d"]])])
        self.run_pipeline(recipe)
        from agmem.hierarchical.adapters import iter_banks
        bank = next(iter_banks(recipe[0]["datasets"][0], ["test"],
                               count_tokens=lambda text: len(self.tokenizer.encode(text))))
        reader = DirectoryReader(self.rows("directories.jsonl"), bank["sessions"],
                                 count_tokens=lambda text: len(self.tokenizer.encode(text)), retention="cumulative")
        state = reader.initial_state()
        self.assertFalse(reader.read(["p0001:c0001"], state, 9999)["ok"])
        opened = reader.read(["p0001"], state, 9999)
        self.assertTrue(opened["ok"])
        selected = reader.read(["p0001:c0001"], opened["state"], 9999)
        self.assertTrue(selected["ok"])
        texts = [r["text"] for r in selected["result"]["records"]]
        self.assertEqual(texts, ["ALPHA a", "ALPHA b"])

    def test_dry_run_has_no_model_or_store_side_effects(self):
        recipe = self.recipe([self.item("q", [["ALPHA a"]])])
        preview = self.run_pipeline(recipe, Pool(forbidden=True), Encoder(forbidden=True), dry_run=True)
        self.assertEqual(preview["protocol"], pipeline.PROTOCOL)
        self.assertFalse(self.output.exists())

    def test_partial_model_is_not_ready_in_dry_run(self):
        root = self.home / "partial-model"
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"fixture")
        identity, tokenizer = pipeline._model_identity(root, allow_missing=True)
        self.assertFalse(identity["ready"])
        self.assertIsNone(tokenizer)

    def test_recipe_selects_native_dimensions_and_checks_local_metadata(self):
        encoder = self.home / "encoder"
        encoder.mkdir()
        pooling = encoder / "1_Pooling"
        pooling.mkdir()
        config_path = self.home / "build.json"
        recipe = {"protocol": pipeline.PROTOCOL, "model_path": "fixture-writer",
                  "output_root": str(self.output), "embedding": {"model_path": str(encoder)},
                  "datasets": [{"name": "locomo"}]}
        for model_id, dimensions in (("Qwen/Qwen3-Embedding-4B", 2560),
                                     ("Qwen/Qwen3-Embedding-0.6B", 1024)):
            with self.subTest(model_id=model_id):
                recipe["embedding"]["model_id"] = model_id
                (encoder / "config.json").write_text(json.dumps({"hidden_size": dimensions}))
                (pooling / "config.json").write_text(json.dumps({"word_embedding_dimension": dimensions}))
                config_path.write_text(json.dumps(recipe))
                with patch.object(pipeline, "_model_identity", return_value=(
                        {"ready": True, "model_revision": "fixture"}, "fixture-tokenizer")), \
                        patch.object(pipeline, "input_files", return_value=[]):
                    config, identity = pipeline.load_recipe(config_path)
                    self.assertEqual(config["embedding"]["expected_dimensions"], dimensions)
                    self.assertEqual(identity["embedding"]["settings"]["expected_dimensions"], dimensions)
                    (encoder / "config.json").write_text(json.dumps({"hidden_size": 128}))
                    with self.assertRaisesRegex(ValueError, "hidden_size"):
                        pipeline.load_recipe(config_path)
                    (encoder / "config.json").write_text(json.dumps({"hidden_size": dimensions}))
                    (pooling / "config.json").write_text(json.dumps({"word_embedding_dimension": 128}))
                    with self.assertRaisesRegex(ValueError, "pooling dimension"):
                        pipeline.load_recipe(config_path)

        recipe["embedding"] = {"model_id": "Qwen/Qwen3-Embedding-4B", "expected_dimensions": 1024}
        config_path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "expected_dimensions=2560"):
            pipeline.load_recipe(config_path)


if __name__ == "__main__":
    unittest.main()
