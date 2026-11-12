"""End-to-end directory persistence using real adapters/Writer, no model service."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from agmem import build_parent_child_directories as pipeline
from agmem.core import InfrastructureError, canonical
from agmem.hierarchical.adapters import iter_banks
from agmem.hierarchical.storage import BuildStore
from agmem.hierarchical.writer import prompt_hashes, writer_settings


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(canonical(messages)) + [1, 2]


class LocalPool:
    """A deterministic remote boundary; all normalization/validation remains real."""

    def __init__(self, *, reject_text=None, forbid_calls=False):
        self.reject_text, self.forbid_calls = reject_text, forbid_calls
        self.calls, self.closed, self.lock = [], False, Lock()

    def factory(self, endpoints, workers, settings, store):
        self.store = store
        return self

    def for_session(self, session):
        def complete(messages, max_tokens, schema, stage):
            with self.lock:
                self.calls.append((session["bank_id"], session["parent_id"]))
            if self.forbid_calls:
                raise AssertionError("Resume must not request a completed session again")
            if self.reject_text and any(self.reject_text in r["text"] for r in session["records"]):
                raise InfrastructureError("Synthetic service failure")
            text, payload = messages[-1]["content"], None
            for index, character in enumerate(text):
                if character == "{":
                    try:
                        candidate, _ = json.JSONDecoder().raw_decode(text[index:])
                    except ValueError:
                        continue
                    if isinstance(candidate, dict) and "write_range" in candidate:
                        payload = candidate
            if payload is None:
                raise AssertionError("Fixture unexpectedly needed rollup")
            result = {"parent": {"title": "Session", "content": "Source topics."},
                      "children": [{"end_block": payload["write_range"]["end"], "title": "Topic",
                                    "content": "Reported facts.", "entities": []}]}
            return {"text": canonical(result), "finish_reason": "stop", "usage": {}}
        return complete

    def close(self):
        self.closed = True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.output = self.home / "built"
        self.tokenizer = Tokenizer()

    def fixture(self, rows, *, name="split_longmemeval_s"):
        path = self.home / (name + ".json")
        path.write_text(json.dumps(rows), encoding="utf-8")
        settings = writer_settings({"context_tokens": 8192, "output_tokens": 64,
                                    "max_output_tokens": 256, "safety_margin": 16})
        config = {"protocol": "parent_child_directory_v1", "model_path": "fake-local-model",
                  "output_root": str(self.output), "workers": 3, "writer": settings,
                  "datasets": [{"name": name, "path": str(path), "split": "test"}],
                  "service": {}, "retry_fallback": True}
        identity = {"protocol": config["protocol"], "writer": {"model": "local-test", "model_revision": "fixed"},
                    "tokenizer_id": "test-tokenizer", "prompt_hashes": prompt_hashes(),
                    "writer_settings": settings, "request_options": {"temperature": 0}}
        return config, identity

    @staticmethod
    def item(question_id, texts):
        return {"question_id": question_id, "question": "PRIVATE QUESTION", "answer": "PRIVATE GOLD",
                "haystack_sessions": [[{"role": "user", "content": text, "has_answer": True}] for text in texts],
                "haystack_session_ids": [f"private-{i}" for i in range(len(texts))],
                "haystack_dates": ["2024-01-01"] * len(texts)}

    def args(self, *, dry_run=False):
        return Namespace(config="unused.yaml", workers=None, datasets=None, splits=["test"],
                         base_urls=["http://localhost:9999/v1"], base_url=None, dry_run=dry_run)

    def run_pipeline(self, recipe, pool, **kwargs):
        with patch.object(pipeline, "load_recipe", return_value=deepcopy(recipe)), patch.object(pipeline, "announce"):
            return pipeline.run(self.args(**kwargs), tokenizer=self.tokenizer, pool_factory=pool.factory)

    def jsonl(self, filename):
        return [json.loads(line) for line in (self.output / filename).read_text().splitlines()]

    def test_build_cross_bank_cache_rebinds_and_resume_performs_no_requests(self):
        recipe = self.fixture([self.item("answer_q1", ["Shared fact"]),
                               self.item("answer_q2", ["Distinct prefix", "Shared fact"])])
        pool = LocalPool()
        report = self.run_pipeline(recipe, pool)
        self.assertEqual(report["states"]["complete"], 3)
        self.assertEqual(report["execution"]["cache_hits"], 1)
        self.assertEqual(len(pool.calls), 2)
        self.assertTrue(pool.closed)
        directories = self.jsonl("directories.jsonl")
        self.assertEqual(len({d["bank_id"] for d in directories}), 2)
        self.assertEqual({d["id"] for d in directories}, {"p0001", "p0002"})
        for directory in directories:
            child = directory["children"][0]
            self.assertEqual(child["id"], directory["id"] + ":c0001")
            self.assertEqual(child["source"]["start_record_id"], directory["id"] + ":r0001")
        # Source binding incorporates the bank, even when public Writer input is identical.
        self.assertEqual(len({d["source"]["snapshot_id"] for d in directories}), 3)
        public = canonical(directories) + canonical(self.jsonl("records.jsonl"))
        self.assertNotIn("PRIVATE GOLD", public)
        self.assertNotIn("answer_q", public)
        first = {name: (self.output / name).read_bytes() for name in
                 ("directories.jsonl", "records.jsonl", "blocks.jsonl", "bank_manifest.jsonl")}
        resumed = LocalPool(forbid_calls=True)
        report = self.run_pipeline(recipe, resumed)
        self.assertEqual(report["execution"]["sessions_resumed"], 3)
        self.assertEqual(resumed.calls, [])
        for name, original in first.items():
            self.assertEqual((self.output / name).read_bytes(), original)

    def test_one_service_failure_keeps_other_sessions_and_retry_only_builds_failure(self):
        recipe = self.fixture([self.item("q", ["Bad service target", "Valid evidence"])])
        pool = LocalPool(reject_text="Bad service")
        with self.assertRaises(InfrastructureError):
            self.run_pipeline(recipe, pool)
        report = json.loads((self.output / "build_report.json").read_text())
        self.assertEqual((report["states"]["failed"], report["states"]["complete"]), (1, 1))
        self.assertEqual(len(self.jsonl("directories.jsonl")), 1)
        self.assertEqual(len(self.jsonl("records.jsonl")), 2)
        self.assertEqual(len(json.loads((self.output / "retry_failed.json").read_text())), 1)
        retry = LocalPool()
        report = self.run_pipeline(recipe, retry)
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(len(retry.calls), 1)
        self.assertEqual(report["execution"]["sessions_resumed"], 1)

    def test_malformed_source_session_is_reported_while_locomo_neighbor_builds(self):
        rows = [{"sample_id": "private", "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob", "session_1": "broken list",
            "session_1_date_time": "2024-01-01", "session_2_date_time": "2024-01-02",
            "session_2": [{"speaker": "Alice", "text": "Unmodified body.", "dia_id": "D2:1"}]},
                 "qa": [{"question": "Private", "answer": "Secret"}]}]
        recipe = self.fixture(rows, name="locomo")
        with self.assertRaises(InfrastructureError):
            self.run_pipeline(recipe, LocalPool())
        report = json.loads((self.output / "build_report.json").read_text())
        self.assertEqual((report["source_failures"], report["states"]["complete"]), (1, 1))
        self.assertEqual(self.jsonl("records.jsonl")[0]["text"], "Unmodified body.")

    def test_source_iterator_exception_drains_and_exports_completed_work(self):
        recipe = self.fixture([self.item("q", ["Valid evidence"])])
        source = iter_banks(recipe[0]["datasets"][0], ["test"], count_tokens=lambda text: len(text.split()))
        bank = next(source)
        def interrupted(*args, **kwargs):
            yield bank
            raise RuntimeError("Synthetic later file read failure")
        pool = LocalPool()
        with patch.object(pipeline, "iter_banks", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "later file read"):
                self.run_pipeline(recipe, pool)
        self.assertTrue(pool.closed)
        self.assertEqual(len(self.jsonl("directories.jsonl")), 1)
        report = json.loads((self.output / "build_report.json").read_text())
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual(report["execution"]["run_status"], "interrupted")
        self.assertEqual(report["execution"]["fatal_error"]["error_type"], "RuntimeError")

    def test_pending_session_is_listed_in_retry_manifest(self):
        recipe = self.fixture([self.item("q", ["Evidence awaiting Writer"])])
        bank = next(iter_banks(recipe[0]["datasets"][0], ["test"],
                               count_tokens=lambda text: len(text.split())))
        store = BuildStore(self.output, recipe[1])
        try:
            store.add_bank(bank)
            store.add_session(bank["sessions"][0])
            report = store.export()
        finally:
            store.close()
        self.assertEqual(report["states"]["pending"], 1)
        retry = json.loads((self.output / "retry_failed.json").read_text())
        self.assertEqual(len(retry), 1)
        self.assertEqual((retry[0]["bank_id"], retry[0]["parent_id"], retry[0]["status"]),
                         (bank["bank_id"], bank["sessions"][0]["parent_id"], "pending"))

    def test_dry_run_has_no_output_or_client_side_effects(self):
        recipe = self.fixture([self.item("q", ["Evidence"])])
        pool = LocalPool(forbid_calls=True)
        with patch.object(pool, "factory", side_effect=AssertionError("dry-run created client")):
            preview = self.run_pipeline(recipe, pool, dry_run=True)
        self.assertEqual(preview["protocol"], "parent_child_directory_v1")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
