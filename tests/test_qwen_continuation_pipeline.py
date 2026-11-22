"""Exercise frozen imports followed by real local-build/resume orchestration."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import Mock, patch

from agmem import build_adjacent_directories as engine
from agmem import build_qwen_continued_directories as qwen
from agmem.build_topic_event_directories import BOUNDARY_PROMPT, TopicEventDirectoryWriter
from agmem.core import InfrastructureError, file_digest
from agmem.hierarchical.writer import prompt_hashes
from tests import test_adjacent_directory_pipeline as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool


class QwenContinuationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdjacentPipelineTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.output
        self.target = self.fixture.home / "continued"
        config, identity = self.fixture.recipe([
            self.fixture.item("q1", [["ALPHA fact"], ["ALPHA question", "BETA clarification"]])])
        config["datasets"][0]["name"] = "split_longmemeval_m"
        config.update(protocol="adjacent_topic_event_deepseek_v1", retry_fallback=False, workers=1)
        resources = BOUNDARY_PROMPT.parent.parent
        identity.update(protocol=config["protocol"],
            writer={"model": "deepseek-flash", "model_revision": "provider-alias:deepseek-flash"},
            datasets=[{"spec": config["datasets"][0], "files": [{"split": "test",
                "path": config["datasets"][0]["path"], "sha256": file_digest(config["datasets"][0]["path"])}]}],
            writer_settings=config["writer"], block_policy="block.v1", boundary_prompt_version="topic_event.v2",
            prompt_hashes={p.name: file_digest(p) for p in resources.iterdir() if p.is_file()},
            base_resource_hashes=prompt_hashes(), implementation_hashes={"test_fixture": "v1"})
        identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
        with patch.object(engine, "load_recipe", return_value=deepcopy((config, identity))), \
                patch.object(engine, "announce"):
            engine.run(self.fixture.args(limit_sessions=1), tokenizer=self.fixture.tokenizer,
                embedding_tokenizer=self.fixture.tokenizer, pool_factory=Pool().factory,
                embedding_factory=Encoder().factory, writer_factory=TopicEventDirectoryWriter)
        with sqlite3.connect(self.source / "build_journal.sqlite3") as db:
            db.execute("UPDATE sessions SET status='failed',error=? WHERE status='pending'",
                       (json.dumps({"error_type": "InfrastructureError", "reason": "HTTP 402"}),))
        self.saved_source = self.source_files()
        self.old_directory = json.loads((self.source / "directories.jsonl").read_text().splitlines()[0])
        self.config, self.identity = deepcopy(config), deepcopy(identity)
        self.config.update(protocol=qwen.PROTOCOL, output_root=str(self.target), source_root=str(self.source))
        self.config["service"] = {"model": "agmem-27b"}
        self.identity.update(protocol=qwen.PROTOCOL, writer={"model": "qwen3.5-27b", "model_revision": "local-weights"})
        self.old_globals = engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results

    def source_files(self):
        return {p.name: file_digest(p) for p in self.source.iterdir() if p.is_file()}

    def run_qwen(self, pool=None, encoder=None, **changes):
        pool, encoder = pool or Pool(), encoder or Encoder(gray=True)
        with patch.object(qwen, "_BASE_LOAD_RECIPE", side_effect=lambda *a, **k: deepcopy((self.config, self.identity))), \
                patch.object(engine, "announce"):
            return qwen.run(self.fixture.args(**changes), tokenizer=self.fixture.tokenizer,
                embedding_tokenizer=self.fixture.tokenizer, pool_factory=pool.factory,
                embedding_factory=encoder.factory)

    def test_local_completion_preserves_remote_directory_and_repeat_run_issues_no_requests(self):
        pool, encoder = Pool(), Encoder(gray=True)
        report = self.run_qwen(pool, encoder)
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(report["execution"]["sessions_resumed"], 1)
        self.assertEqual([call[2] for call in pool.calls],
                         ["adjacent_boundary", "adjacent_child", "adjacent_parent"])
        self.assertEqual({call[1] for call in pool.calls}, {"p0002"})
        rows = [json.loads(line) for line in (self.target / "directories.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0], self.old_directory)
        self.assertEqual(rows[1]["build"]["writer_model"], "qwen3.5-27b")
        self.assertEqual(rows[1]["build"]["writer_revision"], "local-weights")
        self.assertEqual(self.saved_source, self.source_files())
        resumed = self.run_qwen(Pool(forbidden=True), Encoder(forbidden=True))
        self.assertEqual(resumed["execution"]["sessions_resumed"], 2)
        self.assertEqual(self.saved_source, self.source_files())
        self.assertEqual((engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results),
                         self.old_globals)

    def test_dry_run_no_service_calls_no_target_writes(self):
        pool, encoder = Mock(), Mock()
        with patch("agmem.policy.load_tokenizer") as load:
            report = self.run_qwen(pool, encoder, dry_run=True)
        self.assertEqual(report["protocol"], qwen.PROTOCOL)
        self.assertFalse(self.target.exists())
        load.assert_not_called()
        pool.factory.assert_not_called()
        encoder.factory.assert_not_called()
        self.assertEqual(self.saved_source, self.source_files())

    def test_import_only_needs_no_endpoints_and_never_constructs_services(self):
        pool, encoder = Mock(), Mock()
        self.run_qwen(pool, encoder, import_only=True, base_urls=None, embedding_base_url=None)
        self.assertTrue((self.target / "deepseek_import_manifest.json").is_file())
        pool.factory.assert_not_called()
        encoder.factory.assert_not_called()
        with sqlite3.connect(self.target / "build_journal.sqlite3") as db:
            self.assertEqual(dict(db.execute("SELECT status,COUNT(*) FROM sessions GROUP BY status")),
                             {"complete": 1, "pending": 1})

    def test_boundary_outage_is_failed_not_fallback_and_can_resume(self):
        with self.assertRaises(InfrastructureError):
            self.run_qwen(Pool(fail_review=True), Encoder(gray=True))
        with sqlite3.connect(self.target / "build_journal.sqlite3") as db:
            self.assertEqual(dict(db.execute("SELECT status,COUNT(*) FROM sessions GROUP BY status")),
                             {"complete": 1, "failed": 1})
        self.assertEqual(self.run_qwen()["states"]["complete"], 2)
        self.assertEqual(self.source_files(), self.saved_source)
        self.assertEqual((engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results),
                         self.old_globals)

    def test_changed_source_summary_semantics_rejected_before_import_or_calls(self):
        self.identity["writer_settings"]["child_content_tokens"] = 999
        pool, encoder = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "writer_settings"):
            self.run_qwen(pool, encoder)
        self.assertFalse(self.target.exists())
        pool.factory.assert_not_called()
        encoder.factory.assert_not_called()

    def test_refuses_fallback_regeneration_and_remote_key_configuration(self):
        for changes in ({"retry_fallback": True}, {"service": {"model": "deepseek-flash"}},
                        {"service": {"api_key_env": "DEEPSEEK_API_KEY"}}):
            with self.subTest(changes=changes):
                old = deepcopy(self.config)
                self.config.update(changes)
                with self.assertRaises(ValueError):
                    self.run_qwen()
                self.config = old
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
