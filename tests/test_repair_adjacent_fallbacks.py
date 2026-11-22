"""Scoped fallback repair, provenance migration and resumability without GPUs."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agmem import build_adjacent_directories as pipeline
from agmem import repair_adjacent_fallbacks as repair
from agmem.core import InfrastructureError, canonical
from agmem.hierarchical.embedding import embedding_settings
from agmem.hierarchical.segmentation import segmentation_settings
from agmem.hierarchical.writer import writer_settings
from tests.test_adjacent_directory_pipeline import Encoder, Pool, Tokenizer


class LongParentPool(Pool):
    """One real semantic-budget fallback; the same answer fits a 400-token cap."""

    def for_session(self, session):
        complete = super().for_session(session)

        def request(messages, max_tokens, schema, stage):
            result = complete(messages, max_tokens, schema, stage)
            if session["parent_id"] == "p0002" and stage.startswith("adjacent_parent"):
                result["text"] = canonical({"title": "Source topics", "content": " ".join(["fact"] * 250)})
            return result

        return request


class RepairAdjacentFallbackTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.output = self.home / "output"
        self.tokenizer = Tokenizer()
        source = self.home / "source.json"
        source.write_text(json.dumps([{
            "question_id": "q1", "question": "PRIVATE QUERY", "answer": "PRIVATE GOLD",
            "haystack_sessions": [
                [{"role": "user", "content": "ALPHA normal"}],
                [{"role": "user", "content": "BETA overlong parent"}],
                [{"role": "user", "content": "ALPHA pending must stay untouched"}],
            ],
            "haystack_session_ids": ["original1", "original2", "original3"],
            "haystack_dates": ["2024-01-01"] * 3,
        }]))
        self.old_config = {
            "protocol": pipeline.PROTOCOL, "model_path": "fake-writer", "output_root": str(self.output),
            "workers": 2, "retry_fallback": True, "service": {},
            "writer": writer_settings({"context_tokens": 8192, "output_tokens": 512,
                                       "max_output_tokens": 1024, "safety_margin": 16}),
            "segmentation": segmentation_settings({"context_weight": 0}),
            "embedding": {"model_id": "Qwen/Qwen3-Embedding-4B", "model_path": "fake-encoder",
                          **embedding_settings()},
            "datasets": [{"name": "split_longmemeval_s", "path": str(source), "split": "test"}],
        }
        self.old_identity = {
            "protocol": pipeline.PROTOCOL, "writer": {"model": "fixture", "model_revision": "fixed"},
            "tokenizer_id": "fixture-tokenizer", "embedding": {"ready": True, "model_revision": "fixed"},
            "segmentation": deepcopy(self.old_config["segmentation"]),
            "writer_settings": deepcopy(self.old_config["writer"]),
            "prompt_version": pipeline.PROMPT_VERSION,
        }
        self.new_config, self.new_identity = deepcopy(self.old_config), deepcopy(self.old_identity)
        self.new_config["writer"]["parent_content_tokens"] = 400
        self.new_identity["writer_settings"]["parent_content_tokens"] = 400

    def args(self, **changes):
        return Namespace(**{
            "config": "unused", "workers": None, "datasets": ["split_longmemeval_s"],
            "splits": ["train", "test"], "base_urls": ["http://writer1/v1", "http://writer2/v1"],
            "base_url": None, "embedding_base_url": "http://embedding/v1", "dry_run": False,
            "limit_sessions": None, "embedding_model_path": None, "output_root": None,
            **changes,
        })

    def db(self):
        return sqlite3.connect(self.output / "build_journal.sqlite3")

    def metadata(self, name):
        with self.db() as db:
            row = db.execute("SELECT value FROM metadata WHERE key=?", (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def session_rows(self):
        with self.db() as db:
            return db.execute("SELECT bank_id,parent_id,status,directory FROM sessions ORDER BY parent_id").fetchall()

    def empty_store(self):
        store = pipeline.AdjacentBuildStore(self.output, self.old_identity)
        store.close()

    def initial_build(self):
        pool, encoder = LongParentPool(), Encoder()
        with patch.object(pipeline, "load_recipe", return_value=(self.old_config, self.old_identity)), \
                patch.object(pipeline, "announce"):
            result = pipeline.run(self.args(splits=["test"], limit_sessions=2), tokenizer=self.tokenizer,
                                  embedding_tokenizer=self.tokenizer,
                                  pool_factory=pool.factory, embedding_factory=encoder.factory)
        self.assertEqual(result["states"]["complete"], 1)
        self.assertEqual(result["states"]["fallback"], 1)
        self.assertEqual(result["states"]["pending"], 1)
        return result

    def run_repair(self, pool=None, encoder=None, **changes):
        pool, encoder = pool or LongParentPool(), encoder or Encoder(forbidden=True)
        with patch.object(repair, "load_recipe", return_value=(deepcopy(self.new_config),
                                                               deepcopy(self.new_identity))), \
                patch.object(repair, "announce"):
            return repair.run(self.args(**changes), tokenizer=self.tokenizer,
                              embedding_tokenizer=self.tokenizer,
                              pool_factory=pool.factory, embedding_factory=encoder.factory)

    def test_identity_migration_is_read_only_until_applied_and_is_idempotent(self):
        self.empty_store()
        repair.prepare_identity(self.output, self.new_identity, apply=False)
        self.assertEqual(self.metadata("identity"), self.old_identity)
        self.assertIsNone(self.metadata("parent_budget_migrations"))
        repair.prepare_identity(self.output, self.new_identity, apply=True)
        self.assertEqual(self.metadata("identity"), self.new_identity)
        history = self.metadata("parent_budget_migrations")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["from_identity"], self.old_identity)
        self.assertEqual(history[0]["to_identity"], self.new_identity)
        repair.prepare_identity(self.output, self.new_identity, apply=True)
        self.assertEqual(self.metadata("parent_budget_migrations"), history)

    def test_identity_rejects_other_changes_without_mutating_database(self):
        self.empty_store()
        variants = []
        changed = deepcopy(self.new_identity)
        changed["writer_settings"]["child_content_tokens"] = 400
        variants.append(changed)
        changed = deepcopy(self.new_identity)
        changed["segmentation"]["high_threshold"] = .99
        variants.append(changed)
        changed = deepcopy(self.new_identity)
        changed["writer"]["model_revision"] = "different-model"
        variants.append(changed)
        changed = deepcopy(self.new_identity)
        changed["writer_settings"]["parent_content_tokens"] = 100
        variants.append(changed)
        for identity in variants:
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                repair.prepare_identity(self.output, identity, apply=True)
            self.assertEqual(self.metadata("identity"), self.old_identity)
            self.assertIsNone(self.metadata("parent_budget_migrations"))

    def test_selection_only_includes_existing_fallbacks_in_requested_splits(self):
        self.empty_store()
        with self.db() as db:
            for bid, dataset, splits in [
                ("train", "split_longmemeval_s", ["train"]),
                ("test", "split_longmemeval_s", ["test"]),
                ("validation", "split_longmemeval_s", ["validation"]),
                ("shared", "split_longmemeval_s", ["train", "test"]),
                ("locomo", "locomo", ["test"]),
            ]:
                db.execute("INSERT INTO banks VALUES (?,?,?)", (bid, dataset, canonical({
                    "bank_id": bid, "query_memberships": [{"split": s} for s in splits]})))
                for index, status in enumerate(["fallback", "complete", "failed", "pending", "empty"], 1):
                    db.execute("INSERT INTO sessions(bank_id,parent_id,source_hash,snapshot,status) VALUES (?,?,?,?,?)",
                               (bid, f"p{index:04d}", "hash", "{}", status))
        with self.db() as db:
            selected = repair.select_fallbacks(db, ["split_longmemeval_s"], ["train", "test"])
        self.assertEqual(set(selected), {("split_longmemeval_s", b, "p0001") for b in ("train", "test", "shared")})
        self.assertEqual(len(selected), 3)

    def test_repair_reuses_old_boundaries_and_preserves_complete_and_pending_rows(self):
        self.initial_build()
        before = self.session_rows()
        pool, encoder = LongParentPool(), Encoder(forbidden=True)
        report = self.run_repair(pool, encoder)
        after = self.session_rows()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[2], after[2])
        self.assertEqual(after[1][2], "complete")
        self.assertEqual(len(json.loads(after[1][3])["content"].split()), 250)
        self.assertTrue(pool.calls)
        self.assertTrue(all(call[1] == "p0002" for call in pool.calls))
        self.assertEqual(encoder.calls, [])
        self.assertEqual(report["execution"]["selected"], 1)
        self.assertEqual(report["execution"]["repaired"], 1)
        self.assertEqual(report["execution"]["remaining_fallback"], 0)
        self.assertEqual(report["states"]["pending"], 1)
        self.assertEqual(self.metadata("identity"), self.new_identity)
        self.assertEqual(json.loads((self.output / "run_manifest.json").read_text()), self.new_identity)
        again = self.run_repair(Pool(forbidden=True), Encoder(forbidden=True))
        self.assertEqual(again["execution"]["selected"], 0)
        self.assertEqual(self.session_rows(), after)

    def test_transport_failure_keeps_old_fallback_and_can_resume(self):
        self.initial_build()
        before = self.session_rows()
        with self.assertRaises(InfrastructureError):
            self.run_repair(Pool(fail_parent=True))
        self.assertEqual(self.session_rows(), before)
        report = json.loads((self.output / "build_report.json").read_text())
        self.assertEqual(report["execution"]["repair_errors"], 1)
        self.assertEqual(report["execution"]["remaining_fallback"], 1)
        resumed = self.run_repair()
        self.assertEqual(resumed["execution"]["repaired"], 1)
        self.assertEqual(self.session_rows()[1][2], "complete")

    def test_normal_builder_can_later_resume_with_the_400_token_recipe(self):
        self.initial_build()
        self.run_repair()
        before = self.session_rows()
        pool, encoder = Pool(), Encoder()
        with patch.object(pipeline, "load_recipe", return_value=(self.new_config, self.new_identity)), \
                patch.object(pipeline, "announce"):
            report = pipeline.run(self.args(splits=["test"]), tokenizer=self.tokenizer,
                                  embedding_tokenizer=self.tokenizer,
                                  pool_factory=pool.factory, embedding_factory=encoder.factory)
        after = self.session_rows()
        self.assertEqual(before[:2], after[:2])
        self.assertEqual(report["states"]["complete"], 3)
        self.assertEqual(report["states"]["pending"], 0)
        self.assertTrue(pool.calls)
        self.assertTrue(all(call[1] == "p0003" for call in pool.calls))

    def test_dry_run_preserves_identity_and_does_not_start_services(self):
        self.initial_build()
        before = self.session_rows()
        pool_factory, encoder_factory = unittest.mock.Mock(), unittest.mock.Mock()
        with patch.object(repair, "load_recipe", return_value=(self.new_config, self.new_identity)), \
                patch.object(repair, "announce"):
            repair.run(self.args(dry_run=True), tokenizer=self.tokenizer,
                       embedding_tokenizer=self.tokenizer, pool_factory=pool_factory,
                       embedding_factory=encoder_factory)
        pool_factory.assert_not_called()
        encoder_factory.assert_not_called()
        self.assertEqual(self.session_rows(), before)
        self.assertEqual(self.metadata("identity"), self.old_identity)
        self.assertIsNone(self.metadata("parent_budget_migrations"))


if __name__ == "__main__":
    unittest.main()
