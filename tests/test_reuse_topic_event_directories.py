"""Mixed incremental construction preserves old artifacts and audits new ones."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agmem import build_topic_event_directories as topic
from agmem import reuse_adjacent_directories as legacy
from agmem import reuse_topic_event_directories as mixed
from agmem.core import canonical, file_digest
from agmem.hierarchical.embedding import embedding_settings
from agmem.hierarchical.segmentation import segmentation_settings
from agmem.hierarchical.writer import writer_settings
from tests import test_reuse_adjacent_directories as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool


class TopicEventReuseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReuseAdjacentDirectoriesTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        writer = writer_settings({"context_tokens": 8192, "output_tokens": 64,
                                  "max_output_tokens": 256, "safety_margin": 16,
                                  "parent_content_tokens": 400})
        segmentation = segmentation_settings({"context_weight": 0})
        f.old_identity.update(writer_settings=writer, segmentation=segmentation,
            embedding={"ready": True, "model_revision": "fixed"},
            prompt_hashes={"boundary.txt": file_digest(topic.BOUNDARY_PROMPT.parent.parent / "boundary.txt"),
                           "child.txt": "unchanged-child", "parent.txt": "unchanged-parent"},
            implementation_hashes={"build_adjacent_directories.py": "frozen-engine"})
        with f.db(f.source) as db:
            db.execute("UPDATE metadata SET value=? WHERE key='identity'", (canonical(f.old_identity),))
        new_datasets = f.new_identity["datasets"]
        f.new_identity = deepcopy(f.old_identity)
        f.new_identity.update(protocol=topic.PROTOCOL, boundary_prompt_version=topic.BOUNDARY_PROMPT_VERSION,
                              directory_reuse_policy=mixed.POLICY, datasets=new_datasets)
        f.new_identity["prompt_hashes"]["boundary.txt"] = file_digest(topic.BOUNDARY_PROMPT)
        f.new_identity["implementation_hashes"]["build_topic_event_directories.py"] = file_digest(Path(topic.__file__))
        f.new_identity["segmentation"].update(high_threshold=0.72, low_threshold=0.50)
        f.config.update(protocol=topic.PROTOCOL, preserve_existing_directories=True, retry_fallback=False,
                        model_path="fixture-writer", workers=2, writer=writer,
                        segmentation=deepcopy(f.new_identity["segmentation"]),
                        service={}, embedding={"model_id": "Qwen/Qwen3-Embedding-4B", "model_path": "fixture-encoder",
                                               **embedding_settings()})
        self.original_globals = {name: getattr(legacy, name) for name in
            ("load_recipe", "compatible_identity", "_segments", "VERSION", "AdjacentBuildStore")}

    def run_import(self, **changes):
        f = self.fixture
        with patch.object(topic, "load_recipe", return_value=(deepcopy(f.config), deepcopy(f.new_identity))), \
                patch.object(legacy, "announce"):
            return mixed.run(f.args(**changes), tokenizer=f.tokenizer)

    def assert_restored(self):
        self.assertEqual({name: getattr(legacy, name) for name in self.original_globals}, self.original_globals)

    def test_import_preserves_fallbacks_and_records_full_identity_without_old_caches(self):
        f = self.fixture
        before = file_digest(f.source / "build_journal.sqlite3")
        report = self.run_import()
        self.assertEqual(report["counters"]["imported_complete"], 2)
        self.assertEqual(report["counters"]["imported_fallback"], 1)
        self.assertEqual(report["counters"]["imported_pending"], 1)
        self.assertEqual(file_digest(f.source / "build_journal.sqlite3"), before)
        with f.db(f.target) as db, f.db(f.source) as source:
            self.assertEqual(db.execute("SELECT count(*) FROM segment_cache").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM writer_cache").fetchone()[0], 0)
            policy = json.loads(db.execute("SELECT value FROM metadata WHERE key=?", (mixed.POLICY_KEY,)).fetchone()[0])
            self.assertEqual(policy["source_construction_identity"], f.old_identity)
            self.assertFalse(policy["retry_fallback"])
            self.assertEqual(policy["imported_boundary_rule"]["similarity_thresholds"],
                             {"high_threshold": 0.78, "low_threshold": 0.55})
            self.assertEqual(policy["new_boundary_rule"]["similarity_thresholds"],
                             {"high_threshold": 0.72, "low_threshold": 0.50})
            for bid, pid, encoded in db.execute("SELECT bank_id,parent_id,payload FROM reuse_provenance"):
                prior = json.loads(encoded)
                self.assertIsNone(prior["segment_cache_before"])
                self.assertIsNone(prior["segment_cache_after"])
                newdir = db.execute("SELECT directory FROM sessions WHERE bank_id=? AND parent_id=?", (bid, pid)).fetchone()[0]
                olddir = source.execute("SELECT directory FROM sessions WHERE bank_id=? AND parent_id=?",
                                        (prior["source_bank_id"], pid)).fetchone()[0]
                if olddir:
                    newdir, olddir = json.loads(newdir), json.loads(olddir)
                    for key in ("title", "content", "entities", "children"):
                        self.assertEqual(newdir[key], olddir[key])
                    self.assertEqual(newdir["build"]["status"], olddir["build"]["status"])
        rules = [json.loads(line) for line in (f.target / "directory_rule_provenance.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["origin"] == "reused_legacy_directory" for r in rules), 3)
        self.assertEqual(sum(r["origin"] == "pending_topic_event_v2" for r in rules), 1)
        self.assertTrue(all(r["boundary_prompt_version"] == mixed.LEGACY_BOUNDARY_VERSION
                            for r in rules if r["origin"] == "reused_legacy_directory"))
        self.assertTrue(all(r["similarity_thresholds"] == {"high_threshold": 0.78, "low_threshold": 0.55}
                            for r in rules if r["origin"] == "reused_legacy_directory"))
        self.assert_restored()

    def test_only_missing_sessions_are_built_then_resume_and_reimport_preserve_all(self):
        f = self.fixture
        self.run_import()
        with f.db(f.target) as db:
            old_directories = {(b, p): d for b, p, d in db.execute(
                "SELECT bank_id,parent_id,directory FROM sessions WHERE directory IS NOT NULL")}
        args = f.args()
        args.base_urls, args.embedding_base_url, args.limit_sessions = ["http://fixture/v1"], "http://embedding/v1", None
        pool, encoder = Pool(), Encoder()
        with patch.object(topic, "_BASE_LOAD_RECIPE", return_value=(deepcopy(f.config), deepcopy(f.new_identity))), \
                patch.object(topic.legacy, "announce"):
            report = topic.run(args, tokenizer=f.tokenizer, embedding_tokenizer=f.tokenizer,
                               pool_factory=pool.factory, embedding_factory=encoder.factory)
        self.assertEqual(report["execution"]["sessions_resumed"], 3)
        self.assertEqual(report["states"]["complete"], 5)
        self.assertEqual(report["states"]["fallback"], 1)
        with f.db(f.target) as db:
            for key, original in old_directories.items():
                self.assertEqual(db.execute("SELECT directory FROM sessions WHERE bank_id=? AND parent_id=?", key).fetchone()[0], original)
        summary = json.loads((f.target / "directory_rule_summary.json").read_text())
        self.assertEqual(summary["counts"], {"built_topic_event_v2": 3, "reused_legacy_directory": 3})
        rules = [json.loads(line) for line in (f.target / "directory_rule_provenance.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["similarity_thresholds"] == {"high_threshold": 0.72, "low_threshold": 0.50}
                            for r in rules if r["origin"] == "built_topic_event_v2"))
        self.run_import()
        with patch.object(topic, "_BASE_LOAD_RECIPE", return_value=(deepcopy(f.config), deepcopy(f.new_identity))), \
                patch.object(topic.legacy, "announce"):
            report = topic.run(args, tokenizer=f.tokenizer, embedding_tokenizer=f.tokenizer,
                               pool_factory=Pool(forbidden=True).factory, embedding_factory=Encoder(forbidden=True).factory)
        self.assertEqual(report["execution"]["sessions_resumed"], 6)
        self.assert_restored()

    def test_dry_run_and_missing_import_guard_do_not_create_target(self):
        f = self.fixture
        before = file_digest(f.source / "build_journal.sqlite3")
        report = self.run_import(dry_run=True)
        self.assertEqual(report["source_states"]["fallback"], 1)
        self.assertFalse(f.target.exists())
        self.assertEqual(file_digest(f.source / "build_journal.sqlite3"), before)
        with self.assertRaisesRegex(ValueError, "explicit importer"):
            mixed.validate_mixed_policy(f.config, f.new_identity)
        self.assertTrue(mixed.validate_mixed_policy(f.config, f.new_identity, dry_run=True)["requires_import"])
        self.assertFalse(f.target.exists())
        self.assert_restored()

    def test_other_semantic_changes_and_enabling_fallback_retry_are_rejected(self):
        f = self.fixture
        changes = [("segmentation", "high_threshold", .65), ("segmentation", "low_threshold", .45),
                   ("segmentation", "min_cohesion", .25), ("writer_settings", "child_content_tokens", 500),
                   ("writer", "model_revision", "changed"), ("prompt_hashes", "child.txt", "changed"),
                   ("implementation_hashes", "build_adjacent_directories.py", "changed")]
        for section, key, value in changes:
            new = deepcopy(f.new_identity)
            new[section][key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "semantics"):
                mixed.compatible_identity(f.old_identity, new)
        bad = deepcopy(f.old_identity)
        bad["prompt_hashes"]["boundary.txt"] = "not-original"
        with self.assertRaisesRegex(ValueError, "frozen legacy"):
            mixed.compatible_identity(bad, f.new_identity)
        f.config["retry_fallback"] = True
        with self.assertRaisesRegex(ValueError, "retry_fallback=false"):
            self.run_import()
        self.assertFalse(f.target.exists())
        self.assert_restored()

    def test_preserved_directory_mutation_is_detected_without_relabeling(self):
        f = self.fixture
        self.run_import()
        with f.db(f.target) as db:
            db.execute("UPDATE sessions SET status='complete' WHERE status='fallback'")
        previous = (f.target / "directory_rule_provenance.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "Preserved source directory was changed"):
            mixed.export_rule_provenance(f.target)
        self.assertEqual((f.target / "directory_rule_provenance.jsonl").read_bytes(), previous)

    def test_interrupted_import_cannot_be_mistaken_for_ready_build(self):
        f = self.fixture
        with patch.object(legacy, "verify_bank", side_effect=ValueError("interrupted verification")):
            with self.assertRaisesRegex(ValueError, "interrupted verification"):
                self.run_import()
        self.assert_restored()
        with self.assertRaisesRegex(ValueError, "importer has not completed"):
            mixed.validate_mixed_policy(f.config, f.new_identity)
        self.assertTrue(mixed.validate_mixed_policy(f.config, f.new_identity, dry_run=True)["requires_import"])
        self.run_import()
        self.assertFalse(mixed.validate_mixed_policy(f.config, f.new_identity)["requires_import"])

    def test_successful_partial_import_only_allows_covered_splits(self):
        f = self.fixture
        self.run_import(splits=["test"])
        self.assertFalse(mixed.validate_mixed_policy(f.config, f.new_identity, splits=["test"])["requires_import"])
        with self.assertRaisesRegex(ValueError, "requested dataset/split scope"):
            mixed.validate_mixed_policy(f.config, f.new_identity)
        self.run_import(splits=["train", "validation"])
        self.assertFalse(mixed.validate_mixed_policy(f.config, f.new_identity)["requires_import"])


if __name__ == "__main__":
    unittest.main()
