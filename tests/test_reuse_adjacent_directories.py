"""Source-verified directory rebinding without inference or source mutation."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from agmem import reuse_adjacent_directories as reuse
from agmem.build_adjacent_directories import AdjacentBuildStore
from agmem.core import canonical, digest, file_digest
from agmem.hierarchical.adapters import iter_banks
from agmem.hierarchical.binding import bind_directory, directory_version, validate_directory
from agmem.hierarchical.writer import public_session
from tests.test_adjacent_directory_pipeline import Tokenizer


class ReuseAdjacentDirectoriesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.old_data, self.new_data = self.home / "old-data", self.home / "new-data"
        self.source, self.target = self.home / "source", self.home / "target"
        self.source_name = "split_longmemeval_s"
        self.tokenizer = Tokenizer()
        self.count = lambda text: len(self.tokenizer.encode(text))
        rows = []
        for i in range(3):
            rows.append({"question_id": f"q{i}", "question": f"question {i}", "answer": f"answer {i}",
                         "question_type": "multi-session", "question_date": "2024-01-03",
                         "haystack_sessions": canonical([
                             [{"role": "user", "content": f"fact {i}a"}, {"role": "assistant", "content": "reply a"}],
                             [{"role": "user", "content": f"fact {i}b"}, {"role": "assistant", "content": "reply b"}]]),
                         "haystack_session_ids": [f"source_{i}_1", f"source_{i}_2"],
                         "haystack_dates": ["2024-01-01", "2024-01-02"]})
        self.old_rows = {"train": rows[:1], "validation": [], "test": rows[1:]}
        self.new_rows = {"train": rows[:2], "validation": [], "test": rows[2:]}
        schema = pa.Table.from_pylist(rows).schema
        for root, selected in [(self.old_data, self.old_rows), (self.new_data, self.new_rows)]:
            (root / self.source_name).mkdir(parents=True)
            for split, data in selected.items():
                pq.write_table(pa.Table.from_pylist(data, schema=schema), root / self.source_name / f"{split}.parquet")
        def datasets(root):
            spec = {"name": self.source_name, "format": "longmemeval_parquet", "input_root": str(root),
                    "splits": ["train", "validation", "test"], "modality_policy": "existing_captions"}
            return [{"spec": spec, "files": [{"split": split, "path": str(root / self.source_name / f"{split}.parquet"),
                       "sha256": file_digest(root / self.source_name / f"{split}.parquet")}
                      for split in ["train", "validation", "test"]]}]
        self.old_identity = {"protocol": "adjacent_parent_child_directory_v1", "datasets": datasets(self.old_data),
                             "writer": {"model": "fixture", "model_revision": "fixed"},
                             "tokenizer_id": "fixture-tokenizer", "prompt_version": "directory-writer.adjacent.v1",
                             "writer_settings": {"parent_content_tokens": 400}, "segmentation": {"threshold": .8}}
        self.new_identity = {**self.old_identity, "datasets": datasets(self.new_data)}
        self.config = {"output_root": str(self.target), "datasets": [self.new_identity["datasets"][0]["spec"]]}
        old_locations = {row["question_id"]: (split, index) for split, values in self.old_rows.items()
                         for index, row in enumerate(values)}
        old_files = {r["split"]: r for r in self.old_identity["datasets"][0]["files"]}
        new_files = {r["split"]: r for r in self.new_identity["datasets"][0]["files"]}
        self.mapping_rows = []
        for split, values in self.new_rows.items():
            for index, row in enumerate(values):
                osplit, oi = old_locations[row["question_id"]]
                oh, nh = old_files[osplit]["sha256"], new_files[split]["sha256"]
                self.mapping_rows.append({"source": self.source_name, "question_id": row["question_id"],
                    "old_split": osplit, "old_row_index": oi, "new_split": split, "new_row_index": index,
                    "old_file_sha256": oh, "new_file_sha256": nh, "source_session_count": 2,
                    "old_bank_id": "bank_" + digest([self.source_name, osplit, oh, oi])[:24],
                    "new_bank_id": "bank_" + digest([self.source_name, split, nh, index])[:24]})
        self.mapping = self.new_data / "row_mapping.jsonl"
        self.mapping.write_text("".join(canonical(row) + "\n" for row in self.mapping_rows))
        (self.new_data / "manifest.json").write_text(canonical({"row_mapping": {
            "sha256": file_digest(self.mapping), "rows": len(self.mapping_rows)}}))
        self.old_banks = list(iter_banks(self.old_identity["datasets"][0]["spec"], count_tokens=self.count))
        self.new_banks = list(iter_banks(self.new_identity["datasets"][0]["spec"], count_tokens=self.count))
        store = AdjacentBuildStore(self.source, self.old_identity)
        for index, bank in enumerate(self.old_banks[:2]):
            store.add_bank(bank)
            for si, session in enumerate(bank["sessions"]):
                store.add_session(session)
                key = digest({"input": public_session(session), "construction": self.old_identity,
                              "modality_policy": "existing_captions"})
                store.save_segments(key, {"spans": [{"start_block": 1, "end_block": 2}], "stats": {}, "traces": []})
                if index == 1 and si == 1:
                    continue
                result = {"parent": {"title": "Parent topic", "content": "Parent source facts"},
                          "children": [{"end_block": 2, "title": "Child topic", "content": "Child source facts",
                                        "entities": ["source"], "start_reason": "session_start"}],
                          "status": "fallback" if index == 1 else "complete", "mode": "single_session"}
                directory = bind_directory(session, result, self.old_identity["writer"], "fixture-tokenizer",
                                           count_tokens=self.count, prompt_version=self.old_identity["prompt_version"])
                store.commit(session, directory, {"calls": 2})
        store.close()
        (self.source / ".build.lock").touch()

    def args(self, **kwargs):
        return Namespace(**{"config": "unused", "reuse_root": str(self.source), "mapping": str(self.mapping),
                            "datasets": [self.source_name], "splits": ["train", "validation", "test"],
                            "workers": None, "output_root": None, "embedding_model_path": None,
                            "dry_run": False, **kwargs})

    def run_reuse(self, **kwargs):
        with patch.object(reuse, "load_recipe", return_value=(self.config, self.new_identity)), \
                patch.object(reuse, "announce"):
            return reuse.run(self.args(**kwargs), tokenizer=self.tokenizer)

    def db(self, root):
        return sqlite3.connect(root / "build_journal.sqlite3")

    def test_verified_rebind_retains_text_ids_boundaries_and_rekeys_segment_cache(self):
        before = file_digest(self.source / "build_journal.sqlite3")
        report = self.run_reuse()
        self.assertEqual(report["counters"]["imported_complete"], 2)
        self.assertEqual(report["counters"]["imported_fallback"], 1)
        self.assertEqual(report["counters"]["imported_pending"], 1)
        self.assertEqual(report["counters"]["unregistered"], 2)
        self.assertEqual(file_digest(self.source / "build_journal.sqlite3"), before)
        with self.db(self.target) as target, self.db(self.source) as source:
            for mapping in self.mapping_rows[:2]:
                rows = target.execute("SELECT parent_id,snapshot,directory FROM sessions WHERE bank_id=?",
                                      (mapping["new_bank_id"],)).fetchall()
                expected_bank = next(b for b in self.new_banks if b["bank_id"] == mapping["new_bank_id"])
                for pid, snapshot, directory in rows:
                    session = json.loads(snapshot)
                    self.assertEqual(session, next(s for s in expected_bank["sessions"] if s["parent_id"] == pid))
                    old_snapshot, old_directory = source.execute("SELECT snapshot,directory FROM sessions WHERE bank_id=? AND parent_id=?",
                                                                (mapping["old_bank_id"], pid)).fetchone()
                    self.assertNotEqual(session["source_hash"], json.loads(old_snapshot)["source_hash"])
                    if directory:
                        directory, original = json.loads(directory), json.loads(old_directory)
                        validate_directory(directory, session, count_tokens=self.count)
                        self.assertEqual(directory["children"], original["children"])
                        self.assertEqual(directory["content"], original["content"])
                        self.assertNotEqual(directory["directory_version"], original["directory_version"])
                    key = digest({"input": public_session(session), "construction": self.new_identity,
                                  "modality_policy": "existing_captions"})
                    self.assertIsNotNone(target.execute("SELECT payload FROM segment_cache WHERE key=?", (key,)).fetchone())
            self.assertEqual(target.execute("SELECT COUNT(*) FROM reuse_provenance").fetchone()[0], 4)

    def test_resume_does_not_overwrite_newly_built_target_directory(self):
        self.run_reuse()
        mapping = self.mapping_rows[1]
        with self.db(self.target) as db:
            row = db.execute("SELECT directory FROM sessions WHERE bank_id=? AND parent_id='p0001'",
                             (mapping["new_bank_id"],)).fetchone()
            directory = json.loads(row[0]); directory["content"] = "New successful parent answer"
            directory["build"]["status"] = "complete"
            directory["directory_version"] = directory_version(directory)
            db.execute("UPDATE sessions SET status='complete',directory=? WHERE bank_id=? AND parent_id='p0001'",
                       (canonical(directory), mapping["new_bank_id"]))
        report = self.run_reuse()
        self.assertEqual(report["counters"]["target_sessions_preserved"], 4)
        with self.db(self.target) as db:
            self.assertEqual(json.loads(db.execute("SELECT directory FROM sessions WHERE bank_id=? AND parent_id='p0001'",
                                                 (mapping["new_bank_id"],)).fetchone()[0]), directory)

    def test_dry_run_creates_no_target_or_source_artifacts(self):
        before = {p.name: file_digest(p) for p in self.source.iterdir() if p.is_file()}
        report = self.run_reuse(dry_run=True)
        self.assertEqual(report["source_states"], {"complete": 2, "fallback": 1, "pending": 1, "unregistered": 2})
        self.assertFalse(self.target.exists())
        self.assertEqual({p.name: file_digest(p) for p in self.source.iterdir() if p.is_file()}, before)

    def test_changed_semantics_or_mapping_are_rejected_before_target_creation(self):
        self.new_identity["segmentation"] = {"threshold": .9}
        with self.assertRaisesRegex(ValueError, "semantics"):
            self.run_reuse()
        self.new_identity["segmentation"] = deepcopy(self.old_identity["segmentation"])
        self.mapping.write_text(self.mapping.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            self.run_reuse()
        self.assertFalse(self.target.exists())

    def test_dataset_modality_and_adapter_cannot_change_with_only_paths(self):
        for field, value in [("modality_policy", "references_only"), ("format", "longmemeval_json"),
                             ("unexpected_source_policy", "different")]:
            identity = deepcopy(self.new_identity)
            identity["datasets"][0]["spec"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "adapter/modality"):
                reuse.compatible_identity(self.old_identity, identity)
        # An old root may contain additional datasets that are not being reused.
        old = deepcopy(self.old_identity)
        old["datasets"].append({"spec": {"name": "locomo", "format": "locomo_parquet"}, "files": []})
        reuse.compatible_identity(old, self.new_identity)

    def test_private_source_addresses_and_coverage_are_verified(self):
        bid = self.old_banks[0]["bank_id"]
        with self.db(self.source) as db:
            original = json.loads(db.execute("SELECT payload FROM banks WHERE bank_id=?", (bid,)).fetchone()[0])
        variants = []
        for field, value in [("bank_id", "wrong_bank"), ("source_hash", "wrong_hash"),
                             ("original_session_id", "wrong_session"), ("session_index", 999),
                             ("source_format", "wrong_format")]:
            bank = deepcopy(original)
            bank["source_map"][0][field] = value
            variants.append((field, bank))
        for field, value in [("record_id", "p0001:r9999"), ("message_index", 999),
                             ("original_message_id", "wrong_message")]:
            bank = deepcopy(original)
            row = next(r for r in bank["source_map"] if "record_id" in r)
            row[field] = value
            variants.append((field, bank))
        bank = deepcopy(original)
        bank["source_map"].append(deepcopy(bank["source_map"][0]))
        variants.append(("duplicate", bank))
        bank = deepcopy(original)
        bank["source_map"] = bank["source_map"][1:]
        variants.append(("missing_parent", bank))
        bank = deepcopy(original)
        index = next(i for i, r in enumerate(bank["source_map"]) if "record_id" in r)
        del bank["source_map"][index]
        variants.append(("missing_record", bank))
        for label, bank in variants:
            with self.db(self.source) as db:
                db.execute("UPDATE banks SET payload=? WHERE bank_id=?", (canonical(bank), bid))
            with self.subTest(field=label), self.assertRaisesRegex(ValueError, "Source map"):
                self.run_reuse()
        with self.db(self.source) as db:
            db.execute("UPDATE banks SET payload=? WHERE bank_id=?", (canonical(original), bid))

    def test_full_bank_source_map_allows_unregistered_sessions_without_importing_their_addresses(self):
        mapping = self.mapping_rows[1]
        with self.db(self.source) as db:
            db.execute("DELETE FROM sessions WHERE bank_id=? AND parent_id='p0002'", (mapping["old_bank_id"],))
        report = self.run_reuse()
        self.assertEqual(report["counters"]["unregistered"], 3)
        with self.db(self.target) as db:
            bank = json.loads(db.execute("SELECT payload FROM banks WHERE bank_id=?",
                                         (mapping["new_bank_id"],)).fetchone()[0])
        self.assertEqual({r["parent_id"] for r in bank["source_map"]}, {"p0001"})
        self.assertEqual(len(bank["source_map"]), 3)  # One session row and its two complete message addresses.

    def test_snapshot_must_match_original_raw_messages(self):
        with self.db(self.source) as db:
            bid, pid, snapshot = db.execute("SELECT bank_id,parent_id,snapshot FROM sessions ORDER BY bank_id,parent_id LIMIT 1").fetchone()
            session = json.loads(snapshot)
            session["records"][0]["text"] = "unexpected mutation"
            db.execute("UPDATE sessions SET snapshot=? WHERE bank_id=? AND parent_id=?", (canonical(session), bid, pid))
        with self.assertRaisesRegex(ValueError, "snapshot text"):
            self.run_reuse()

    def test_failed_source_session_remains_pending_and_error_is_audited(self):
        with self.db(self.source) as db:
            db.execute("UPDATE sessions SET status='failed',error=? WHERE status='pending'",
                       (canonical({"reason": "ReadTimeout"}),))
        self.run_reuse()
        with self.db(self.target) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sessions WHERE status='pending'").fetchone()[0], 1)
            entries = [json.loads(r[0]) for r in db.execute("SELECT payload FROM reuse_provenance")]
            failed = next(r for r in entries if r["source_status"] == "failed")
            self.assertEqual(failed["source_error"], {"reason": "ReadTimeout"})


if __name__ == "__main__":
    unittest.main()
