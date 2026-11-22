"""Freezing exact source/query mappings while the builder journal remains independent."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agmem.core import Example
from agmem.data import ADAPTER_VERSION, clean_history
from agmem.hierarchical.adapters import iter_banks
from agmem.hierarchical.binding import bind_directory
from agmem.hierarchical.reader import DirectoryReader
from agmem.hierarchical.storage import BuildStore
from agmem.hierarchical_training_store import HierarchicalTrainingStore
from agmem.prepare_hierarchical_training import prepare_frozen_store
from agmem.store import Store


class FrozenHierarchyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.build, self.target = [self.root / name for name in ("prepared", "build", "frozen")]
        self.config = {"data": {"directory_protocol": "hierarchical_v1", "layout": "combined",
                               "sources": ["locomo"], "splits": ["train", "validation"],
                               "prepared_root": str(self.target)}}
        self.rows = [self.history("Alice", "First exact history"), self.history("Bob", "Different bank content")]
        files = {}
        for split, rows in [("train", self.rows), ("validation", self.rows[:1])]:
            path = self.root / (split + ".json")
            path.write_text(json.dumps(rows)); files[split] = str(path)
        self.banks = list(iter_banks({"name": "locomo", "format": "locomo_parquet", "input_files": files},
                                    ["train", "validation"], count_tokens=len))
        source = Store(self.source)
        self.memories = []
        for index, row in enumerate(self.rows):
            memory = source.add_memory(clean_history(row)); self.memories.append(memory)
            source.add_example(Example(f"locomo/train/{index:06d}", "locomo", "train", memory,
                                       "Question?", gold_answers=["Answer"]))
        source.add_example(Example("locomo/validation/000000", "locomo", "validation", self.memories[0],
                                   "Another question?", gold_answers=["Answer"]))
        source.set_metadata("prepare_identity", {"sources": ["locomo"], "splits": ["train", "validation"],
                            "adapter_version": ADAPTER_VERSION, "storage_format": Store.FORMAT})
        source.set_metadata("prepare_complete", True)
        source.close()
        self.journal = BuildStore(self.build, {"protocol": "adjacent_parent_child_directory_v1"})
        self.addCleanup(self.journal.close)
        for bank in self.banks:
            self.journal.add_bank(bank)
            for session in bank["sessions"]:
                self.journal.add_session(session)
                directory = self.directory(session)
                self.journal.commit(session, directory, {})

    @staticmethod
    def history(speaker, text):
        return {"question_id": speaker + "-question", "haystack_session_ids": [speaker + "-session"],
                "haystack_dates": ["2024-01-01"],
                "haystack_sessions": [[{"role": speaker, "content": text, "dia_id": "D1:1"},
                                       {"role": "other", "content": "Follow-up", "dia_id": "D1:2"}]]}

    @staticmethod
    def directory(session):
        result = {"status": "complete", "mode": "single_session",
                  "parent": {"title": "Session title", "content": "A factual session summary"},
                  "children": [{"end_block": len(session["blocks"]), "title": "Topic", "content": "Topic summary",
                                "entities": [session["participants"][0]]}]}
        return bind_directory(session, result, {"model": "fixture", "model_revision": "r1"}, "tokens-v1",
                              count_tokens=len, prompt_version="directory-writer.adjacent.v1")

    def prepare(self):
        return prepare_frozen_store(self.config, self.source, self.build)

    def test_freeze_preserves_queries_histories_and_completion_cache_interface(self):
        before = {p.name: p.stat().st_mtime_ns for p in self.source.iterdir()}
        report = self.prepare()
        self.assertEqual(report["examples"], {"train": 2, "validation": 1})
        self.assertEqual((report["banks"], report["sessions"]), (2, 2))
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in self.source.iterdir()})
        store = HierarchicalTrainingStore(self.target, memory_cache_size=4)
        self.addCleanup(store.close)
        self.assertEqual(store.validate_directories(["train", "validation"]), report["directory_version"])
        self.assertEqual([e.memory_id for e in store.examples("train")], self.memories)
        for index, memory_id in enumerate(self.memories):
            directories, sessions = store.hierarchical_bank(memory_id, report["directory_version"])
            self.assertEqual(sessions[0]["records"][0]["text"], self.rows[index]["haystack_sessions"][0][0]["content"])
            self.assertEqual(directories[0]["id"], "p0001")
            self.assertEqual(directories[0]["bank_id"], sessions[0]["bank_id"])
        self.assertNotEqual(store.hierarchical_bank(self.memories[0], report["directory_version"])[0][0]["bank_id"],
                            store.hierarchical_bank(self.memories[1], report["directory_version"])[0][0]["bank_id"])
        store.put_completion("task-key", {"answer": "cached"})
        self.assertEqual(store.completion("task-key"), {"answer": "cached"})

    def test_readonly_cached_banks_deepcopy_for_reader_and_version_guard(self):
        version = self.prepare()["directory_version"]
        store = HierarchicalTrainingStore(self.target)
        self.addCleanup(store.close)
        directories, sessions = store.hierarchical_bank(self.memories[0], version)
        with self.assertRaises(TypeError):
            directories[0]["content"] = "changed"
        with self.assertRaises(TypeError):
            sessions[0]["records"].append({})
        reader = DirectoryReader(directories, sessions, count_tokens=len, retention="cumulative")
        self.assertEqual(reader.initial_state()["bank_id"], directories[0]["bank_id"])
        private = deepcopy(directories); private[0]["content"] = "ordinary mutable copy"
        self.assertNotEqual(private[0]["content"], directories[0]["content"])
        with self.assertRaisesRegex(ValueError, "version"):
            store.hierarchical_bank(self.memories[0], "different")
        with self.assertRaisesRegex(ValueError, "flat"):
            store.directory(self.memories[0], version)
        with self.assertRaisesRegex(ValueError, "frozen"):
            store.add_memory([])
        with self.assertRaisesRegex(ValueError, "frozen"):
            store.set_metadata("directory_version", "changed")

    def test_cross_bank_query_membership_is_rejected_even_when_parent_ids_match(self):
        first, second = self.banks
        first_members, second_members = first["query_memberships"], second["query_memberships"]
        first["query_memberships"], second["query_memberships"] = second_members, first_members
        with self.journal.db:
            for bank in (first, second):
                payload = {k: v for k, v in bank.items() if k != "sessions"}
                self.journal.db.execute("UPDATE banks SET payload=? WHERE bank_id=?", (json.dumps(payload), bank["bank_id"]))
        with self.assertRaisesRegex(ValueError, "source_map|snapshot"):
            self.prepare()
        self.assertFalse(self.target.exists())

    def test_missing_membership_and_partial_or_fallback_banks_are_rejected(self):
        session = self.banks[0]["sessions"][0]
        for status in ("pending", "failed", "fallback"):
            with self.subTest(status=status):
                with self.journal.db:
                    self.journal.db.execute("UPDATE sessions SET status=? WHERE bank_id=?", (status, session["bank_id"]))
                with self.assertRaisesRegex(ValueError, "not complete"):
                    self.prepare()
                self.assertFalse(self.target.exists())
        with self.journal.db:
            self.journal.db.execute("UPDATE sessions SET status='complete'")
            bank = deepcopy(self.banks[0]); bank["query_memberships"] = []
            self.journal.db.execute("UPDATE banks SET payload=? WHERE bank_id=?", (json.dumps(bank), bank["bank_id"]))
        with self.assertRaisesRegex(ValueError, "missing.*membership"):
            self.prepare()

    def test_source_address_tampering_is_rejected(self):
        bank = deepcopy(self.banks[0])
        for source in bank["source_map"]:
            source["original_session_id"] = "wrong-session"
        with self.journal.db:
            self.journal.db.execute("UPDATE banks SET payload=? WHERE bank_id=?", (json.dumps(bank), bank["bank_id"]))
        with self.assertRaisesRegex(ValueError, "source_map"):
            self.prepare()

    def test_frozen_artifact_corruption_fails_before_training(self):
        self.prepare()
        path = self.target / "hierarchical_banks.jsonl"
        path.write_text(path.read_text().replace("Topic summary", "Altered topic"))
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            HierarchicalTrainingStore(self.target)

    def test_private_record_address_tampering_is_rejected(self):
        bank = deepcopy(self.banks[0])
        next(row for row in bank["source_map"] if "record_id" in row)["message_index"] = 99
        with self.journal.db:
            self.journal.db.execute("UPDATE banks SET payload=? WHERE bank_id=?", (json.dumps(bank), bank["bank_id"]))
        with self.assertRaisesRegex(ValueError, "record source_map"):
            self.prepare()

    def test_live_journal_progress_cannot_change_frozen_snapshot_and_resume_is_idempotent(self):
        report = self.prepare()
        with self.journal.db:
            self.journal.db.execute("UPDATE sessions SET status='failed', directory=NULL")
        resumed = self.prepare()
        self.assertTrue(resumed["reused"])
        self.assertEqual(resumed["directory_version"], report["directory_version"])
        store = HierarchicalTrainingStore(self.target)
        self.addCleanup(store.close)
        self.assertEqual(store.validate_directories(["train"]), report["directory_version"])

    def test_non_independent_target_is_rejected(self):
        self.config["data"]["prepared_root"] = str(self.source / "nested")
        with self.assertRaisesRegex(ValueError, "independent"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
