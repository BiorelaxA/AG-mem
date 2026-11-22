"""Full context-segment build, source binding and resume without model services."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from agmem import build_adjacent_directories as legacy
from agmem import build_segment_context_directories as context
from agmem.core import InfrastructureError, canonical, file_digest
from agmem.hierarchical.reader import DirectoryReader
from tests import test_adjacent_directory_pipeline as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool


class RecordingPool(Pool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = []

    def for_session(self, session):
        complete = super().for_session(session)

        def record(messages, max_tokens, schema, stage):
            self.messages.append((stage, deepcopy(messages)))
            return complete(messages, max_tokens, schema, stage)
        return record


class SegmentContextPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdjacentPipelineTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.original_globals = (legacy.PROTOCOL, legacy.PROMPT_VERSION,
                                 legacy.load_recipe, legacy.DATASETS)

    def recipe(self, body=None, *, three_splits=False):
        body = body or ["ALPHA one", "ALPHA two", "BETA three", "ALPHA four",
                        "ALPHA five", "ALPHA six", "BETA seven", "BETA eight", "ALPHA nine"]
        item = self.fixture.item("private-question", [body])
        cfg, identity = self.fixture.recipe([item])
        cfg.update(protocol=context.PROTOCOL, child_context_segments=3)
        cfg["writer"]["context_blocks"] = 0
        cfg["segmentation"].update(high_threshold=.72, low_threshold=.50)
        cfg["datasets"][0].update(name="locomo", format="locomo_parquet")
        if three_splits:
            spec = cfg["datasets"][0]
            spec.pop("path")
            spec.pop("split")
            spec["splits"] = ["train", "validation", "test"]
            spec["input_files"] = {}
            for split in spec["splits"]:
                row = deepcopy(item)
                row["question_id"] = f"private-{split}"
                row["question"] = f"PRIVATE QUERY {split}"
                row["answer"] = f"PRIVATE GOLD {split}"
                path = self.fixture.home / f"{split}.json"
                path.write_text(json.dumps([row]))
                spec["input_files"][split] = str(path)
        identity.update(protocol=context.PROTOCOL, prompt_version=legacy.PROMPT_VERSION,
                        writer_settings=deepcopy(cfg["writer"]), segmentation=deepcopy(cfg["segmentation"]),
                        prompt_hashes={"boundary.txt": "old", "child.txt": "old"},
                        implementation_hashes={"build_adjacent_directories.py": "frozen"})
        return cfg, identity

    def run_context(self, recipe, pool=None, encoder=None, **args):
        pool, encoder = pool or RecordingPool(), encoder or Encoder()
        with patch.object(context, "_BASE_LOAD_RECIPE", return_value=deepcopy(recipe)), \
             patch.object(legacy, "announce"):
            return context.run(self.fixture.args(**args), tokenizer=self.fixture.tokenizer,
                               embedding_tokenizer=self.fixture.tokenizer,
                               pool_factory=pool.factory, embedding_factory=encoder.factory)

    def assert_globals_restored(self):
        self.assertEqual((legacy.PROTOCOL, legacy.PROMPT_VERSION,
                          legacy.load_recipe, legacy.DATASETS), self.original_globals)

    def test_complete_previous_segments_deduplicate_splits_and_preserve_source_ranges(self):
        recipe = self.recipe(three_splits=True)
        pool = RecordingPool()
        report = self.run_context(recipe, pool, splits=["train", "validation", "test"])
        self.assertEqual(report["sessions"], 1)
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual(len(self.fixture.rows("bank_manifest.jsonl")[0]["query_memberships"]), 3)
        children = [c[3] for c in pool.calls if c[2] == "adjacent_child"]
        self.assertEqual(len(children), 5)
        self.assertEqual(children[0]["context_segments"], [])
        expected_spans = [(1, 2), (3, 3), (4, 6), (7, 8), (9, 9)]
        for i, payload in enumerate(children):
            previous = payload["context_segments"]
            self.assertEqual([(p["start_block"], p["end_block"]) for p in previous],
                             expected_spans[max(0, i - 3):i])
            self.assertEqual([b["index"] for p in previous for b in p["blocks"]],
                             [index for start, end in expected_spans[max(0, i - 3):i]
                              for index in range(start, end + 1)])
            start, end = expected_spans[i]
            self.assertEqual([b["index"] for b in payload["blocks"]], list(range(start, end + 1)))
            self.assertTrue(all(b["index"] < start for p in previous for b in p["blocks"]))
        final = children[-1]
        self.assertEqual([b["index"] for p in final["context_segments"] for b in p["blocks"]],
                         [3, 4, 5, 6, 7, 8])  # Three full segments, not three Blocks.
        self.assertEqual(pool.calls[-1][2], "adjacent_parent")
        self.assertEqual(len(pool.calls[-1][3]["children"]), 5)
        self.assertEqual([b["index"] for b in pool.calls[-1][3]["blocks"]], list(range(1, 10)))
        directory = self.fixture.rows("directories.jsonl")[0]
        self.assertEqual([(c["source"]["start_block"], c["source"]["end_block"])
                          for c in directory["children"]], expected_spans)
        raw = canonical(pool.messages)
        for hidden in ("PRIVATE QUERY", "PRIVATE GOLD", "answer_hidden", "has_answer"):
            self.assertNotIn(hidden, raw)
        manifest = json.loads((self.fixture.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["protocol"], context.PROTOCOL)
        self.assertEqual(manifest["child_prompt_version"], "segment_context.v1")
        self.assertEqual(manifest["child_context_policy"]["previous_segments"], 3)
        self.assertEqual(manifest["child_context_policy"]["overflow"],
                         "fail_without_truncating_or_dropping_context")
        self.assertEqual(manifest["implementation_hashes"]["build_segment_context_directories.py"],
                         file_digest(Path(context.__file__)))
        self.assertIn("segment_context_writer.py", manifest["implementation_hashes"])
        db = sqlite3.connect(self.fixture.output / "build_journal.sqlite3")
        try:
            session = json.loads(db.execute("SELECT snapshot FROM sessions").fetchone()[0])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM segment_cache").fetchone()[0], 1)
        finally:
            db.close()
        reader = DirectoryReader([directory], [session], count_tokens=lambda s: len(self.fixture.tokenizer.encode(s)),
                                 retention="cumulative", context_blocks=0)
        state = reader.read(["p0001"], reader.initial_state(), 9999)["state"]
        result = reader.read(["p0001:c0005"], state, 9999)
        self.assertEqual([r["text"] for r in result["result"]["records"]], ["ALPHA nine"])
        saved = (self.fixture.output / "directories.jsonl").read_bytes()
        resumed = self.run_context(recipe, Pool(forbidden=True), Encoder(forbidden=True),
                                   splits=["train", "validation", "test"])
        self.assertEqual(resumed["execution"]["sessions_resumed"], 1)
        self.assertEqual((self.fixture.output / "directories.jsonl").read_bytes(), saved)
        self.assert_globals_restored()

    def test_new_gray_prompt_is_used_and_parent_is_built_last(self):
        pool = RecordingPool()
        report = self.run_context(self.recipe(["ALPHA request", "BETA clarification"]),
                                  pool, Encoder(gray=True))
        self.assertEqual(report["states"]["complete"], 1)
        boundary = [messages[0]["content"] for stage, messages in pool.messages
                    if stage.startswith("adjacent_boundary")]
        expected = (Path(__file__).resolve().parents[1] / "agmem/hierarchical/resources/adjacent/"
                    "topic_event_v2/boundary.txt").read_text()
        self.assertEqual(boundary, [expected])
        self.assertEqual([c[2] for c in pool.calls], ["adjacent_boundary", "adjacent_child", "adjacent_parent"])
        self.assert_globals_restored()

    def test_existing_legacy_root_is_rejected_before_model_services(self):
        old = self.fixture.recipe([self.fixture.item("q", [["ALPHA request", "BETA clarification"]])])
        self.fixture.run_pipeline(old)
        saved = (self.fixture.output / "directories.jsonl").read_bytes()
        pool, encoder = Pool(forbidden=True), Encoder(forbidden=True)
        with self.assertRaisesRegex(ValueError, "identity|fresh|existing"):
            self.run_context(self.recipe(), pool, encoder)
        self.assertFalse(pool.calls or encoder.calls)
        self.assertEqual((self.fixture.output / "directories.jsonl").read_bytes(), saved)
        self.assert_globals_restored()

    def test_changed_context_identity_cannot_resume_and_exceptions_restore_globals(self):
        recipe = self.recipe()
        self.run_context(recipe)
        changed = deepcopy(recipe)
        changed[1]["segmentation"]["min_cohesion"] = .99
        with self.assertRaisesRegex(ValueError, "identity|fresh|existing"):
            self.run_context(changed, Pool(forbidden=True), Encoder(forbidden=True))
        self.assert_globals_restored()

    def test_parent_failure_can_resume_using_new_protocol_segmentation_cache(self):
        recipe = self.recipe()
        with self.assertRaises(InfrastructureError):
            self.run_context(recipe, Pool(fail_parent=True))
        self.assert_globals_restored()
        report = self.run_context(recipe, Pool(), Encoder(forbidden=True))
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual(report["execution"]["segment_cache_hits"], 1)
        self.assert_globals_restored()

    def test_dry_run_does_not_create_output(self):
        result = self.run_context(self.recipe(), Pool(forbidden=True), Encoder(forbidden=True), dry_run=True)
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(result["protocol"], context.PROTOCOL)
        self.assert_globals_restored()


if __name__ == "__main__":
    unittest.main()
