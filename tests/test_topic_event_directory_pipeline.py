"""Versioned boundary prompts preserve v1 identity and isolate persisted caches."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from agmem import build_adjacent_directories as legacy
from agmem import build_topic_event_directories as topic
from agmem.core import InfrastructureError, file_digest
from agmem.hierarchical.reader import DirectoryReader
from tests import test_adjacent_directory_pipeline as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool


class RecordingPool(Pool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.boundary_system_prompts = []

    def for_session(self, session):
        complete = super().for_session(session)

        def record(messages, max_tokens, schema, stage):
            if stage.startswith("adjacent_boundary"):
                self.boundary_system_prompts.append(messages[0]["content"])
            return complete(messages, max_tokens, schema, stage)
        return record


class TopicEventPipelineTests(unittest.TestCase):
    def setUp(self):
        # Reuse fixtures by composition, without inheriting/rerunning their tests.
        self.fixture = fixtures.AdjacentPipelineTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.original_globals = (legacy.PROTOCOL, legacy.PROMPT_VERSION, legacy.load_recipe, legacy.DATASETS)

    def recipe(self):
        config, identity = self.fixture.recipe([
            self.fixture.item("q", [["ALPHA request", "BETA clarification"]])])
        config["protocol"] = identity["protocol"] = topic.PROTOCOL
        identity["prompt_version"] = topic.PROMPT_VERSION
        identity["prompt_hashes"] = {"boundary.txt": "legacy-boundary", "child.txt": "unchanged"}
        identity["implementation_hashes"] = {"build_adjacent_directories.py": "frozen"}
        return config, identity

    def run_topic(self, recipe, pool=None, encoder=None, **args):
        pool, encoder = pool or RecordingPool(), encoder or Encoder(gray=True)
        with patch.object(topic, "_BASE_LOAD_RECIPE", return_value=deepcopy(recipe)), \
                patch.object(legacy, "announce"):
            return topic.run(self.fixture.args(**args), tokenizer=self.fixture.tokenizer,
                             embedding_tokenizer=self.fixture.tokenizer,
                             pool_factory=pool.factory, embedding_factory=encoder.factory)

    def assert_globals_restored(self):
        self.assertEqual((legacy.PROTOCOL, legacy.PROMPT_VERSION, legacy.load_recipe, legacy.DATASETS),
                         self.original_globals)

    def test_longmemeval_m_is_enabled_only_in_the_versioned_worker(self):
        encoder = self.fixture.home / 'encoder'
        encoder.mkdir()
        (encoder / 'config.json').write_text(json.dumps({'hidden_size': 2560}))
        path = self.fixture.home / 'm_recipe.json'
        raw = {'protocol': topic.PROTOCOL, 'model_path': 'fixture-writer',
               'output_root': str(self.fixture.output), 'embedding': {'model_path': str(encoder)},
               'datasets': [{'name': 'split_longmemeval_m'}]}
        with patch.object(legacy, '_model_identity', return_value=(
                {'ready': True, 'model_revision': 'fixture'}, 'fixture-tokenizer')), \
                patch.object(legacy, 'input_files', return_value=[]):
            path.write_text(json.dumps(raw))
            config, _ = topic.load_recipe(path)
            self.assertEqual(config['datasets'][0]['name'], 'split_longmemeval_m')
            self.assertEqual(topic.argument_parser().parse_args(
                ['--datasets', 'split_longmemeval_m']).datasets, ['split_longmemeval_m'])
            self.assertNotIn('split_longmemeval_m', legacy.DATASETS)
            raw['protocol'] = legacy.PROTOCOL
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'excluded'):
                legacy.load_recipe(path)
        self.assert_globals_restored()

    def test_actual_boundary_prompt_persisted_version_and_resume(self):
        pool = RecordingPool()
        report = self.run_topic(self.recipe(), pool)
        self.assertEqual(report["states"]["complete"], 1)
        self.assertTrue(pool.boundary_system_prompts)
        self.assertEqual(set(pool.boundary_system_prompts), {topic.BOUNDARY_PROMPT.read_text()})
        old_prompt = topic.BOUNDARY_PROMPT.parent.parent / "boundary.txt"
        self.assertNotEqual(pool.boundary_system_prompts[0], old_prompt.read_text())
        directory = self.fixture.rows("directories.jsonl")[0]
        self.assertEqual(directory["schema_version"], "directory.v1")
        self.assertEqual(directory["build"]["prompt_version"], topic.PROMPT_VERSION)
        self.assertEqual([(c["source"]["start_block"], c["source"]["end_block"])
                          for c in directory["children"]], [(1, 2)])
        manifest = json.loads((self.fixture.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["protocol"], topic.PROTOCOL)
        self.assertEqual(manifest["boundary_prompt_version"], topic.BOUNDARY_PROMPT_VERSION)
        self.assertEqual(manifest["prompt_hashes"]["boundary.txt"], file_digest(topic.BOUNDARY_PROMPT))
        self.assertEqual(manifest["implementation_hashes"]["build_topic_event_directories.py"],
                         file_digest(Path(topic.__file__)))
        db = sqlite3.connect(self.fixture.output / "build_journal.sqlite3")
        try:
            self.assertEqual(db.execute("select count(*) from segment_cache").fetchone()[0], 1)
            session = json.loads(db.execute("select snapshot from sessions").fetchone()[0])
        finally:
            db.close()
        reader = DirectoryReader([directory], [session],
                                 count_tokens=lambda text: len(self.fixture.tokenizer.encode(text)),
                                 retention="cumulative")
        opened = reader.read(["p0001"], reader.initial_state(), 9999)
        evidence = reader.read(["p0001:c0001"], opened["state"], 9999)
        self.assertTrue(evidence["ok"])
        self.assertEqual([r["text"] for r in evidence["result"]["records"]],
                         ["ALPHA request", "BETA clarification"])
        retry = self.run_topic(self.recipe(), Pool(forbidden=True), Encoder(forbidden=True))
        self.assertEqual(retry["execution"]["sessions_resumed"], 1)
        self.assert_globals_restored()

    def test_v2_refuses_existing_v1_root_before_services(self):
        old_recipe = self.fixture.recipe([self.fixture.item("q", [["ALPHA request", "BETA clarification"]])])
        self.fixture.run_pipeline(old_recipe)
        saved = (self.fixture.output / "directories.jsonl").read_bytes()
        pool, encoder = Pool(forbidden=True), Encoder(forbidden=True)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_topic(self.recipe(), pool, encoder)
        self.assertFalse(pool.calls or encoder.calls)
        self.assertEqual((self.fixture.output / "directories.jsonl").read_bytes(), saved)
        self.assert_globals_restored()

    def test_exception_restores_engine_globals(self):
        pool = Pool(fail_parent=True)
        with self.assertRaises(InfrastructureError):
            self.run_topic(self.recipe(), pool)
        self.assertTrue(any(call[2] == "adjacent_parent" for call in pool.calls))
        self.assert_globals_restored()

    def test_loader_error_restores_globals_and_dry_run_does_not_create_store(self):
        with patch.object(topic, "_BASE_LOAD_RECIPE", side_effect=ValueError("bad recipe")):
            with self.assertRaisesRegex(ValueError, "bad recipe"):
                topic.load_recipe("unused")
        self.assert_globals_restored()
        self.run_topic(self.recipe(), Pool(forbidden=True), Encoder(forbidden=True), dry_run=True)
        self.assertFalse(self.fixture.output.exists())
        self.assert_globals_restored()

    def test_real_recipe_loading_changes_only_versioned_semantics(self):
        encoder = self.fixture.home / "encoder"
        encoder.mkdir()
        (encoder / "config.json").write_text(json.dumps({"hidden_size": 2560}))
        path = self.fixture.home / "recipe.json"
        raw = {"protocol": self.original_globals[0], "model_path": "fixture-writer",
               "output_root": str(self.fixture.output), "embedding": {"model_path": str(encoder)},
               "datasets": [{"name": "locomo"}]}
        with patch.object(legacy, "_model_identity", return_value=(
                {"ready": True, "model_revision": "fixture"}, "fixture-tokenizer")), \
                patch.object(legacy, "input_files", return_value=[]):
            path.write_text(json.dumps(raw))
            _, old = legacy.load_recipe(path)
            with self.assertRaisesRegex(ValueError, topic.PROTOCOL):
                topic.load_recipe(path)
            self.assert_globals_restored()
            raw["protocol"] = topic.PROTOCOL
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, self.original_globals[0]):
                legacy.load_recipe(path)
            _, new = topic.load_recipe(path)
        self.assert_globals_restored()
        expected = deepcopy(old)
        expected["protocol"] = topic.PROTOCOL
        expected["boundary_prompt_version"] = topic.BOUNDARY_PROMPT_VERSION
        expected["prompt_hashes"]["boundary.txt"] = file_digest(topic.BOUNDARY_PROMPT)
        expected["implementation_hashes"]["build_topic_event_directories.py"] = file_digest(Path(topic.__file__))
        self.assertEqual(new, expected)
        self.assertNotEqual(old, new)


if __name__ == "__main__":
    unittest.main()
