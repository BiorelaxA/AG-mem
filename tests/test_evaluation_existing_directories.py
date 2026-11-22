"""Required mixed flat snapshots are checked in place and never rebuilt."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agmem.core import atomic_json
from agmem.evaluation_data import load_existing_source
from agmem.evaluate_checkpoints import _run_suite, checkpoint_model
from tests import test_build_local_directories as directory_fixtures


class ExistingDirectoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = directory_fixtures.LocalDirectoryBuildTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.origin('gpt', [0])
        f.run_with_pool(lambda messages: 'Existing local summary')
        self.path = f.root / 'completed/test/split_longmemeval_s'
        self.config = json.loads((self.path / 'completed_config.json').read_text())

    def load(self):
        return load_existing_source(self.config, 'split_longmemeval_s', self.path)

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.path.iterdir() if p.is_file()}

    def test_complete_mixed_snapshot_is_read_without_any_write_or_model(self):
        # Force in-memory index reconstruction and verify it is not persisted.
        for p in self.path.glob('*.index.json'):
            p.unlink()
        before = self.snapshot()
        with patch('agmem.evaluation_data.ensure_prepared', side_effect=AssertionError('must not prepare')), \
             patch('agmem.evaluation_data.build_directories', side_effect=AssertionError('must not build')):
            result = self.load()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['validated_provenance_entries'], 4)
        self.assertEqual(result['by_writer'], {'gpt-5-mini': 1, 'qwen3.5-27B (new)': 3})
        self.assertEqual(result['config']['data']['prepared_root'], str(self.path))
        self.assertEqual(before, self.snapshot())

    def test_pending_build_is_rejected_without_repair(self):
        atomic_json(self.path / 'local_build_status.json', {'status': 'building'})
        with self.assertRaisesRegex(ValueError, 'not published'):
            self.load()

    def test_changed_provenance_entry_is_rejected(self):
        path = self.path / 'directory_provenance.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['origin']['writer_identity']['model_revision'] = 'wrong-writer'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'matching Writer provenance'):
            self.load()

    def test_changed_raw_body_is_rejected(self):
        path = self.path / 'memory_records.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['records'][0]['messages'][0]['content'] = 'Changed original evidence'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'session body'):
            self.load()

    def test_different_dataset_snapshot_and_missing_store_are_rejected(self):
        meta = json.loads((self.path / 'manifest.json').read_text())
        meta['prepare_identity']['input_manifest_sha256'] = '0' * 64
        atomic_json(self.path / 'manifest.json', meta)
        with self.assertRaisesRegex(ValueError, 'selected dataset snapshot'):
            self.load()
        missing = self.fixture.root / 'missing'
        with self.assertRaisesRegex(ValueError, 'not complete'):
            load_existing_source(self.config, 'split_longmemeval_s', missing)
        self.assertFalse(missing.exists())

    def test_required_build_stage_never_falls_back_to_writer(self):
        suite = {'datasets': ['split_longmemeval_s'], 'checkpoints': {'best': 'checkpoint-00000020'},
                 'task_agents': ['local'], 'output_root': str(self.fixture.root / 'eval'),
                 'prepared_root': str(self.fixture.root / 'unused'),
                 'required_prepared_sources': {'split_longmemeval_s': str(self.path)}}
        args = SimpleNamespace(local_base_url=None, remote_model=None, datasets=None,
                               checkpoints=None, task_agents=None, limit=None, stage='build')
        with patch('agmem.evaluation_data.prepare_source', side_effect=AssertionError('must not prepare')), \
             patch('agmem.evaluation_data.build_source_directories', side_effect=AssertionError('must not build')), \
             patch('agmem.evaluate_checkpoints.announce'):
            _run_suite(args, suite, self.config, {})


class RequiredCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'run'
        self.suite = {'training_run': str(self.root), 'checkpoints': {}, 'checkpoint_steps': {}}
        for label, step in [('best', 20), ('final', 32)]:
            checkpoint = self.root / f'checkpoint-{step:08d}'
            atomic_json(checkpoint / 'model/config.json', {})
            (checkpoint / 'model/model-1.safetensors').write_bytes(b'fixture')
            self.suite['checkpoints'][label] = checkpoint.name
            self.suite['checkpoint_steps'][label] = step
        self.model = self.root / self.suite['checkpoints']['best'] / 'model'

    def test_inference_checkpoints_need_no_trainer_state(self):
        for label in ['best', 'final']:
            checkpoint = self.root / self.suite['checkpoints'][label]
            with self.subTest(label=label):
                self.assertFalse((checkpoint / 'trainer.pt').exists())
                self.assertEqual(checkpoint_model(self.suite, label), checkpoint / 'model')
                self.suite['checkpoints'][label] = str(checkpoint)
                self.assertEqual(checkpoint_model(self.suite, label), checkpoint / 'model')
                self.assertFalse((checkpoint / 'trainer.pt').exists())

    def test_wrong_or_invalid_expected_step_is_rejected(self):
        for step in [32, -1, True, '20']:
            with self.subTest(step=step):
                self.suite['checkpoint_steps']['best'] = step
                with self.assertRaisesRegex(ValueError, 'required training run/step'):
                    checkpoint_model(self.suite, 'best')

    def test_checkpoint_outside_selected_training_run_is_rejected(self):
        self.suite['checkpoints']['best'] = str(self.model.parent)
        self.suite['training_run'] = str(self.root.parent / 'other-run')
        with self.assertRaisesRegex(ValueError, 'required training run/step'):
            checkpoint_model(self.suite, 'best')

    def test_missing_model_config_is_rejected(self):
        (self.model / 'config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing checkpoint model'):
            checkpoint_model(self.suite, 'best')

    def test_missing_or_empty_model_weights_are_rejected(self):
        weights = self.model / 'model-1.safetensors'
        weights.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing checkpoint model'):
            checkpoint_model(self.suite, 'best')
        weights.touch()
        with self.assertRaisesRegex(ValueError, 'Missing checkpoint model'):
            checkpoint_model(self.suite, 'best')

    def test_indexed_model_requires_all_nonempty_shards(self):
        atomic_json(self.model / 'model.safetensors.index.json', {
            'weight_map': {'a': 'model-1.safetensors', 'b': 'model-2.safetensors'}})
        with self.assertRaisesRegex(ValueError, 'weight shards'):
            checkpoint_model(self.suite, 'best')
        shard = self.model / 'model-2.safetensors'
        shard.touch()
        with self.assertRaisesRegex(ValueError, 'Missing checkpoint model'):
            checkpoint_model(self.suite, 'best')
        shard.write_bytes(b'fixture')
        self.assertEqual(checkpoint_model(self.suite, 'best'), self.model)


if __name__ == '__main__':
    unittest.main()
