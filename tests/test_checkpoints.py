import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from agmem.checkpoints import best_checkpoint, retain_checkpoints
from agmem.core import atomic_json
from agmem.pipeline import save_checkpoint


class SavedModel:
    def save_pretrained(self, path, **kwargs):
        atomic_json(Path(path) / 'config.json', {'fixture': True})


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'training': {'keep_last_checkpoints': 2}, 'reward': {'formula': 'outcome_gated_v2'}}
        self.policy = SimpleNamespace(model=SavedModel(), tokenizer=SavedModel(), policy_version=0)
        self.optimizer = SimpleNamespace(state_dict=lambda: {'resume_fixture': 123})

    def save(self, step, reward=None, outcome=None, pooled_reward=100):
        self.policy.policy_version = step
        report = None if reward is None else {
            'source_macro': {'reward': reward, 'outcome_f1': outcome},
            'pooled': {'reward': pooled_reward}}
        return save_checkpoint(self.config, self.policy, self.optimizer, self.root, step, 0, step,
                               validation=report)

    def names(self):
        return sorted(p.name for p in self.root.glob('checkpoint-*') if p.is_dir())

    def test_recent_and_two_independent_best_checkpoints_survive_pruning(self):
        self.save(1, 0.7, 0.8)
        self.save(2, 0.6, 0.9)
        self.save(3)
        self.save(4)
        self.assertEqual(len(self.names()), 4)
        self.assertEqual(best_checkpoint(self.root).name, 'checkpoint-00000001')
        self.assertEqual(best_checkpoint(self.root, 'best_outcome').name, 'checkpoint-00000002')
        improved = self.save(5, 0.8, 0.95, pooled_reward=-100)
        self.assertEqual(self.names(), ['checkpoint-00000004', 'checkpoint-00000005'])
        self.assertEqual(best_checkpoint(self.root), improved)
        import torch
        state = torch.load(improved / 'trainer.pt', weights_only=True)
        self.assertEqual(state['optimizer'], {'resume_fixture': 123})
        self.assertEqual((state['step'], state['next_index'], state['policy_version']), (5, 5, 5))

    def test_ties_keep_earlier_model_and_manifest_is_recoverable(self):
        first = self.save(1, -1, 0)
        self.save(2, -1, 0)
        self.save(3)
        self.save(4)
        # Simulate interruption after checkpoint commit but before pointer update.
        (self.root / 'best_checkpoints.json').write_text('{}')
        restored = retain_checkpoints(self.root, 2)
        self.assertEqual(restored['best_reward']['checkpoint'], first.name)
        self.assertTrue(first.exists())
        self.assertEqual(self.names(), ['checkpoint-00000001', 'checkpoint-00000003', 'checkpoint-00000004'])

    def test_failed_save_and_uncommitted_validation_never_replace_best(self):
        first = self.save(1, 0.5, 0.6)
        with patch('torch.save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.save(2, 1, 1)
        partial = self.root / 'checkpoint-00000002.building'
        atomic_json(partial / 'validation_metrics.json', {'source_macro': {'reward': 1, 'outcome_f1': 1}})
        retain_checkpoints(self.root, 1)
        self.assertEqual(best_checkpoint(self.root), first)
        self.assertTrue(partial.exists())

    def test_best_and_latest_failed_save_preserves_both_existing_checkpoints(self):
        self.config['training']['checkpoint_retention'] = 'best_and_latest'
        first = self.save(1, 0.5, 0.9)
        latest = self.save(2, 0.6, 0.8)
        previous = (self.root / 'best_checkpoints.json').read_bytes()
        with patch('torch.save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.save(3, 1, 1)
        self.assertTrue((first / 'trainer.pt').is_file())
        self.assertTrue((latest / 'trainer.pt').is_file())
        self.assertEqual((self.root / 'best_checkpoints.json').read_bytes(), previous)
        self.assertFalse((self.root / 'checkpoint-00000003').exists())

    def test_best_and_latest_prunes_only_after_new_checkpoint_commit(self):
        self.config['training']['checkpoint_retention'] = 'best_and_latest'
        first = self.save(1, 0.5, 0.9)
        latest = self.save(2, 0.6, 0.8)
        destination = self.root / 'checkpoint-00000003'

        def retain_after_commit(run_dir, keep_last, *, policy):
            self.assertEqual(policy, 'best_and_latest')
            self.assertTrue((destination / 'trainer.pt').is_file())
            self.assertTrue((destination / 'model/config.json').is_file())
            self.assertTrue((destination / 'validation_metrics.json').is_file())
            self.assertFalse(destination.with_suffix('.building').exists())
            self.assertTrue(first.is_dir())
            self.assertTrue(latest.is_dir())
            # Atomic saving temporarily requires room for a third checkpoint.
            self.assertEqual(len(self.names()), 3)
            return retain_checkpoints(run_dir, keep_last, policy=policy)

        with patch('agmem.pipeline.retain_checkpoints', side_effect=retain_after_commit):
            self.save(3, 0.7, 0.7)
        self.assertEqual(self.names(), [first.name, destination.name])

    def test_no_validation_has_no_best_and_bad_scores_fail_before_write(self):
        self.save(1)
        self.save(2)
        self.save(3)
        self.assertIsNone(best_checkpoint(self.root))
        self.assertEqual(self.names(), ['checkpoint-00000002', 'checkpoint-00000003'])
        with self.assertRaisesRegex(ValueError, 'finite validation'):
            self.save(4, float('nan'), 0.1)
        self.assertFalse((self.root / 'checkpoint-00000004.building').exists())


if __name__ == '__main__':
    unittest.main()
