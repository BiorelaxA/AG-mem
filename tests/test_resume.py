from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from agmem.core import atomic_json, load_config
from agmem.resume import resume_metadata, validate_resume_config


class RetrievalResumeTests(unittest.TestCase):
    def setUp(self):
        self.old = load_config('configs/default.yaml')
        self.old['retrieval']['max_ids_per_call'] = 10
        self.new = deepcopy(self.old)
        self.new['retrieval']['max_ids_per_call'] = 4
        self.new['training']['output_dir'] = 'new-experiment/train'

    def test_explicit_fork_allows_only_limit_and_destination_changes(self):
        validate_resume_config(self.old, self.new, fork_retrieval=True)
        with self.assertRaisesRegex(ValueError, 'retrieval'):
            validate_resume_config(self.old, self.new)
        for section, key, value in [('reader', 'max_context_tokens', 32768),
                                    ('retrieval', 'max_read_calls', 4),
                                    ('reward', 'format_penalty', 0.02),
                                    ('training', 'learning_rate', 1e-4),
                                    ('data', 'sources', ['locomo'])]:
            new = deepcopy(self.new)
            new[section][key] = value
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, section):
                validate_resume_config(self.old, new, fork_retrieval=True)
        new = deepcopy(self.new)
        new['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'seed'):
            validate_resume_config(self.old, new, fork_retrieval=True)

    def test_source_metadata_used_and_old_artifacts_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'old', Path(tmp)/'new'
            checkpoint = source/'checkpoint-00000073'
            checkpoint.mkdir(parents=True)
            self.old['training']['output_dir'] = str(source)
            self.new['training']['output_dir'] = str(target)
            original = {'config': self.old, 'directory_version': 'unchanged'}
            atomic_json(source/'run.json', original)
            before = (source/'run.json').read_bytes()
            self.assertEqual(resume_metadata(self.new, checkpoint, fork_retrieval=True), original)
            self.assertFalse(target.exists())
            self.assertEqual((source/'run.json').read_bytes(), before)
            target.mkdir()
            (target/'train_metrics.jsonl').write_text('existing experiment')
            with self.assertRaisesRegex(ValueError, 'fresh'):
                resume_metadata(self.new, checkpoint, fork_retrieval=True)
            self.new['training']['output_dir'] = str(source)
            with self.assertRaisesRegex(ValueError, 'fresh'):
                resume_metadata(self.new, checkpoint, fork_retrieval=True)

    def test_subsequent_ordinary_resume_and_tensorboard_compatibility(self):
        validate_resume_config(self.new, deepcopy(self.new))
        old = deepcopy(self.new)
        del old['training']['tensorboard']
        validate_resume_config(old, self.new)

    def test_id_protocol_fork_is_narrow_and_explicit(self):
        old = deepcopy(self.new)
        new = deepcopy(old)
        new['retrieval']['id_protocol'] = 'repair_v2'
        new['training']['output_dir'] = 'id-repair/train'
        new['training']['validation_every'] = 25
        validate_resume_config(old, new, fork_id_protocol=True)
        with self.assertRaisesRegex(ValueError, 'retrieval'):
            validate_resume_config(old, new)
        with self.assertRaisesRegex(ValueError, 'retrieval'):
            validate_resume_config(old, new, fork_retrieval=True)
        for section, key, value in [('retrieval', 'max_ids_per_call', 2),
                                    ('reward', 'format_penalty', .2),
                                    ('training', 'learning_rate', 1e-4),
                                    ('reader', 'max_context_tokens', 32768)]:
            changed = deepcopy(new)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume_config(old, changed, fork_id_protocol=True)
        with self.assertRaisesRegex(ValueError, 'one resume fork'):
            validate_resume_config(old, new, fork_retrieval=True, fork_id_protocol=True)

    def test_validation_interval_can_change_on_ordinary_resume(self):
        current = deepcopy(self.new)
        current['training']['validation_every'] = 25
        validate_resume_config(self.new, current)
        current['training']['learning_rate'] = 1e-4
        with self.assertRaisesRegex(ValueError, 'training'):
            validate_resume_config(self.new, current)

    def test_id_protocol_fork_requires_fresh_directory_and_supports_normal_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'old', Path(tmp)/'new'
            checkpoint = source/'checkpoint-00000100'
            checkpoint.mkdir(parents=True)
            old = deepcopy(self.new)
            old['training']['output_dir'] = str(source)
            new = deepcopy(old)
            new['retrieval']['id_protocol'] = 'repair_v2'
            new['training']['output_dir'] = str(target)
            atomic_json(source/'run.json', {'config': old})
            resume_metadata(new, checkpoint, fork_id_protocol=True)
            atomic_json(target/'run.json', {'config': new})
            resume_metadata(new, target/'checkpoint-00000101')
            with self.assertRaisesRegex(ValueError, 'fresh'):
                resume_metadata(new, checkpoint, fork_id_protocol=True)


if __name__ == '__main__':
    unittest.main()
