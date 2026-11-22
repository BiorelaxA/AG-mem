"""The fixed split/import precedes S-test and M-all inference on GPUs4/5."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SemanticTwoGpuScriptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.log = self.root / 'calls.jsonl'
        executable = self.root / 'python'
        executable.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
with Path(os.environ['TEST_CALLS']).open('a') as out:
    out.write(json.dumps({'args': sys.argv[1:], 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}) + '\\n')
if 'agmem.reuse_topic_event_directories' in sys.argv and os.environ.get('FAIL_IMPORT') == '1':
    raise SystemExit(9)
''')
        executable.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(self.root) + os.pathsep + os.environ['PATH'],
                    'AGMEM_PROJECT_DIR': str(ROOT), 'TEST_CALLS': str(self.log)}
        for name in ('CUDA_VISIBLE_DEVICES', 'AGMEM_WRITER_WORKERS', 'AGMEM_WRITER_MAX_NUM_SEQS', 'FAIL_IMPORT'):
            self.env.pop(name, None)

    def invoke(self, *args):
        result = subprocess.run(['bash', str(ROOT / 'scripts/build_longmemeval_semantic_2gpu.sh'), *args],
                                env=self.env, text=True, capture_output=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_prepare_import_and_requested_build_scopes(self):
        result, calls = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['args'][2] for r in calls], ['agmem.longmemeval_fixed_split',
            'agmem.reuse_topic_event_directories', 'agmem.build_adjacent_directories_gpu',
            'agmem.build_adjacent_directories_gpu'])
        self.assertTrue(all(r['gpus'] == '4,5' for r in calls))
        for row in calls[2:]:
            args = row['args']
            self.assertIn('--scope-completion', args)
            self.assertEqual(args[args.index('--workers') + 1], '64')
            self.assertEqual(args[args.index('--max-num-seqs') + 1], '64')
        self.assertEqual(calls[2]['args'][-4:], ['--datasets','split_longmemeval_s','--splits','test'])
        self.assertEqual(calls[3]['args'][-6:],
                         ['--datasets','split_longmemeval_m','--splits','train','validation','test'])

    def test_import_only_has_no_gpu_launcher(self):
        result, calls = self.invoke('--import-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)

    def test_import_failure_does_not_deploy_any_model(self):
        self.env['FAIL_IMPORT'] = '1'
        result, calls = self.invoke()
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(calls), 2)

    def test_m_only_and_dryrun_are_forwarded(self):
        result, calls = self.invoke('--m-only', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('--dry-run' in r['args'] for r in calls))
        self.assertEqual(calls[-1]['args'][-6:],
                         ['--datasets','split_longmemeval_m','--splits','train','validation','test'])

    def test_fixed_recipe_new_models_thresholds_and_no_rebuild(self):
        cfg = yaml.safe_load((ROOT/'configs/build_adjacent_qwen_longmemeval_31535150.yaml').read_text())
        self.assertEqual(cfg['model_path'], 'models/qwen3.5-27b')
        self.assertEqual(cfg['embedding']['model_path'], 'models/qwen3-embedding-4b')
        self.assertEqual((cfg['segmentation']['high_threshold'],cfg['segmentation']['low_threshold']),(.72,.50))
        self.assertTrue(cfg['preserve_existing_directories'])
        self.assertTrue(cfg['allow_migrated_source_identity'])
        self.assertFalse(cfg['retry_fallback'])
        self.assertEqual({s['input_root'] for s in cfg['datasets']},
                         {'data/processed/longmemeval_315_35_150_seed42_20261113'})


if __name__ == '__main__':
    unittest.main()
