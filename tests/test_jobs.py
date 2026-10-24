from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

import httpx

from agmem.core import Example, atomic_json, file_digest, load_config
from agmem.data import ADAPTER_VERSION, clean_history
from agmem.jobs import (build_all, build_source, ensure_prepared, file_lock,
                        reuse_test_directories, source_config, train_and_test)
from agmem.services import ChatService, DirectoryBuilder
from agmem.store import Store


class IndependentJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config('configs/train_eval_locomo_s.yaml')
        self.config['data'].update(input_root=str(self.root / 'input'), prepared_root=str(self.root / 'prepared'))
        self.config['jobs'].update(test_prepared_root=str(self.root / 'prepared_test'), legacy_prepared_root=str(self.root / 'absent'))
        self.config['training']['output_dir'] = str(self.root / 'experiments/locomo_s/train')
        atomic_json(self.root / 'input/manifest.json', {})
        self.job = self.root / 'job'
        self.job.mkdir()

    def make_store(self, config, records):
        store = Store(config['data']['prepared_root'])
        identity = {'input_manifest_sha256': file_digest(self.root / 'input/manifest.json'),
                    'adapter_version': ADAPTER_VERSION, 'storage_format': Store.FORMAT,
                    'sources': config['data']['sources'], 'splits': config['data']['splits'],
                    'limit_per_source_split': None}
        store.set_metadata('prepare_identity', identity)
        store.set_metadata('prepare_complete', True)
        memory = store.add_memory(records)
        for split in config['data']['splits']:
            store.add_example(Example('fixture/' + split, 'locomo', split, memory, 'q'))
        return store, memory

    def test_source_configs_isolate_test_and_preserve_writer_identity(self):
        train = source_config(self.config, 'locomo')
        test = source_config(self.config, 'locomo', test=True)
        self.assertEqual(train['data']['splits'], ['train', 'validation'])
        self.assertEqual(test['data']['splits'], ['test'])
        self.assertNotEqual(train['data']['prepared_root'], test['data']['prepared_root'])
        self.assertEqual(train['writer'], test['writer'])
        self.assertEqual(train['service'], test['service'])
        self.assertEqual(self.config['data']['sources'], ['locomo', 'split_longmemeval_s'])

    def test_lock_conflicts_and_shared_legacy_guard(self):
        path = self.root / '.lock'
        with file_lock(path, shared=True), file_lock(path, shared=True):
            with self.assertRaisesRegex(RuntimeError, 'Another job'):
                with file_lock(path):
                    self.fail('exclusive access must fail')
        with file_lock(path):
            with self.assertRaises(RuntimeError):
                with file_lock(path, shared=True):
                    self.fail('legacy writer must exclude both new jobs')
        with file_lock(path):
            pass

    def test_preparation_reuse_and_mismatch_rejection(self):
        config = source_config(self.config, 'locomo')
        records = clean_history({'haystack_sessions': [[{'role': 'user', 'content': 'fact'}]]})
        store, _ = self.make_store(config, records)
        store.close()
        with patch('agmem.jobs.prepare') as prepare:
            ensure_prepared(config)
            prepare.assert_not_called()
            changed = deepcopy(config)
            changed['data']['splits'] = ['test']
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                ensure_prepared(changed)
            prepare.assert_not_called()

    def test_test_reuse_copies_identical_history_only_and_does_not_mix_queries(self):
        train_cfg = source_config(self.config, 'locomo')
        test_cfg = source_config(self.config, 'locomo', test=True)
        records = clean_history({'haystack_sessions': [[{'role': 'user', 'content': 'fact'}]], 'haystack_session_ids': ['one']})
        origin, mid = self.make_store(train_cfg, records)
        service = SimpleNamespace(identity={'model': 'fixture'}, complete=lambda *a, **k: 'summary')
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        builder = DirectoryBuilder(origin, service, self.config['writer'], tokenizer)
        entries = builder.build(mid)
        origin.set_metadata('directory_version', builder.version)
        origin.close()
        target, test_mid = self.make_store(test_cfg, records)
        extra = target.add_memory(clean_history({'haystack_sessions': [[{'role': 'user', 'content': 'different'}]]}))
        target.close()
        self.assertEqual(mid, test_mid)
        original_examples = (Path(train_cfg['data']['prepared_root']) / 'train_examples.jsonl').read_bytes()
        self.assertEqual(reuse_test_directories(train_cfg, test_cfg), 1)
        target = Store(test_cfg['data']['prepared_root'])
        self.assertEqual(target.directory_entry(mid, builder.version, records[0]['id']), entries[0])
        self.assertEqual(len(target.directory_entries.keys()), 1)
        self.assertNotEqual(extra, mid)
        self.assertEqual(target.examples('train'), [])
        target.close()
        self.assertEqual((Path(train_cfg['data']['prepared_root']) / 'train_examples.jsonl').read_bytes(), original_examples)
        self.assertEqual(reuse_test_directories(train_cfg, test_cfg), 0)

    def test_transport_port_does_not_change_cache_or_directory_version(self):
        store = Store(self.root / 'cache')
        self.addCleanup(store.close)
        service_config = {**self.config['service'], 'model_revision': 'fixture',
                          'request_base_url': 'http://127.0.0.1:12345/v1'}
        service = ChatService(service_config, store, 'writer')
        service.client.close()
        requests = []
        def handle(request):
            requests.append(str(request.url))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'summary'}}]})
        service.client = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(service.client.close)
        version = DirectoryBuilder(store, service, self.config['writer'], None).version
        self.assertEqual(service.complete([], 300), 'summary')
        service.config['request_base_url'] = 'http://127.0.0.1:12346/v1'
        self.assertEqual(service.complete([], 300), 'summary')
        self.assertEqual(requests, ['http://127.0.0.1:12345/v1/chat/completions'])
        self.assertEqual(version, DirectoryBuilder(store, service, self.config['writer'], None).version)

    def test_directory_job_processes_all_splits_without_training(self):
        config = deepcopy(self.config)
        config['data']['sources'] += ['split_longmemeval_m']
        with patch('agmem.jobs.build_source') as build, patch('agmem.jobs.run_logged') as run:
            build_all(config)
            self.assertEqual([(c.args[1], c.kwargs.get('test', False)) for c in build.call_args_list],
                             [(s, test) for s in config['data']['sources'] for test in (False, True)])
            run.assert_not_called()

    def test_all_writer_defers_locked_source_and_processes_other_sources_first(self):
        config = deepcopy(self.config)
        config['data']['sources'] = ['locomo', 'split_longmemeval_m']
        lock = file_lock(Path(config['data']['prepared_root']) / '.locks/locomo.lock')
        lock.__enter__()
        released = False
        def release(_):
            nonlocal released
            lock.__exit__(None, None, None)
            released = True
        try:
            with patch('agmem.jobs.build_source') as build, patch('agmem.jobs.time.sleep', side_effect=release):
                build_all(config)
                self.assertEqual([c.args[1] for c in build.call_args_list],
                                 ['split_longmemeval_m', 'split_longmemeval_m', 'locomo', 'locomo'])
        finally:
            if not released:
                lock.__exit__(None, None, None)

    def test_train_job_builds_only_ls_then_trains_then_builds_test_and_evaluates(self):
        order = []
        train_dir = Path(self.config['training']['output_dir'])
        def build(config, source, test=False):
            order.append(('build', source, test))
        def run(args, log):
            if 'train' in args:
                order.append(('train',))
                checkpoint = train_dir / 'checkpoint-00000001'
                atomic_json(checkpoint / 'model/config.json', {})
                (checkpoint / 'trainer.pt').write_text('fixture')
                atomic_json(train_dir / 'run.json', {'config': self.config})
            else:
                order.append(('test',))
                self.assertEqual(args[args.index('--split') + 1], 'test')
                self.assertIn(str(train_dir / 'checkpoint-00000001/model'), args)
                output = Path(args[args.index('--output') + 1])
                atomic_json(output / 'metrics.json', {'fixture': True})
        with patch('agmem.jobs.build_source', side_effect=build), patch('agmem.jobs.run_logged', side_effect=run):
            train_and_test(self.config, self.job / 'config.yaml', self.job)
        self.assertEqual(order, [('build', 'locomo', False), ('build', 'split_longmemeval_s', False), ('train',),
                                 ('build', 'locomo', True), ('build', 'split_longmemeval_s', True), ('test',)])
        self.assertTrue((train_dir / 'train.complete').exists())
        self.assertTrue((train_dir.parent / 'evaluation.complete.json').exists())
        # A repeated job skips committed training/evaluation, but validates dirs.
        order.clear()
        store = SimpleNamespace(close=lambda: None)
        with patch('agmem.jobs.build_source', side_effect=build), patch('agmem.jobs.run_logged', side_effect=run), \
                patch('agmem.jobs.open_prepared', return_value=store), patch('agmem.pipeline.validate_directories'):
            train_and_test(self.config, self.job / 'config.yaml', self.job)
        self.assertEqual(order, [('build', 'locomo', True), ('build', 'split_longmemeval_s', True)])

    def test_failed_training_does_not_mark_complete_or_start_test(self):
        with patch('agmem.jobs.build_source') as build, patch('agmem.jobs.run_logged', side_effect=RuntimeError('train failed')):
            with self.assertRaisesRegex(RuntimeError, 'train failed'):
                train_and_test(self.config, self.job / 'config.yaml', self.job)
        self.assertEqual(build.call_count, 2)
        self.assertFalse((Path(self.config['training']['output_dir']) / 'train.complete').exists())

    def test_test_uses_validation_best_instead_of_latest_checkpoint(self):
        train_dir = Path(self.config['training']['output_dir'])
        evaluated = []
        def run(args, log):
            if 'train' in args:
                for step, reward in [(1, 0.8), (2, -1.0)]:
                    checkpoint = train_dir / f'checkpoint-{step:08d}'
                    atomic_json(checkpoint / 'model/config.json', {})
                    (checkpoint / 'trainer.pt').write_text('fixture')
                    atomic_json(checkpoint / 'validation_metrics.json',
                                {'source_macro': {'reward': reward, 'outcome_f1': (reward + 1) / 2}})
                atomic_json(train_dir / 'run.json', {'config': self.config})
            else:
                evaluated.append(args[args.index('--checkpoint') + 1])
                atomic_json(Path(args[args.index('--output') + 1]) / 'metrics.json', {})
        with patch('agmem.jobs.build_source'), patch('agmem.jobs.run_logged', side_effect=run):
            train_and_test(self.config, self.job / 'config.yaml', self.job)
        self.assertEqual(evaluated, [str(train_dir / 'checkpoint-00000001/model')])
        self.assertEqual(json.loads((train_dir.parent / 'evaluation.complete.json').read_text())['checkpoint'],
                         str(train_dir / 'checkpoint-00000001'))


if __name__ == '__main__':
    unittest.main()
