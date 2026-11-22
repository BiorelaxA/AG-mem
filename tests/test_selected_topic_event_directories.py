"""Selected completion must prove whole input coverage without hiding API failures."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from agmem import build_topic_event_directories as topic
from agmem import build_selected_topic_event_directories as selected
from agmem.core import InfrastructureError, atomic_json, canonical, file_digest
from agmem.hierarchical.adapters import input_files, iter_banks
from agmem.hierarchical.binding import bind_directory
from agmem.hierarchical.storage import BuildStore
from tests import test_adjacent_directory_pipeline as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool


class SelectedTopicCompletionTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.AdjacentPipelineTests(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        config, identity = self.f.recipe([self.f.item('unused', [['ALPHA one fact']])])
        self.data = self.f.home / 'data'
        specs = []
        for source in ('split_longmemeval_s', 'split_longmemeval_m'):
            path = self.data / source
            path.mkdir(parents=True)
            for split in ('train', 'validation', 'test'):
                rows = [self.f.item(f'{source}-{split}-{index}', [['ALPHA one fact']])
                        for index in range(2 if split == 'test' else 1)]
                pq.write_table(pa.Table.from_pylist(rows), path / f'{split}.parquet')
            specs.append({'name': source, 'format': 'longmemeval_parquet',
                          'input_root': str(self.data), 'splits': ['train', 'validation', 'test'],
                          'modality_policy': 'existing_captions'})
        config.update(protocol=topic.PROTOCOL, datasets=specs)
        identity.update(protocol=topic.PROTOCOL, prompt_version=topic.PROMPT_VERSION,
                        prompt_hashes={'boundary.txt': 'legacy', 'child.txt': 'unchanged'},
                        implementation_hashes={'build_adjacent_directories.py': 'frozen'},
                        datasets=[{'spec': spec, 'files': [
                            {'split': split, 'path': str(path), 'sha256': file_digest(path)}
                            for split, path in input_files(spec)]} for spec in specs])
        self.base = (config, identity)
        with patch.object(topic, '_BASE_LOAD_RECIPE', return_value=deepcopy(self.base)):
            self.config, self.identity = topic.load_recipe('unused')
        self.args = self.f.args(datasets=['split_longmemeval_s'], splits=['test'])
        self.banks = {}
        store = BuildStore(self.f.output, self.identity)
        try:
            for split in ('train', 'test'):
                banks = list(iter_banks(specs[0], [split], count_tokens=self.count,
                                       tokenizer_id=self.identity['tokenizer_id']))
                self.banks[split] = banks
                for bank in banks:
                    store.add_bank(bank)
                    for session in bank['sessions']:
                        store.add_session(session)
                        if split == 'test':
                            result = {'status': 'complete', 'mode': 'single_session',
                                      'parent': {'title': 'One topic', 'content': 'One fact.'},
                                      'children': [{'end_block': len(session['blocks']),
                                                    'title': 'One fact', 'content': 'One fact.', 'entities': []}]}
                            directory = bind_directory(session, result, self.identity['writer'],
                                                       self.identity['tokenizer_id'], count_tokens=self.count,
                                                       prompt_version=topic.PROMPT_VERSION)
                            store.commit(session, directory, {})
            store.export({'run_status': 'incomplete', 'datasets': ['split_longmemeval_s'],
                          'splits': ['test'], 'limit_sessions': None})
        finally:
            store.close()

    def count(self, text):
        return len(self.f.tokenizer.encode(text))

    def mutate(self, sql, params=()):
        with sqlite3.connect(self.f.output / 'build_journal.sqlite3') as db:
            db.execute(sql, params)

    def verify(self, args=None):
        return selected.verify_selected_scope(args or self.args, self.config, self.identity)

    def invoke(self, **factories):
        with patch.object(topic, '_BASE_LOAD_RECIPE', return_value=deepcopy(self.base)), \
                patch.object(topic.legacy, 'announce'), redirect_stdout(StringIO()):
            return selected.run(self.args, tokenizer=self.f.tokenizer,
                                embedding_tokenizer=self.f.tokenizer,
                                pool_factory=factories.get('pool_factory', Pool(forbidden=True).factory),
                                embedding_factory=factories.get('embedding_factory', Encoder(forbidden=True).factory))

    def test_selected_test_is_complete_despite_unselected_train_pending(self):
        result = self.verify()
        self.assertIsInstance(result, dict)
        report = json.loads((self.f.output / 'build_report.json').read_text())
        self.assertEqual(report['states']['pending'], 1)
        self.assertEqual(report['execution']['run_status'], 'incomplete')

    def test_run_accepts_only_the_completed_selected_scope_and_preserves_global_report(self):
        result = self.invoke()
        self.assertIsInstance(result, dict)
        report = json.loads((self.f.output / 'build_report.json').read_text())
        self.assertEqual(report['states']['pending'], 1)
        self.assertEqual(report['execution']['run_status'], 'incomplete')
        self.assertEqual(report['execution']['sessions_resumed'], 2)

    def test_selected_missing_bank_parent_or_failed_status_are_not_complete(self):
        bank_id = self.banks['test'][0]['bank_id']
        cases = [
            ('bank', 'DELETE FROM banks WHERE bank_id=?', (bank_id,)),
            ('parent', 'DELETE FROM sessions WHERE bank_id=?', (bank_id,)),
            ('failed', "UPDATE sessions SET status='failed',directory=NULL WHERE bank_id=?", (bank_id,)),
        ]
        database = self.f.output / 'build_journal.sqlite3'
        original = database.read_bytes()
        for case, sql, params in cases:
            with self.subTest(case=case):
                database.write_bytes(original)
                self.mutate(sql, params)
                with self.assertRaises((ValueError, InfrastructureError)):
                    self.verify()

    def test_unregistered_selected_m_does_not_succeed_as_empty_work(self):
        args = deepcopy(self.args)
        args.datasets = ['split_longmemeval_m']
        with self.assertRaises((ValueError, InfrastructureError)):
            self.verify(args)

    def test_changed_directory_is_rejected(self):
        bank_id = self.banks['test'][0]['bank_id']
        with sqlite3.connect(self.f.output / 'build_journal.sqlite3') as db:
            value = json.loads(db.execute('SELECT directory FROM sessions WHERE bank_id=?', (bank_id,)).fetchone()[0])
            value['content'] = 'Tampered source description'
            db.execute('UPDATE sessions SET directory=? WHERE bank_id=?', (canonical(value), bank_id))
        with self.assertRaises((ValueError, InfrastructureError)):
            self.verify()

    def test_missing_directory_export_is_not_complete(self):
        (self.f.output / 'directories.jsonl').write_text('')
        with self.assertRaises((ValueError, InfrastructureError)):
            self.verify()

    def test_selected_request_failure_is_never_converted_to_success(self):
        bank_id = self.banks['test'][0]['bank_id']
        self.mutate("UPDATE sessions SET status='pending',directory=NULL WHERE bank_id=?", (bank_id,))
        with self.assertRaises((ValueError, InfrastructureError)):
            self.invoke(pool_factory=Pool(fail_parent=True).factory, embedding_factory=Encoder().factory)
        report = json.loads((self.f.output / 'build_report.json').read_text())
        self.assertGreater(report['states']['failed'], 0)

    def test_nonterminal_api_exception_is_propagated_unchanged(self):
        error = InfrastructureError('API transport failed')
        with patch.object(topic, 'run', side_effect=error), \
                patch.object(topic, 'load_recipe', return_value=(self.config, self.identity)), \
                self.assertRaises(InfrastructureError) as raised:
            selected.run(self.args)
        self.assertIs(raised.exception, error)

    def test_forged_terminal_text_with_interrupted_report_is_not_swallowed(self):
        path = self.f.output / 'build_report.json'
        report = json.loads(path.read_text())
        report['execution'].update(run_status='interrupted', fatal_error={'error_type': 'TimeoutError'})
        atomic_json(path, report)
        error = InfrastructureError(f'Some sessions failed or remain pending; inspect {path}; rerun to resume')
        with patch.object(topic, 'run', side_effect=error), \
                patch.object(topic, 'load_recipe', return_value=(self.config, self.identity)), \
                self.assertRaises(InfrastructureError):
            selected.run(self.args)

    def test_same_terminal_text_from_another_raise_site_is_not_swallowed(self):
        path = self.f.output / 'build_report.json'
        error = InfrastructureError(f'Some sessions failed or remain pending; inspect {path}; rerun to resume')
        with patch.object(topic, 'run', side_effect=error), \
                patch.object(topic, 'load_recipe', return_value=(self.config, self.identity)), \
                self.assertRaises(InfrastructureError):
            selected.run(self.args)

    def test_dry_run_delegates_without_writing_or_validating_completion(self):
        args = deepcopy(self.args)
        args.dry_run = True
        before = {str(p.relative_to(self.f.output)): p.read_bytes()
                  for p in self.f.output.rglob('*') if p.is_file()}
        preview = {'ready_for_inference': True, 'scope': 'test'}
        with patch.object(topic, 'run', return_value=preview) as run, \
                patch.object(topic, 'load_recipe', return_value=(self.config, self.identity)), \
                patch.object(selected, 'verify_selected_scope', side_effect=AssertionError('No completion checks')):
            result = selected.run(args)
        self.assertEqual(result, preview)
        run.assert_called_once()
        after = {str(p.relative_to(self.f.output)): p.read_bytes()
                 for p in self.f.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_limit_sessions_retains_incomplete_smoke_semantics(self):
        self.args.limit_sessions = 1
        for bank in self.banks['test']:
            self.mutate("UPDATE sessions SET status='pending',directory=NULL WHERE bank_id=?", (bank['bank_id'],))
        result = self.invoke(pool_factory=Pool().factory, embedding_factory=Encoder().factory)
        self.assertEqual(result['execution']['run_status'], 'limited')
        self.assertTrue(result['execution']['selection_limited'])
        self.assertNotIn('selected_scope', result)
        self.assertGreater(result['states']['pending'], 0)


if __name__ == '__main__':
    unittest.main()
