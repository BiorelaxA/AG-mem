"""Migrated source paths and reinstalled models retain honest old directory provenance."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import canonical, digest, file_digest
from agmem import reuse_topic_event_directories as mixed
from tests.test_reuse_topic_event_directories import TopicEventReuseTests


class MigratedTopicEventReuseTests(unittest.TestCase):
    def setUp(self):
        self.owner = TopicEventReuseTests(methodName='runTest')
        self.owner.setUp()
        self.addCleanup(self.owner.doCleanups)
        self.f = self.owner.fixture
        f = self.f
        self.parent_manifest = f.old_data / 'manifest.json'
        self.parent_manifest.write_text(canonical({'status': 'complete', 'fixture': 'original input'}))
        self.update_manifest()

    def update_manifest(self):
        f = self.f
        (f.new_data / 'manifest.json').write_text(canonical({
            'parent_manifest': str(self.parent_manifest),
            'parent_manifest_sha256': file_digest(self.parent_manifest),
            'row_mapping': {'sha256': file_digest(f.mapping), 'rows': len(f.mapping_rows)}}))

    def migrate_addresses(self):
        f = self.f
        offline = f.home / 'unmounted-old-host'
        replacements = {}
        for item in f.old_identity['datasets']:
            item['spec']['input_root'] = str(offline)
            for row in item['files']:
                replacements[row['path']] = str(offline / item['spec']['name'] / Path(row['path']).name)
                row['path'] = replacements[row['path']]
        with f.db(f.source) as db:
            db.execute("UPDATE metadata SET value=? WHERE key='identity'", (canonical(f.old_identity),))
            for bid, payload in db.execute('SELECT bank_id,payload FROM banks').fetchall():
                bank = json.loads(payload)
                for row in bank['source_map']:
                    row['file'] = replacements[row['file']]
                db.execute('UPDATE banks SET payload=? WHERE bank_id=?', (canonical(bank), bid))

    def reinstall_models(self):
        f = self.f
        f.config['allow_migrated_source_identity'] = True
        f.new_identity['writer']['model_revision'] = 'new-host-writer-fingerprint'
        f.new_identity['tokenizer_id'] = 'new-host-tokenizer-fingerprint'
        f.new_identity['embedding']['model_revision'] = 'new-host-embedding-fingerprint'

    def test_migrated_parquet_and_models_preserve_source_directories_without_old_cache(self):
        f = self.f
        self.migrate_addresses()
        self.reinstall_models()
        before = file_digest(f.source / 'build_journal.sqlite3')
        result = self.owner.run_import()
        self.assertEqual(result['counters']['imported_complete'], 2)
        self.assertEqual(result['counters']['imported_fallback'], 1)
        self.assertEqual(before, file_digest(f.source / 'build_journal.sqlite3'))
        with f.db(f.target) as db:
            policy = json.loads(db.execute('SELECT value FROM metadata WHERE key=?', (mixed.POLICY_KEY,)).fetchone()[0])
            self.assertTrue(policy['allow_migrated_source_identity'])
            self.assertEqual(policy['source_construction_identity'], f.old_identity)
            self.assertEqual(policy['target_model_provenance']['writer'], f.new_identity['writer'])
            self.assertIn('not_asserted_equal', policy['model_fingerprint_relation'])
            self.assertEqual(db.execute('SELECT count(*) FROM segment_cache').fetchone()[0], 0)
            for (encoded,) in db.execute('SELECT directory FROM sessions WHERE directory IS NOT NULL'):
                directory = json.loads(encoded)
                self.assertEqual(directory['build']['writer_revision'], f.old_identity['writer']['model_revision'])
                self.assertEqual(directory['source']['tokenizer_id'], f.old_identity['tokenizer_id'])
            for (encoded,) in db.execute('SELECT payload FROM banks'):
                bank = json.loads(encoded)
                self.assertTrue(all(str(f.new_data) in r['file'] for r in bank['source_map']))

    def test_relocated_source_change_is_rejected_before_creating_target(self):
        f = self.f
        self.migrate_addresses()
        path = f.old_data / f.source_name / 'train.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]['question'] = 'changed source question'
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        with self.assertRaisesRegex(ValueError, 'Source parquet changed'):
            self.owner.run_import()
        self.assertFalse(f.target.exists())

    def test_parent_manifest_change_blocks_relocation(self):
        f = self.f
        self.migrate_addresses()
        self.parent_manifest.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'parent manifest changed'):
            self.owner.run_import()
        self.assertFalse(f.target.exists())

    def test_migration_option_does_not_allow_other_model_or_semantic_changes(self):
        f = self.f
        self.reinstall_models()
        for section, key, value in [('writer', 'model', 'different-family'),
                                    ('writer_settings', 'child_content_tokens', 500),
                                    ('prompt_hashes', 'child.txt', 'different-prompt'),
                                    ('implementation_hashes', 'build_adjacent_directories.py', 'different-engine')]:
            changed = deepcopy(f.new_identity)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'semantics'):
                mixed.compatible_identity(f.old_identity, changed, allow_migrated_source_identity=True)

    def test_train_to_validation_move_rebinds_membership_and_keeps_summary(self):
        f = self.f
        rows = {'train': [f.old_rows['test'][0]], 'validation': [f.old_rows['train'][0]],
                'test': [f.old_rows['test'][1]]}
        schema = pq.read_schema(f.new_data / f.source_name / 'train.parquet')
        for split, values in rows.items():
            pq.write_table(pa.Table.from_pylist(values, schema=schema), f.new_data / f.source_name / f'{split}.parquet')
        for file in f.new_identity['datasets'][0]['files']:
            file['sha256'] = file_digest(file['path'])
        mapping_by_id = {row['question_id']: row for row in f.mapping_rows}
        f.mapping_rows = []
        for split, values in rows.items():
            sha = file_digest(f.new_data / f.source_name / f'{split}.parquet')
            for index, raw in enumerate(values):
                row = {**mapping_by_id[raw['question_id']], 'new_split': split, 'new_row_index': index,
                       'new_file_sha256': sha,
                       'new_bank_id': 'bank_' + digest([f.source_name, split, sha, index])[:24]}
                f.mapping_rows.append(row)
        f.mapping.write_text(''.join(canonical(row) + '\n' for row in f.mapping_rows))
        self.update_manifest()
        self.migrate_addresses()
        self.owner.run_import()
        moved = next(m for m in f.mapping_rows if m['new_split'] == 'validation')
        with f.db(f.target) as db:
            bank = json.loads(db.execute('SELECT payload FROM banks WHERE bank_id=?', (moved['new_bank_id'],)).fetchone()[0])
            self.assertEqual(bank['query_memberships'][0]['split'], 'validation')
            self.assertEqual(bank['query_memberships'][0]['original_question_id'], moved['question_id'])
            self.assertEqual(db.execute('SELECT count(*) FROM sessions WHERE bank_id=? AND status=?',
                                       (moved['new_bank_id'], 'complete')).fetchone()[0], 2)

    def test_new_m_dataset_with_no_source_banks_is_marked_fresh_not_reused(self):
        f = self.f
        source = 'split_longmemeval_m'
        schema = pq.read_schema(f.old_data / f.source_name / 'train.parquet')
        for root, rows in [(f.old_data, f.old_rows), (f.new_data, f.new_rows)]:
            (root / source).mkdir()
            for split, values in rows.items():
                pq.write_table(pa.Table.from_pylist(values, schema=schema), root / source / f'{split}.parquet')
        spec = {**f.new_identity['datasets'][0]['spec'], 'name': source}
        f.new_identity['datasets'].append({'spec': spec, 'files': [
            {'split': split, 'path': str(f.new_data / source / f'{split}.parquet'),
             'sha256': file_digest(f.new_data / source / f'{split}.parquet')}
            for split in ('train', 'validation', 'test')]})
        f.config['datasets'].append(spec)
        new_rows = []
        for row in f.mapping_rows:
            old_sha = file_digest(f.old_data / source / f"{row['old_split']}.parquet")
            new_sha = file_digest(f.new_data / source / f"{row['new_split']}.parquet")
            new_rows.append({**row, 'source': source, 'old_file_sha256': old_sha, 'new_file_sha256': new_sha,
                            'old_bank_id': 'bank_' + digest([source, row['old_split'], old_sha, row['old_row_index']])[:24],
                            'new_bank_id': 'bank_' + digest([source, row['new_split'], new_sha, row['new_row_index']])[:24]})
        f.mapping_rows.extend(new_rows)
        f.mapping.write_text(''.join(canonical(row) + '\n' for row in f.mapping_rows))
        self.update_manifest()
        report = self.owner.run_import(datasets=[f.source_name, source])
        self.assertEqual(report['selected_queries'], 6)
        self.assertEqual(report['counters']['unregistered'], 8)
        self.assertEqual(report['counters']['banks_reused'], 2)
        self.assertFalse(mixed.validate_mixed_policy(f.config, f.new_identity, datasets=[source])['requires_import'])


if __name__ == '__main__':
    unittest.main()
