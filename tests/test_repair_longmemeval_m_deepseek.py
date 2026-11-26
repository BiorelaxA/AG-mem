import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agmem import repair_longmemeval_m_deepseek as repair
from agmem.core import canonical
from agmem.hierarchical.binding import bind_directory
from agmem.repair_topic_event_fallbacks import PLACEHOLDER, apply_repair_patch
from tests.test_adjacent_writer import Transport, WordTokenizer, session


class Tokenizer(WordTokenizer):
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class DeepSeekRepairTests(unittest.TestCase):
    def test_validation_retries_then_clips_only_if_still_over_budget(self):
        for kind, limit in [('child', 400), ('parent', 600)]:
            for retry_length in (limit - 5, limit + 15):
                with self.subTest(kind=kind, retry_length=retry_length):
                    def respond(client, data, budget, stage):
                        length = retry_length if 'repair' in data else limit + 25
                        entry = {'title': 'Facts', 'content': '文' * length}
                        if kind == 'child':
                            entry['entities'] = []
                        return {'text': canonical(entry), 'finish_reason': 'stop'}
                    transport = Transport(respond)
                    writer = repair.ValidationRepairWriter(Tokenizer(), dict(child_content_tokens=400,
                        parent_content_tokens=600, semantic_retries=2), transport)
                    result = writer._request_entry({}, kind, 'adjacent_' + kind)
                    self.assertEqual(result['content'], '文' * min(limit, retry_length))
                    self.assertEqual(len(transport.calls), 2)
                    self.assertIn(str(limit), transport.calls[-1][-1]['repair']['errors'][0])

    def test_repair_retry_truncate_preserve_and_resume(self):
        tokenizer = Tokenizer()
        transcript = session(4)
        old = bind_directory(transcript, {
            'parent': {'title': 'Original records', 'content': PLACEHOLDER},
            'children': [
                {'end_block': 2, 'title': 'First', 'content': 'Keep these facts.', 'entities': ['First']},
                {'end_block': 4, 'title': 'Original records', 'content': PLACEHOLDER, 'entities': []}],
            'status': 'fallback', 'mode': 'single_session'}, {'model': 'qwen', 'revision': '1'},
            'local', count_tokens=lambda s: len(s.split()))
        def respond(client, data, budget, stage):
            if stage == 'adjacent_child':
                return {'text': '{"title":"broken"', 'finish_reason': 'stop'}
            value = {'title': '长' * 40, 'content': '文' * 700}
            if stage == 'adjacent_child_repair':
                self.assertEqual([1, 2], [b['index'] for b in data['context_segments'][0]['blocks']])
                value['entities'] = [' X ', 'X', ' '] + [str(i) for i in range(20)]
            else:
                self.assertEqual(len(data['children'][1]['content']), 400)
            return {'text': canonical(value), 'finish_reason': 'stop'}
        transport = Transport(respond)
        pool = SimpleNamespace(for_session=lambda s: transport, close=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            output = source/'out'
            (source/'tokenizer.json').write_text('{}')
            with sqlite3.connect(source/'build_journal.sqlite3') as db:
                db.execute('create table metadata(key,value)')
                db.execute("insert into metadata values ('identity','{}')")
                db.execute('create table banks(bank_id,dataset,payload)')
                db.execute('insert into banks values (?,?,?)', (transcript['bank_id'], 'split_longmemeval_m',
                           canonical({'query_memberships': [{'split': 'train'}]})))
                db.execute('create table sessions(bank_id,parent_id,source_hash,snapshot,status,directory)')
                db.execute('insert into sessions values (?,?,?,?,?,?)', (transcript['bank_id'],
                           transcript['parent_id'], transcript['source_hash'], canonical(transcript),
                           'fallback', canonical(old)))
            args = SimpleNamespace(workers=2, dry_run=False, limit_sessions=None)
            settings = dict(child_content_tokens=400, parent_content_tokens=600, context_blocks=0)
            service = dict(base_url='https://api.deepseek.com', model='deepseek-flash')
            with patch.object(repair, 'load_tokenizer', return_value=tokenizer), \
                    patch.object(repair, 'DeepSeekWriterPool', return_value=pool) as factory:
                self.assertEqual(repair.run_source(source, output, args, settings, service, source), 0)
                path = output/'run_manifest.json'
                identity = json.loads(path.read_text())
                identity['implementation'] = repair.SUMMARY_ONLY_IMPLEMENTATION
                path.write_text(json.dumps(identity))
                self.assertEqual(repair.run_source(source, output, args, settings, service, source), 0)
                factory.assert_called_once()
            row = json.loads((output/'repair_patches.jsonl').read_text())
            result = apply_repair_patch(old, row, transcript, {})
            self.assertEqual(result['children'][0], old['children'][0])
            self.assertEqual(len(result['content']), 600)
            self.assertEqual(len(result['children'][1]['content']), 400)
            self.assertEqual(result['title'], '长' * 24)
            self.assertEqual(result['children'][1]['title'], '长' * 24)
            self.assertEqual(result['children'][1]['entities'], ['X'] + [str(i) for i in range(11)])
            self.assertEqual(row['repair_stats']['child_summaries_truncated'], 1)
            self.assertEqual(row['repair_stats']['parent_summaries_truncated'], 1)
            self.assertEqual([c[3] for c in transport.calls],
                             ['adjacent_child', 'adjacent_child_repair', 'adjacent_parent'])
            self.assertEqual(json.loads((output/'complete.json').read_text())['n'], 1)
