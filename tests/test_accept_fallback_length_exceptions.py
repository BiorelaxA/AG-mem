"""Approved lengths reuse saved text without completing missing parent summaries."""
from copy import deepcopy
import json
import sqlite3
import unittest
from unittest.mock import patch

from agmem import accept_fallback_length_exceptions as accepted
from agmem import repair_topic_event_fallbacks as repair
from agmem.core import InfrastructureError, canonical, digest
from agmem.hierarchical.binding import directory_version
from tests import test_repair_topic_event_fallbacks as fixtures
from tests.test_adjacent_directory_pipeline import Pool


class LengthAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.PlaceholderRepairTests(methodName='runTest')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.output=self.f.f.home/'accepted'
        self.config=deepcopy(self.f.config);self.identity=deepcopy(self.f.identity)
        self.config['output_root']=str(self.output)
        self.config['writer'].update(parent_content_tokens=500,child_content_tokens=500,title_tokens=30)
        self.identity['writer_settings']=deepcopy(self.config['writer'])

    def args(self, **changes):
        args=self.f.args()
        args.previous_repairs=str(self.f.output)
        args.prepare_only=False
        for k,v in changes.items():setattr(args,k,v)
        return args

    def prepare(self, **changes):
        with patch.object(repair.topic,'load_recipe',return_value=(self.config,self.identity)), \
                patch.object(accepted,'announce'):
            return accepted.prepare(self.args(**changes),tokenizer=self.f.f.tokenizer)

    def source_row(self):
        with self.f.f.db() as db:
            raw=db.execute("select snapshot,directory from sessions where parent_id='p0002'").fetchone()
        return json.loads(raw[0]),json.loads(raw[1])

    def failed_previous(self, child=False):
        if child:
            with self.f.f.db() as db:
                bid,raw=db.execute("select bank_id,directory from sessions where parent_id='p0002'").fetchone()
                d=json.loads(raw);d['children'][0].update(title='Original records',content=repair.PLACEHOLDER,entities=[])
                d['entities']=[];d['directory_version']=directory_version(d)
                db.execute('update sessions set directory=? where bank_id=? and parent_id=?',(canonical(d),bid,'p0002'))
        with self.assertRaises(InfrastructureError):
            self.f.invoke(Pool(forbidden=True))

    def saved_response(self, kind='parent', *, words=420, title_words=3, finish='stop', tamper=False):
        session,d=self.source_row()
        writer=accepted.ApprovedWriter(self.f.f.tokenizer,self.config['writer'],None)
        writer.prepare(session)
        inputs=[{'start_block':c['source']['start_block'],'end_block':c['source']['end_block'],
                 **{k:c[k] for k in ('title','content','entities')}} for c in d['children']]
        start,end=(1,len(session['blocks'])) if kind=='parent' else (
            d['children'][0]['source']['start_block'],d['children'][0]['source']['end_block'])
        payload=writer._raw_payload(start,end,kind,inputs if kind=='parent' else [])
        payload['limits']['parent_content_tokens']=400
        payload['limits']['child_content_tokens']=300
        request={'messages':writer._entry_messages(payload,kind),'max_tokens':4096,'temperature':0}
        value={'title':' '.join(['topic']*title_words),'content':' '.join(['fact']*words)}
        if kind=='child':value['entities']=[]
        sha='bad' if tamper else digest(request)
        common={'bank_id':session['bank_id'],'parent_id':session['parent_id'],
                'stage':'adjacent_'+kind+'_repair','request_sha256':sha}
        with sqlite3.connect(self.f.output/'build_journal.sqlite3') as db:
            for event in ({**common,'event':'writer_request','request':request},
                          {**common,'event':'writer_response','response':{'finish_reason':finish,'text':canonical(value)}}):
                db.execute('insert into writer_traces(payload) values(?)',(canonical(event),))
        return value

    def test_successful_previous_result_is_preserved_exactly(self):
        self.f.invoke()
        before=(self.f.output/'repair_patches.jsonl').read_bytes()
        _,_,report=self.prepare()
        after=json.loads((self.output/'repair_patches.jsonl').read_text())
        self.assertEqual(after['directory'],json.loads(before)['directory'])
        self.assertEqual((report['repaired'],report['remaining']),(1,0))
        self.assertEqual(before,(self.f.output/'repair_patches.jsonl').read_bytes())
        self.assertEqual(self.prepare()[2]['repaired'],1)

    def test_overlong_parent_is_recovered_without_requests(self):
        self.failed_previous();entry=self.saved_response(words=472)
        _,_,report=self.prepare()
        row=json.loads((self.output/'repair_patches.jsonl').read_text())
        self.assertEqual(row['directory']['content'],entry['content'])
        self.assertEqual(report['remaining'],0)
        self.assertTrue((self.output/'complete.json').exists())

    def test_parent_title_27_is_allowed(self):
        self.failed_previous();self.saved_response(words=50,title_words=27)
        self.assertEqual(self.prepare()[2]['repaired'],1)

    def test_child_only_does_not_complete_session_and_parent_only_is_generated_later(self):
        self.failed_previous(child=True);entry=self.saved_response(kind='child',words=417)
        _,components,report=self.prepare()
        self.assertEqual((report['repaired'],report['remaining']),(0,1))
        self.assertFalse((self.output/'complete.json').exists())
        indexed={(r['bank_id'],r['parent_id'],r['child_id']):r for r in components}
        pool=Pool()
        with patch.object(repair.topic,'load_recipe',return_value=(self.config,self.identity)),patch.object(repair,'announce'):
            report=repair.run(self.args(),tokenizer=self.f.f.tokenizer,pool_factory=pool.factory,
                              writer_factory=lambda *a,**kw: accepted.ApprovedWriter(*a,components=indexed,**kw))
        self.assertEqual(report['remaining'],0)
        self.assertEqual([c[2] for c in pool.calls],['adjacent_parent'])
        row=json.loads((self.output/'repair_patches.jsonl').read_text())
        self.assertEqual(row['directory']['children'][0]['content'],entry['content'])

    def test_truncated_saved_response_is_rejected(self):
        self.failed_previous();self.saved_response(finish='length')
        with self.assertRaisesRegex(ValueError,'stop responses'):self.prepare()

    def test_request_hash_tampering_is_rejected(self):
        self.failed_previous();self.saved_response(tamper=True)
        with self.assertRaisesRegex(ValueError,'request hash'):self.prepare()

    def test_above_new_budget_remains_invalid(self):
        self.failed_previous();self.saved_response(words=501)
        with self.assertRaisesRegex(ValueError,'exceeds 500'):self.prepare()

    def test_child_title_limit_remains_24(self):
        self.failed_previous(child=True);self.saved_response(kind='child',words=417,title_words=27)
        with self.assertRaisesRegex(ValueError,'child.title exceeds 24'):self.prepare()

    def test_dryrun_does_not_create_new_output(self):
        self.failed_previous();self.saved_response()
        _,_,report=self.prepare(dry_run=True)
        self.assertEqual(report['accepted_saved_parents'],1)
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
