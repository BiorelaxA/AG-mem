"""Parent timeouts reuse only child outputs with source-bound validation proof."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agmem import repair_pending_parents as pending
from agmem.accept_fallback_length_exceptions import ApprovedWriter
from agmem.build_adjacent_directories import AdjacentBuildStore
from agmem.core import InfrastructureError,canonical,digest
from agmem.hierarchical.writer import writer_settings
from tests import test_adjacent_writer as fixture
from tests.test_adjacent_directory_pipeline import Pool


class PendingParentTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.source=self.root/'base';self.output=self.root/'completed'
        self.session=fixture.session(3)
        self.settings=writer_settings({'context_tokens':8192,'output_tokens':512,'max_output_tokens':1024,
                                      'safety_margin':16,'parent_content_tokens':500,'child_content_tokens':500,'title_tokens':30})
        self.writer=ApprovedWriter(fixture.TOKENIZER,self.settings,None);self.writer.prepare(self.session)
        self.child={'start_block':1,'end_block':3,'title':'Topic','content':'All relevant facts.','entities':[]}
        self.bank={'bank_id':self.session['bank_id'],'dataset':'split_longmemeval_s',
                   'query_memberships':[{'sample_id':'split_longmemeval_s/train/000000','split':'train','row_index':0}],
                   'source_map':[{'bank_id':self.session['bank_id'],'parent_id':self.session['parent_id']}],
                   'issues':[]}
        self.identity={'writer':{'model':'fixture','model_revision':'revision'},'tokenizer_id':'fixture-tokenizer'}
        self.config={'model_path':'fixture','output_root':str(self.output),'workers':2,'writer':self.settings,
                     'datasets':[{'name':'split_longmemeval_s'}],'service':{}}
        store=AdjacentBuildStore(self.source,self.identity);store.add_bank(self.bank);store.add_session(self.session);store.close()
        (self.source/'.build.lock').touch()

    def events(self):
        entry={k:self.child[k] for k in ('title','content','entities')}
        response={'text':canonical(entry),'finish_reason':'stop'}
        cr={'messages':self.writer._entry_messages(self.writer._raw_payload(1,3,'child',[]),'child')}
        pr={'messages':self.writer._entry_messages(self.writer._raw_payload(1,3,'parent',[self.child]),'parent')}
        return [{'event':'writer_request','stage':'adjacent_child','request':cr,'request_sha256':digest(cr)},
                {'event':'writer_response','stage':'adjacent_child','request_sha256':digest(cr),'response':response},
                {'event':'writer_validated','stage':'adjacent_child','status':'complete',
                 'response_sha256':digest(response),'messages_sha256':digest(cr['messages'])},
                {'event':'writer_request','stage':'adjacent_parent','request':pr,'request_sha256':digest(pr)}]

    def test_validated_children_are_recovered_without_generation(self):
        children,proof=pending.recovered_children(self.events(),self.writer)
        self.assertEqual(children,[self.child]);self.assertEqual(len(proof['children']),1)

    def test_child_without_successful_validation_is_rejected(self):
        events=[e for e in self.events() if e['event']!='writer_validated']
        with self.assertRaisesRegex(ValueError,'lacks matching successful validation'):
            pending.recovered_children(events,self.writer)

    def test_truncated_child_response_is_rejected(self):
        events=self.events();events[1]['response']['finish_reason']='length'
        with self.assertRaises(ValueError):pending.recovered_children(events,self.writer)

    def test_incomplete_block_coverage_is_rejected(self):
        events=self.events();p=events[-1]
        data=json.loads(p['request']['messages'][-1]['content']);data['children'][0]['end_block']=2
        p['request']['messages'][-1]['content']=canonical(data);p['request_sha256']=digest(p['request'])
        with self.assertRaisesRegex(ValueError,'omit'):pending.recovered_children(events,self.writer)

    def test_parent_request_source_changes_are_rejected_even_with_recomputed_hash(self):
        events=self.events();p=events[-1]
        data=json.loads(p['request']['messages'][-1]['content']);data['blocks']=[]
        p['request']['messages'][-1]['content']=canonical(data);p['request_sha256']=digest(p['request'])
        with self.assertRaisesRegex(ValueError,'source differs'):pending.recovered_children(events,self.writer)

    def invoke(self,pool=None,dry_run=False):
        selected=[{'bank':self.bank,'session':self.session,'children':[self.child],
                   'archived_root':'fixture','archived_bank_id':'fixture','proof':{'validated':True}}]
        args=Namespace(config='fixture',workers=2,output_root=None,source_root=str(self.source),
                       datasets=['split_longmemeval_s'],splits=['train'],base_urls=['http://fixture/v1'],
                       base_url=None,limit_sessions=None,dry_run=dry_run)
        pool=pool or Pool()
        with patch.object(pending.fallback.topic,'load_recipe',return_value=(self.config,self.identity)), \
                patch.object(pending,'read_candidates',return_value=(self.identity,selected)),patch.object(pending,'announce'):
            return pending.run(args,tokenizer=fixture.TOKENIZER,pool_factory=pool.factory)

    def test_only_parent_is_generated_and_resume_preserves_source(self):
        before={p.name:p.read_bytes() for p in self.source.iterdir() if p.is_file()}
        pool=Pool();report=self.invoke(pool)
        self.assertEqual(report['completed'],1);self.assertEqual(report['remaining'],0)
        self.assertEqual([r[2] for r in pool.calls],['adjacent_parent'])
        p=json.loads((self.output/'pending_parent_patches.jsonl').read_text())
        restored=pending.apply_pending_parent_patch(None,p,self.session,self.identity)
        self.assertEqual(restored['children'][0]['content'],self.child['content'])
        self.assertEqual(self.invoke(Pool(forbidden=True))['remaining'],0)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.source.iterdir() if p.is_file()})
        with self.assertRaises(ValueError):pending.apply_pending_parent_patch(restored,p,self.session,self.identity)

    def test_failure_is_not_marked_complete(self):
        with self.assertRaises(InfrastructureError):self.invoke(Pool(fail_parent=True))
        self.assertFalse((self.output/'complete.json').exists())

    def test_dry_run_has_no_generated_output(self):
        self.invoke(Pool(forbidden=True),dry_run=True)
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
