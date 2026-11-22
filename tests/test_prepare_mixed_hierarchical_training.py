"""Mixed snapshots preserve every query and apply only verified complete repairs."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import atomic_json,canonical,digest,file_digest
from agmem.data import ADAPTER_VERSION,adapt_row,clean_history
from agmem.hierarchical.adapters import iter_banks
from agmem.hierarchical.binding import bind_directory,directory_version
from agmem.hierarchical.storage import BuildStore
from agmem.hierarchical_training_store import HierarchicalTrainingStore
from agmem.prepare_hierarchical_training import prepare_frozen_store
from agmem.prepare_mixed_hierarchical_training import prepare_mixed
from agmem.repair_topic_event_fallbacks import VERSION as FALLBACK_PROTOCOL,PLACEHOLDER
from agmem.repair_pending_parents import PROTOCOL as PENDING_PROTOCOL
from agmem.store import Store


class MixedFrozenTrainingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.s='split_longmemeval_s'; self.output=self.root/'mixed'
        self.inputs={}; self.specs=[]; self.journals={}; self.banks={}
        self.identity={'writer':{'model':'fixture','model_revision':'r'},'tokenizer_id':'fixture'}
        for name in ('locomo',self.s):
            raw=self.root/(name+'_inputs'); self.inputs[name]=raw
            raw_rows={split:[self.row(f'{name}-{split}-{i}',f'{name}-{split}-{i}') for i in range(n)]
                      for split,n in (('train',2),('validation',1),('test',1))}
            if name=='locomo':
                for split in ('validation','test'):
                    raw_rows[split][0]['haystack_sessions']=deepcopy(raw_rows['train'][0]['haystack_sessions'])
                    raw_rows[split][0]['haystack_session_ids']=deepcopy(raw_rows['train'][0]['haystack_session_ids'])
            files={}
            for split,rows in raw_rows.items():
                p=raw/name/f'{split}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows),p)
                files[split]={'sha256':file_digest(p),'rows':len(rows)}
            atomic_json(raw/'manifest.json',{'sources':[{'source':name,'output_files':files}]})
            build=self.root/(name+'_build'); journal=BuildStore(build,self.identity)
            self.journals[name]=journal;self.addCleanup(journal.close)
            banks=list(iter_banks({'name':name,'format':'locomo_parquet' if name=='locomo' else 'longmemeval_parquet',
                                  'input_root':str(raw),'splits':['train','validation','test']},count_tokens=len))
            self.banks[name]=banks
            for bank in banks:
                journal.add_bank(bank)
                for session in bank['sessions']:
                    journal.add_session(session)
                    member=bank['query_memberships'][0]
                    pending=name==self.s and member['split']=='train' and member['row_index']==1
                    bad=name==self.s and (member['split']=='test' or member['split']=='train' and member['row_index']==0)
                    if not pending:journal.commit(session,self.directory(session,bad),{})
            if name=='locomo':
                prepared=self.root/'locomo_prepared'; store=Store(prepared)
                for split in ('train','validation'):
                    for i,row in enumerate(raw_rows[split]):
                        mid=store.add_memory(clean_history(row));store.add_example(adapt_row(row,name,split,i,mid))
                store.set_metadata('prepare_identity',{'input_manifest_sha256':file_digest(raw/'manifest.json'),
                    'sources':[name],'splits':['train','validation'],'limit_per_source_split':None,
                    'adapter_version':ADAPTER_VERSION,'storage_format':Store.FORMAT})
                store.set_metadata('prepare_complete',True);store.close()
                frozen=self.root/'locomo_frozen'
                prepare_frozen_store({'data':{'directory_protocol':'hierarchical_v1','layout':'combined',
                    'sources':['locomo'],'splits':['train','validation'],'prepared_root':str(frozen)}},prepared,build)
                self.specs.append({'name':name,'input_root':str(raw),'frozen_root':str(frozen)})
            else:
                self.fallback=self.root/'fallback';self.pending=self.root/'pending'
                self.overlays(build)
                self.specs.append({'name':name,'input_root':str(raw),'directory_root':str(build),
                    'fallback_repairs':str(self.fallback),'pending_repairs':str(self.pending)})
        self.config={'data':{'directory_protocol':'hierarchical_v1','layout':'combined',
            'sources':['locomo',self.s],'splits':['train','validation'],'prepared_root':str(self.output),
            'source_specs':self.specs}}

    @staticmethod
    def row(qid,text):
        return {'question_id':qid,'question_type':'multi-session','question':'Question '+qid,'answer':'Answer '+qid,
                'question_date':'2024-03-02','haystack_sessions':[[{'role':'user','content':text,'dia_id':'a'},
                                                             {'role':'assistant','content':'response '+text,'dia_id':'b'}]],
                'haystack_session_ids':['history-'+text],'haystack_dates':['2024-01-01']}

    def directory(self,session,bad=False):
        return bind_directory(session,{'status':'fallback' if bad else 'complete','mode':'single_session',
            'parent':{'title':'A parent','content':PLACEHOLDER if bad else 'A complete summary'},
            'children':[{'end_block':len(session['blocks']),'title':'Child','content':'Child facts','entities':[]}]},
            self.identity['writer'],'fixture',count_tokens=len,prompt_version='directory-writer.adjacent.v1')

    def overlays(self,build):
        fallback=[];pending=[];fs=[];ps=[]
        for bank in self.banks[self.s]:
            session=bank['sessions'][0];mid=bank['query_memberships'][0]
            directory=self.directory(session)
            if mid['split']=='validation':continue
            key={'bank_id':session['bank_id'],'parent_id':session['parent_id'],'source_hash':session['source_hash'],
                 'source_identity_sha256':digest(self.identity),'after_directory_sha256':digest(directory),'directory':directory}
            if mid['split']=='train' and mid['row_index']==1:
                child=directory['children'][0]
                pending.append({'protocol':PENDING_PROTOCOL,**key,'before_status':'pending','before_directory_sha256':None,
                    'recovered_children':[{'start_block':child['source']['start_block'],'end_block':child['source']['end_block'],
                                          **{k:child[k] for k in ('title','content','entities')}}]})
                ps.append({'session':session})
            else:
                before=self.directory(session,True)
                fallback.append({'protocol':FALLBACK_PROTOCOL,**key,'before_status':'fallback',
                    'before_directory_sha256':digest(before),'before_directory_version':before['directory_version']})
                fs.append({'bank_id':session['bank_id'],'parent_id':session['parent_id']})
        for root,patches,selected,manifest_name,patch_name,key in (
            (self.fallback,fallback,fs,'repair_manifest.json','repair_patches.jsonl','selection_sha256'),
            (self.pending,pending,ps,'pending_manifest.json','pending_parent_patches.jsonl','manifest_sha256')):
            root.mkdir();manifest={'source_root':str(build.resolve()),'source_identity_sha256':digest(self.identity),'selected':selected}
            atomic_json(root/manifest_name,manifest);atomic_json(root/'run_manifest.json',{'repair':'fixture'})
            (root/patch_name).write_text(''.join(canonical(p)+'\n' for p in patches))
            atomic_json(root/'complete.json',{key:digest(manifest),'n':len(patches),'patches_sha256':file_digest(root/patch_name)})

    def test_all_queries_preserved_repairs_applied_tests_excluded_and_resume(self):
        report=prepare_mixed(self.config)
        self.assertEqual(report['examples'],{'train':4,'validation':2})
        self.assertEqual(report['banks'],5)
        self.assertEqual(report['by_source'][self.s]['repairs_applied'],{'train/fallback':1,'train/pending':1})
        store=HierarchicalTrainingStore(self.output)
        self.addCleanup(store.close)
        self.assertEqual({e.source for e in store.examples('train')},{'locomo',self.s})
        self.assertNotIn('test',store.snapshot['splits'])
        self.assertFalse(any('/test/' in q for q in store.snapshot['query_to_bank']))
        self.assertTrue(all(m['split']!='test' for bid in store.snapshot['banks']
            for m in store.hierarchies.get((bid,))['query_memberships']))
        for memory in store.memory_ids():
            directories,_=store.hierarchical_bank(memory,report['directory_version'])
            self.assertTrue(all(d['build']['status']=='complete' for d in directories))
        self.assertTrue(prepare_mixed(self.config)['reused'])

    def test_missing_repair_cannot_silently_drop_question(self):
        del self.config['data']['source_specs'][1]['pending_repairs']
        with self.assertRaisesRegex(ValueError,'no complete directory'):prepare_mixed(self.config)
        self.assertFalse(self.output.exists())

    def test_patch_drift_is_rejected_before_freeze(self):
        path=self.fallback/'repair_patches.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'overlay'):prepare_mixed(self.config)
        self.assertFalse(self.output.exists())

    def test_check_only_writes_no_output_or_source_indices(self):
        root=Path(self.specs[0]['frozen_root']);before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in root.iterdir()}
        report=prepare_mixed(self.config,check_only=True)
        self.assertEqual(report['status'],'inputs_verified');self.assertFalse(self.output.exists())
        self.assertEqual(before,{p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_selected_directory_drift_invalidates_existing_snapshot(self):
        prepare_mixed(self.config)
        bank=next(b for b in self.banks[self.s] if b['query_memberships'][0]['split']=='validation')
        session=bank['sessions'][0];changed=self.directory(session);changed['title']='New source title'
        changed['directory_version']=directory_version(changed)
        self.journals[self.s].commit(session,changed,{})
        with self.assertRaisesRegex(ValueError,'different inputs'):prepare_mixed(self.config)

    def test_test_split_is_not_supported_for_training_preparation(self):
        self.config['data']['splits']=['train','test']
        with self.assertRaisesRegex(ValueError,'train/validation only'):prepare_mixed(self.config)

    def use_locomo_journal(self):
        """Model a new directory build while leaving the old frozen copy intact."""
        root=self.root/'locomo_build'
        construction=deepcopy(self.identity)
        manifest=json.loads((self.inputs['locomo']/'manifest.json').read_text())
        construction['protocol']='adjacent_topic_event_context3_v1'
        construction['datasets']=[{'spec':{'name':'locomo','format':'locomo_parquet'},
            'files':[{'split':split,'path':str(self.inputs['locomo']/'locomo'/f'{split}.parquet'),
                      'sha256':manifest['sources'][0]['output_files'][split]['sha256']}
                     for split in ('train','validation','test')]}]
        journal=self.journals['locomo']
        with journal.db:
            journal.db.execute("UPDATE metadata SET value=? WHERE key='identity'",(canonical(construction),))
        atomic_json(root/'run_manifest.json',construction)
        for bank in self.banks['locomo']:
            for session in bank['sessions']:
                result={'status':'complete','mode':'single_session',
                    'parent':{'title':'New parent','content':'New context3 directory'},
                    'children':[{'end_block':1,'title':'First','content':'New first segment','entities':[]},
                                {'end_block':2,'title':'Second','content':'New second segment','entities':[]}]}
                directory=bind_directory(session,result,self.identity['writer'],'fixture',
                    count_tokens=len,prompt_version='directory-writer.adjacent.v1')
                journal.commit(session,directory,{})
        self.specs[0]={'name':'locomo','input_root':str(self.inputs['locomo']),'directory_root':str(root)}
        return root,construction

    def test_new_locomo_journal_reuses_histories_and_uses_new_segments(self):
        old_root=Path(self.specs[0]['frozen_root'])
        old_manifest=(old_root/'hierarchical_manifest.json').read_bytes()
        source,_=self.use_locomo_journal()
        before={p.name:(file_digest(p),p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
        report=prepare_mixed(self.config)
        self.assertEqual(report['examples'],{'train':4,'validation':2})
        self.assertEqual(report['by_source']['locomo']['banks'],2)
        self.assertEqual(report['by_source']['locomo']['sessions'],2)
        self.assertEqual(report['banks'],5)
        self.assertEqual(report['by_source'][self.s]['repairs_applied'],{'train/fallback':1,'train/pending':1})
        frozen=HierarchicalTrainingStore(self.output);self.addCleanup(frozen.close)
        query_banks=frozen.snapshot['query_to_bank']
        self.assertEqual(query_banks['locomo/train/000000'],query_banks['locomo/validation/000000'])
        self.assertFalse(any('/test/' in sid for sid in query_banks))
        for bid,info in frozen.snapshot['banks'].items():
            bank=frozen.hierarchies.get((bid,))
            self.assertTrue(all(m['split'] in ('train','validation') for m in bank['query_memberships']))
            if info['dataset']=='locomo':
                self.assertEqual(len(bank['directories'][0]['children']),2)
                self.assertEqual(bank['directories'][0]['content'],'New context3 directory')
        self.assertEqual(frozen.snapshot['construction_identity']['by_source']['locomo']['protocol'],
                         'adjacent_topic_event_context3_v1')
        self.assertTrue(prepare_mixed(self.config)['reused'])
        self.assertEqual(old_manifest,(old_root/'hierarchical_manifest.json').read_bytes())
        self.assertEqual(before,{p.name:(file_digest(p),p.stat().st_mtime_ns)
                                 for p in source.iterdir() if p.is_file()})

    def test_new_locomo_check_only_leaves_source_and_output_untouched(self):
        source,_=self.use_locomo_journal()
        before={p.name:(file_digest(p),p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
        report=prepare_mixed(self.config,check_only=True)
        self.assertEqual(report['status'],'inputs_verified')
        self.assertEqual(report['by_source']['locomo']['queries'],{'train':2,'validation':1})
        self.assertFalse(self.output.exists())
        self.assertEqual(before,{p.name:(file_digest(p),p.stat().st_mtime_ns)
                                 for p in source.iterdir() if p.is_file()})

    def test_new_locomo_rejects_changed_input_fingerprint(self):
        source,construction=self.use_locomo_journal()
        construction['datasets'][0]['files'][1]['sha256']='0'*64
        with self.journals['locomo'].db:
            self.journals['locomo'].db.execute("UPDATE metadata SET value=? WHERE key='identity'",
                                              (canonical(construction),))
        atomic_json(source/'run_manifest.json',construction)
        with self.assertRaisesRegex(ValueError,'input Parquet differs for validation'):
            prepare_mixed(self.config)
        self.assertFalse(self.output.exists())

    def test_new_locomo_rejects_changed_query_membership(self):
        self.use_locomo_journal()
        bank=deepcopy(next(b for b in self.banks['locomo']
                           if any(m['sample_id']=='locomo/train/000000' for m in b['query_memberships'])))
        member=next(m for m in bank['query_memberships'] if m['sample_id']=='locomo/train/000000')
        member['original_question_id']='foreign-query'
        journal=self.journals['locomo']
        with journal.db:
            journal.db.execute('UPDATE banks SET payload=? WHERE bank_id=?',
                               (canonical({k:v for k,v in bank.items() if k!='sessions'}),bank['bank_id']))
        with self.assertRaisesRegex(ValueError,'mismatched hierarchy membership'):
            prepare_mixed(self.config)
        self.assertFalse(self.output.exists())

    def test_new_locomo_cannot_overwrite_existing_old_frozen_mixture(self):
        prepare_mixed(self.config)
        path=self.output/'hierarchical_manifest.json';before=path.read_bytes()
        self.use_locomo_journal()
        with self.assertRaisesRegex(ValueError,'different inputs'):
            prepare_mixed(self.config)
        self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
