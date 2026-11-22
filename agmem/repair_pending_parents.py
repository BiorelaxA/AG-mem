"""Complete timed-out parents using their already validated child summaries.

Original pending rows and archived traces stay read-only. The small output is
an insertion overlay, distinct from patches replacing existing fallback rows.
"""
import argparse
from contextlib import closing, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import time

from . import repair_topic_event_fallbacks as fallback
from .accept_fallback_length_exceptions import ApprovedWriter
from .build_adjacent_directories import AdjacentBuildStore
from .core import InfrastructureError, atomic_json, canonical, digest, file_digest
from .execution import bounded_results
from .hierarchical.adjacent_writer import validate_spans
from .hierarchical.binding import bind_directory, validate_directory
from .hierarchical.client import WriterPool
from .hierarchical.records import validate_session
from .hierarchical.writer import token_count, validate_build_result
from .jobs import file_lock
from .pipeline import announce
from .policy import load_tokenizer
from .reuse_adjacent_directories import source_connection, source_lock, rebind_session


PROTOCOL='pending_parent_completion.v1'
DEFAULT_CONFIG='configs/repair_longmemeval_s_fallbacks_len500.yaml'
DEFAULT_OUTPUT='runs/parent_child_directories/longmemeval_s_31535150_pending_parents_20261114'


def verify_payload(payload, writer, kind, start, end, children):
    context=len(payload.get('context_blocks',[]))
    if context>writer.settings['context_blocks']:
        raise ValueError('Unexpected source context length')
    expected=writer._raw_payload(start,end,kind,children,context_count=context)
    for k in ('input_kind','scope','source_kind','session_metadata','headers','write_range',
              'blocks','context_blocks','children','output_language'):
        if payload.get(k)!=expected.get(k):
            raise ValueError('Archived request source differs: '+k)


def recovered_children(events, writer):
    requests={e['request_sha256']:e for e in events if e.get('event')=='writer_request'}
    parents=[e for e in events if e.get('event')=='writer_request' and e.get('stage')=='adjacent_parent']
    if not parents:
        raise ValueError('No saved full-session parent request')
    last=parents[-1]
    if digest(last['request'])!=last['request_sha256']:
        raise ValueError('Archived parent request hash changed')
    payload=json.loads(last['request']['messages'][-1]['content'])
    children=payload.get('children',[])
    validate_spans(children,len(writer.public['blocks']))
    verify_payload(payload,writer,'parent',1,len(writer.public['blocks']),children)
    proof=[]
    responses=[e for e in events if e.get('event')=='writer_response'
               and e.get('stage') in {'adjacent_child','adjacent_child_repair'}]
    validations={(e.get('response_sha256'),e.get('messages_sha256')) for e in events
                 if e.get('event')=='writer_validated' and e.get('status')=='complete'
                 and e.get('stage') in {'adjacent_child','adjacent_child_repair'}}
    for child in children:
        entry={k:child[k] for k in ('title','content','entities')}
        writer._validate_entry(entry,'child',{})
        if fallback.is_placeholder(entry):
            raise ValueError('A saved child is still a placeholder')
        matched=None
        for e in reversed(responses):
            response=e['response']; request=requests.get(e['request_sha256'],{}).get('request')
            if response.get('finish_reason')!='stop' or request is None:
                continue
            if json.loads(response['text'])!=entry:
                continue
            if digest(request)!=e['request_sha256']:
                raise ValueError('Archived child request hash changed')
            if (digest(response),digest(request['messages'])) not in validations:
                continue
            data=json.loads(request['messages'][-1]['content'])
            if data.get('write_range')!={'start':child['start_block'],'end':child['end_block']}:
                continue
            verify_payload(data,writer,'child',child['start_block'],child['end_block'],[])
            matched={'request_sha256':e['request_sha256'],'response_sha256':digest(response),
                     'start_block':child['start_block'],'end_block':child['end_block']}
            break
        if matched is None:
            raise ValueError('Saved child lacks matching successful validation and source evidence')
        proof.append(matched)
    return deepcopy(children),{'parent_request_sha256':last['request_sha256'],'children':proof}


def read_candidates(db, source, config, args, tokenizer, locks):
    datasets=args.datasets or ['split_longmemeval_s']; splits=args.splits or ['train']
    identity=json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
    if json.loads((source/'run_manifest.json').read_text())!=identity:
        raise ValueError('Base journal and manifest differ')
    selected=[]; sources={}
    for bid,pid,raw,payload,dataset,provenance in db.execute(
        "SELECT s.bank_id,s.parent_id,s.snapshot,b.payload,b.dataset,p.payload FROM sessions s "
        "JOIN banks b USING(bank_id) LEFT JOIN reuse_provenance p USING(bank_id,parent_id) "
        "WHERE s.status='pending' AND s.directory IS NULL ORDER BY s.bank_id,s.parent_id"):
        bank=json.loads(payload)
        if dataset not in datasets or not any(m['split'] in splits for m in bank['query_memberships']):
            continue
        if provenance is None:
            raise ValueError('Pending session has no archived generation provenance')
        prior=json.loads(provenance); session=json.loads(raw)
        if prior.get('source_status')!='failed' or 'ReadTimeout' not in prior.get('source_error',{}).get('reason',''):
            raise ValueError('This parent-only workflow requires a recorded Writer ReadTimeout')
        old_root=Path(prior['source_root']).resolve()
        if old_root not in sources:
            locks.enter_context(source_lock(old_root))
            sources[old_root]=locks.enter_context(closing(source_connection(old_root)))
        old=sources[old_root]
        original=old.execute('SELECT snapshot,status,directory FROM sessions WHERE bank_id=? AND parent_id=?',
                             (prior['source_bank_id'],pid)).fetchone()
        if original is None or original[1]!='failed' or original[2] is not None:
            raise ValueError('Archived session state changed')
        old_session=json.loads(original[0])
        rebound,_=rebind_session(old_session,None,bid,count_tokens=lambda t:token_count(tokenizer,t))
        if rebound!=session or old_session['source_hash']!=prior['source_hash_before']:
            raise ValueError('Recovered session differs from the current immutable source')
        events=[json.loads(row[0]) for row in old.execute(
            "SELECT payload FROM writer_traces WHERE json_extract(payload,'$.bank_id')=? "
            "AND json_extract(payload,'$.parent_id')=? ORDER BY seq",(prior['source_bank_id'],pid))]
        writer=ApprovedWriter(tokenizer,config['writer'],None); writer.prepare(session)
        children,proof=recovered_children(events,writer)
        selected.append({'bank':bank,'session':session,'children':children,
                         'archived_root':str(old_root),'archived_bank_id':prior['source_bank_id'],'proof':proof})
    return identity,selected


def validate_completed(session, children, directory, count_tokens=None):
    validate_directory(directory,session,count_tokens=count_tokens)
    if directory['build']['status']!='complete' or fallback.is_placeholder(directory):
        raise ValueError('Parent completion is not complete')
    if len(children)!=len(directory['children']):
        raise ValueError('Recovered child count changed')
    for i,(old,new) in enumerate(zip(children,directory['children']),1):
        if (new['id']!=f"{session['parent_id']}:c{i:04d}"
                or any(old[k]!=new[k] for k in ('title','content','entities'))
                or any(old[k]!=new['source'][k] for k in ('start_block','end_block'))):
            raise ValueError('Validated child text/ID/boundary changed')


def apply_pending_parent_patch(base_directory,patch,session,source_identity=None,source_status='pending'):
    if base_directory is not None or source_status!='pending' or patch.get('protocol')!=PROTOCOL:
        raise ValueError('Pending-parent patch only fills an absent pending directory')
    expected={'bank_id':session['bank_id'],'parent_id':session['parent_id'],
              'source_hash':session['source_hash'],'before_status':'pending','before_directory_sha256':None}
    if any(patch.get(k)!=v for k,v in expected.items()):
        raise ValueError('Pending-parent patch source mismatch')
    if source_identity is not None and patch.get('source_identity_sha256')!=digest(source_identity):
        raise ValueError('Pending-parent patch construction identity mismatch')
    d=patch['directory']
    if digest(d)!=patch['after_directory_sha256']:
        raise ValueError('Pending-parent patch directory hash changed')
    validate_completed(session,patch['recovered_children'],d)
    return deepcopy(d)


def run(args, *, tokenizer=None,pool_factory=WriterPool):
    config,base_identity=fallback.topic.load_recipe(args.config,args.workers,
        output_root=args.output_root or DEFAULT_OUTPUT,allow_missing_embedding=True)
    source=Path(args.source_root).resolve(); output=Path(config['output_root'])
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Pending-parent output must be separate from base')
    tokenizer=tokenizer or load_tokenizer(config['model_path'])
    count=lambda text:token_count(tokenizer,text)
    with ExitStack() as locks:
        locks.enter_context(source_lock(source))
        db=locks.enter_context(closing(source_connection(source)))
        source_identity,selected=read_candidates(db,source,config,args,tokenizer,locks)
        manifest={'protocol':PROTOCOL,'artifact_kind':'pending_parent_insertion_overlay',
                  'source_root':str(source),'source_identity_sha256':digest(source_identity),
                  'selected':selected,'child_policy':'reuse_validated_source_entries_without_regeneration'}
        identity={**base_identity,'pending_parent_completion':{'selection_sha256':digest(manifest),
                  'guidance_sha256':digest(fallback.GUIDANCE),'implementation_sha256':file_digest(Path(__file__))}}
        if (output/'pending_manifest.json').exists() and json.loads((output/'pending_manifest.json').read_text())!=manifest:
            raise ValueError('Pending-parent selection/source changed')
        if (output/'run_manifest.json').exists() and json.loads((output/'run_manifest.json').read_text())!=identity:
            raise ValueError('Pending-parent implementation/settings changed')
        announce('pending_parent_repair_plan',selected=len(selected),children_reused=sum(len(r['children']) for r in selected),
                 output_root=str(output),requires_embedding=False)
        if args.dry_run:return {'selected':len(selected),'requires_embedding':False}
        endpoints=args.base_urls or ([args.base_url] if args.base_url else None)
        if not endpoints:raise ValueError('Provide the local Qwen Writer endpoint')
        with file_lock(output/'.build.lock'):
            store=AdjacentBuildStore(output,identity); pool=None; done=0
            try:
                atomic_json(output/'pending_manifest.json',manifest)
                pending=[]
                for row in selected:
                    session=row['session'];bank=deepcopy(row['bank'])
                    bank['source_map']=[m for m in bank['source_map'] if m.get('parent_id')==session['parent_id']]
                    store.add_bank(bank);store.add_session(session)
                    status,directory=store.committed(session['bank_id'],session['parent_id'])
                    if status=='complete':validate_completed(session,row['children'],directory,count);done+=1
                    else:pending.append(row)
                if args.limit_sessions:pending=pending[:args.limit_sessions]
                if pending:pool=pool_factory(endpoints,config['workers'],config.get('service',{}),store)
                def generate(row):
                    session=row['session']; writer=ApprovedWriter(tokenizer,config['writer'],pool.for_session(session))
                    try:
                        writer.prepare(session)
                        parent,failed=writer._summarize_range(1,len(session['blocks']),'parent',row['children'])
                        if failed or fallback.is_placeholder(parent):raise InfrastructureError('Parent summary is still invalid')
                        result={'parent':parent,'children':[
                            {'end_block':c['end_block'],'start_reason':'session_start' if i==0 else 'semantic',
                             **{k:c[k] for k in ('title','content','entities')}} for i,c in enumerate(row['children'])],
                            'status':'complete','mode':'windowed' if writer._windowed else 'single_session','stats':writer.stats}
                        validate_build_result(result,session,tokenizer,config['writer'])
                        directory=bind_directory(session,result,identity['writer'],identity['tokenizer_id'],
                                                 count_tokens=count,prompt_version=fallback.topic.PROMPT_VERSION)
                        validate_completed(session,row['children'],directory,count)
                        return {'directory':directory,'stats':writer.stats}
                    except Exception as e:return {'error_type':type(e).__name__,'reason':str(e)[:1600]}
                for row,result in bounded_results(generate,pending,config['workers']):
                    if 'directory' in result:store.commit(row['session'],result['directory'],result['stats']);done+=1
                    else:store.failed(row['session'],result['error_type'],result['reason'])
                    announce('pending_parent_repair_progress',bank_id=row['session']['bank_id'],parent_id=row['session']['parent_id'],
                             completed=done,total=len(selected),**{k:v for k,v in result.items() if k!='directory' and k!='stats'})
            finally:
                if pool is not None:pool.close()
                try:
                    store.export({'kind':PROTOCOL})
                    patches=[]
                    for row in selected:
                        session=row['session'];status,d=store.committed(session['bank_id'],session['parent_id'])
                        if status!='complete':continue
                        validate_completed(session,row['children'],d,count)
                        patches.append({'protocol':PROTOCOL,'bank_id':session['bank_id'],'parent_id':session['parent_id'],
                            'source_hash':session['source_hash'],'before_status':'pending','before_directory_sha256':None,
                            'source_identity_sha256':digest(source_identity),'recovered_children':row['children'],
                            'archived_child_proof':row['proof'],'directory':d,'after_directory_sha256':digest(d)})
                    temp=output/'pending_parent_patches.jsonl.tmp'
                    temp.write_text(''.join(canonical(p)+'\n' for p in patches));temp.replace(output/'pending_parent_patches.jsonl')
                    report={'protocol':PROTOCOL,'selected':len(selected),'completed':len(patches),
                            'remaining':len(selected)-len(patches),'status':'ready' if len(patches)==len(selected) else 'incomplete'}
                    atomic_json(output/'pending_parent_report.json',report)
                    if not report['remaining']:atomic_json(output/'complete.json',{'n':len(patches),
                        'manifest_sha256':digest(manifest),'patches_sha256':file_digest(output/'pending_parent_patches.jsonl')})
                finally:store.close()
        if report['remaining'] and not args.limit_sessions:raise InfrastructureError('Some pending parents remain incomplete; rerun to resume')
        announce('pending_parent_repair_finished',**report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--source-root',default=fallback.DEFAULT_SOURCE)
    parser.add_argument('--output-root')
    parser.add_argument('--datasets',nargs='+',default=['split_longmemeval_s'])
    parser.add_argument('--splits',nargs='+',default=['train'])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--base-url');parser.add_argument('--base-urls',nargs='+')
    parser.add_argument('--limit-sessions',type=int)
    parser.add_argument('--dry-run',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
