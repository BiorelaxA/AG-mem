"""Reuse saved repairs with user-approved field budgets; generate missing parents.

The old repair and base stores stay read-only. Successful old directories keep
their exact bytes. Saved stop responses are checked against the exact source
payload before accepting their lengths. A child alone never completes a session.
"""
import argparse
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path

from . import repair_topic_event_fallbacks as repair
from .build_adjacent_directories import AdjacentBuildStore
from .core import atomic_json, canonical, digest, file_digest
from .hierarchical.binding import bind_directory
from .hierarchical.writer import token_count, validate_build_result, _text_budget
from .jobs import file_lock
from .pipeline import announce
from .policy import load_tokenizer
from .reuse_adjacent_directories import source_connection, source_lock


DEFAULT_CONFIG = 'configs/repair_longmemeval_s_fallbacks_len500.yaml'
DEFAULT_PREVIOUS = 'runs/parent_child_directories/longmemeval_s_31535150_fallback_repair_20261114'
POLICY = 'accept_saved_length_exceptions_500_500_30.v1'


class ApprovedWriter(repair.ConciseRepairWriter):
    def __init__(self, *args, components=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.components = components or {}

    def _validate_entry(self, output, kind, payload):
        super()._validate_entry(output, kind, payload)
        if kind == 'child':
            _text_budget(output['title'], 'child.title', 24, self.tokenizer)

    def repair(self, session, previous):
        previous = deepcopy(previous)
        for child in previous['children']:
            key = (session['bank_id'], session['parent_id'], child['id'])
            if key in self.components:
                if not repair.is_placeholder(child):
                    raise ValueError('Saved component can only replace a placeholder child')
                child.update(deepcopy(self.components[key]['entry']))
        return super().repair(session, previous)


def latest_entry(db, row, kind, writer, children, start, end):
    session = row['session']
    result = db.execute(
        "SELECT payload FROM writer_traces WHERE json_extract(payload,'$.event')='writer_response' "
        "AND json_extract(payload,'$.bank_id')=? AND json_extract(payload,'$.parent_id')=? "
        "AND json_extract(payload,'$.stage') IN (?,?) ORDER BY seq DESC LIMIT 1",
        (session['bank_id'], session['parent_id'], 'adjacent_'+kind, 'adjacent_'+kind+'_repair')).fetchone()
    if result is None:
        return None
    event = json.loads(result[0]); response = event['response']
    if response.get('finish_reason') != 'stop' or not isinstance(response.get('text'), str):
        raise ValueError('Only complete saved stop responses may be accepted')
    prior = db.execute(
        "SELECT payload FROM writer_traces WHERE json_extract(payload,'$.event')='writer_request' "
        "AND json_extract(payload,'$.bank_id')=? AND json_extract(payload,'$.parent_id')=? "
        "AND json_extract(payload,'$.request_sha256')=? ORDER BY seq DESC LIMIT 1",
        (session['bank_id'], session['parent_id'], event['request_sha256'])).fetchone()
    if prior is None:
        raise ValueError('Saved response has no paired request')
    request = json.loads(prior[0])['request']
    if digest(request) != event['request_sha256']:
        raise ValueError('Saved request hash changed')
    payload = json.loads(request['messages'][-1]['content'])
    context = len(payload.get('context_blocks', []))
    if context > writer.settings['context_blocks']:
        raise ValueError('Saved child context differs from the repair recipe')
    expected = writer._raw_payload(start, end, kind, children, context_count=context)
    for key in ('input_kind','scope','source_kind','session_metadata','headers','write_range',
                'blocks','context_blocks','children','output_language'):
        if payload.get(key) != expected.get(key):
            raise ValueError('Saved response belongs to different source/children: '+key)
    entry = json.loads(response['text'])
    writer._validate_entry(entry, kind, payload)
    if repair.is_placeholder(entry):
        raise ValueError('A placeholder is not an acceptable saved response')
    return {'entry': entry, 'request_sha256': event['request_sha256'],
            'response_sha256': digest(response), 'prior_stage': event['stage'],
            'field_tokens': {k: token_count(writer.tokenizer, entry[k]) for k in ('title','content')}}


def prepare(args, *, tokenizer=None):
    config, base_identity = repair.topic.load_recipe(args.config, args.workers,
        output_root=getattr(args, 'output_root', None), allow_missing_embedding=True)
    if (config['writer']['parent_content_tokens'], config['writer']['child_content_tokens'],
            config['writer']['title_tokens']) != (500, 500, 30):
        raise ValueError('This exception workflow requires approved budgets 500/500/30')
    source, previous, output = (Path(v).resolve() for v in
                               (args.source_root, args.previous_repairs, config['output_root']))
    if any(output == old or output in old.parents or old in output.parents for old in (source, previous)):
        raise ValueError('Use a new output root separate from both previous stores')
    datasets, splits = args.datasets or ['split_longmemeval_s'], args.splits or ['train','validation','test']
    with source_lock(source), source_lock(previous), closing(source_connection(previous)) as db:
        source_identity, selected = repair._read_source(source, datasets, splits)
        manifest = repair._selection_manifest(source, source_identity, selected, datasets, splits)
        if json.loads((previous/'repair_manifest.json').read_text()) != manifest:
            raise ValueError('Previous repair selection differs from the exact source')
        old_identity = json.loads((previous/'run_manifest.json').read_text())
        stored = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if stored is None or json.loads(stored[0]) != old_identity:
            raise ValueError('Previous repair journal identity changed')
        identity = {**deepcopy(base_identity), 'repair_overlay': {
            'version': repair.VERSION, 'selection_sha256': digest(manifest),
            'prompt_guidance': repair.GUIDANCE, 'prompt_guidance_sha256': digest(repair.GUIDANCE),
            'boundary_policy': manifest['boundary_policy'],
            'implementation_sha256': file_digest(Path(repair.__file__))}}
        tokenizer = tokenizer or load_tokenizer(config['model_path'])
        count = lambda text: token_count(tokenizer, text)
        seeded, components, parents = [], [], []
        for row in selected:
            session, before = row['session'], row['directory']
            old = db.execute('SELECT snapshot,status,directory,result_stats FROM sessions WHERE bank_id=? AND parent_id=?',
                             (session['bank_id'],session['parent_id'])).fetchone()
            if old is None or json.loads(old[0]) != session:
                raise ValueError('Previous repair source snapshot differs')
            if old[1] == 'complete':
                directory = json.loads(old[2])
                repair.validate_replacement(before, directory, session, count_tokens=count)
                seeded.append((row, directory, json.loads(old[3] or '{}'), 'previous_complete'))
                continue
            writer = ApprovedWriter(tokenizer, config['writer'], None)
            writer.prepare(session)
            children, parent_inputs = [], []
            for child in before['children']:
                start, end = child['source']['start_block'], child['source']['end_block']
                entry = {k:deepcopy(child[k]) for k in ('title','content','entities')}
                if repair.is_placeholder(child):
                    saved = latest_entry(db,row,'child',writer,[],start,end)
                    if saved is None:
                        raise ValueError('No saved child response is available')
                    components.append({'bank_id':session['bank_id'],'parent_id':session['parent_id'],
                                       'child_id':child['id'],'source_hash':session['source_hash'],**saved})
                    entry = saved['entry']
                children.append({'end_block':end,'start_reason':child['start_reason'],**entry})
                parent_inputs.append({'start_block':start,'end_block':end,**entry})
            saved = latest_entry(db,row,'parent',writer,parent_inputs,1,len(session['blocks']))
            if saved is None:
                continue
            result = {'parent':saved['entry'],'children':children,'status':'complete',
                      'mode':before['build']['mode'],'stats':{'accepted_saved_parent':1},'traces':[]}
            validate_build_result(result,session,tokenizer,config['writer'])
            directory = bind_directory(session,result,identity['writer'],identity['tokenizer_id'],
                                       count_tokens=count,prompt_version=repair.topic.PROMPT_VERSION)
            repair.validate_replacement(before,directory,session,count_tokens=count)
            seeded.append((row,directory,result['stats'],'accepted_saved_parent'))
            parents.append({'bank_id':session['bank_id'],'parent_id':session['parent_id'],**saved})
        accepted = {'policy':POLICY,'previous_repairs':str(previous),'source_root':str(source),
                    'previous_identity_sha256':digest(old_identity),'selection_sha256':digest(manifest),
                    'limits':{'parent_content_tokens':500,'child_content_tokens':500,
                              'parent_title_tokens':30,'child_title_tokens':24},
                    'previous_complete':[{'bank_id':r['session']['bank_id'],'parent_id':r['session']['parent_id'],
                                          'directory_sha256':digest(d)} for r,d,_,kind in seeded if kind=='previous_complete'],
                    'accepted_parents':parents,'accepted_children':components,
                    'implementation_sha256':file_digest(Path(__file__))}
        acceptance_file = output/'length_acceptance.json'
        if acceptance_file.exists() and json.loads(acceptance_file.read_text()) != accepted:
            raise ValueError('Saved length acceptance/source evidence changed')
        plan = {'selected':len(selected),'previous_complete':len(accepted['previous_complete']),
                'accepted_saved_parents':len(parents),'accepted_saved_children':len(components),
                'parents_to_generate':len(selected)-len(seeded),'output_root':str(output)}
        if args.dry_run:
            announce('fallback_length_acceptance_plan',**plan)
            return config,components,plan
        with file_lock(output/'.build.lock'):
            store = AdjacentBuildStore(output,identity)
            try:
                atomic_json(output/'repair_manifest.json',manifest)
                atomic_json(acceptance_file,accepted)
                for row in selected:
                    bank = deepcopy(row['bank'])
                    bank['source_map'] = [m for m in bank.get('source_map', [])
                                          if m.get('parent_id') == row['session']['parent_id']]
                    store.add_bank(bank); store.add_session(row['session'])
                for row,directory,stats,kind in seeded:
                    status,existing=store.committed(row['session']['bank_id'],row['session']['parent_id'])
                    if status=='complete':
                        if existing != directory:
                            raise ValueError('An adopted complete result changed')
                        continue
                    store.commit(row['session'],directory,{**stats,'acceptance_origin':kind})
                summary=repair._publish(store,selected,manifest,{'kind':POLICY,'run_status':'prepared_only',
                                        'length_acceptance_sha256':digest(accepted)})
            finally:store.close()
        announce('fallback_length_acceptance_prepared',**plan,repaired=summary['repaired'],remaining=summary['remaining'])
        return config,components,summary


def run(args):
    _, components, result = prepare(args)
    if args.dry_run or args.prepare_only:
        return result
    indexed={(r['bank_id'],r['parent_id'],r['child_id']):r for r in components}
    def writer_factory(*values, **kwargs):
        return ApprovedWriter(*values,components=indexed,**kwargs)
    return repair.run(args,writer_factory=writer_factory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--source-root',default=repair.DEFAULT_SOURCE)
    parser.add_argument('--previous-repairs',default=DEFAULT_PREVIOUS)
    parser.add_argument('--output-root')
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--splits',nargs='+')
    parser.add_argument('--workers',type=int)
    urls=parser.add_mutually_exclusive_group()
    urls.add_argument('--base-url');urls.add_argument('--base-urls',nargs='+')
    parser.add_argument('--limit-sessions',type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':
    main()
