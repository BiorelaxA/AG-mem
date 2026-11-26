"""Freeze mixed train/validation queries with exact, repaired hierarchy provenance.

Existing frozen sources remain read-only. Live build journals are read through
SQLite snapshots; completed repair overlays are applied before the immutable
training store is published. No question, session or repair failure is skipped.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

from .core import atomic_json, canonical, digest, file_digest, load_config
from .data import ADAPTER_VERSION, adapt_row, clean_history
from .hierarchical_training_store import (BANK_FILE, PROTOCOL, SNAPSHOT_FILE, SNAPSHOT_FORMAT,
    HierarchicalTrainingStore, snapshot_version, validate_bank)
from .jobs import file_lock
from .prepare_hierarchical_training import journal_snapshot, _match_history
from .repair_topic_event_fallbacks import apply_repair_patch
from .repair_pending_parents import apply_pending_parent_patch
from .store import Store


VERSION = 'mixed_hierarchical_preparation.v1'
SPLITS = ['train', 'validation']


def _inputs(spec, splits):
    root=Path(spec['input_root']).resolve()
    manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    entries=[r for r in manifest['sources'] if r['source']==spec['name']]
    if len(entries)!=1:raise ValueError('Input manifest lacks a unique selected source')
    files={}
    for split in splits:
        path=root/spec['name']/f'{split}.parquet'
        sha=file_digest(path)
        if sha!=entries[0]['output_files'][split]['sha256']:
            raise ValueError(f'Input Parquet changed: {path}')
        files[split]={'path':str(path),'sha256':sha,'rows':entries[0]['output_files'][split].get('rows')}
    prepared={'input_manifest_sha256':file_digest(manifest_path),'sources':[spec['name']],
              'splits':splits,'limit_per_source_split':None,'adapter_version':ADAPTER_VERSION,
              'storage_format':Store.FORMAT}
    return prepared,{'input_root':str(root),'manifest_sha256':prepared['input_manifest_sha256'],'files':files}


def _memory(records):
    references=[{k:v for k,v in r.items() if k!='messages'} for r in records]
    return {'memory_id':'mem_'+digest(references),'records':records}


def _check_input_counts(files, split, actual):
    expected=files[split]['rows']
    if expected is not None and expected!=actual:
        raise ValueError(f'Input row count differs for {split}: expected {expected}, found {actual}')


def _frozen_source(spec, prepared, provenance, splits):
    import pyarrow.parquet as pq
    root=Path(spec['frozen_root']).resolve()
    # No close(): source close() would rewrite index files. _JsonlFile reads
    # open/close their individual handles and leave no persistent descriptors.
    store=HierarchicalTrainingStore(root)
    source_identity=store.snapshot['prepare_identity']
    source_identity=source_identity.get('datasets', {}).get(spec['name'], source_identity)
    if (spec['name'] not in store.snapshot['sources'] or not set(splits).issubset(store.snapshot['splits'])
            or source_identity.get('input_manifest_sha256')!=prepared['input_manifest_sha256']):
        raise ValueError('Frozen source differs from the requested dataset/version')
    examples=[]; memories={}; banks={}
    for split in splits:
        values=[e for e in store.examples(split) if e.source==spec['name']]
        if any(e.source!=spec['name'] or e.split!=split for e in values):
            raise ValueError('Frozen source has an unexpected query source/split')
        raw=pq.read_table(provenance['files'][split]['path'],columns=['question','answer']).to_pylist()
        if len(values)!=len(raw):raise ValueError('Frozen source omits input queries')
        _check_input_counts(provenance['files'],split,len(values))
        for i,row in enumerate(raw):
            sid=f"{spec['name']}/{split}/{i:06d}"
            match=next((e for e in values if e.sample_id==sid),None)
            if match is None or match.query!=row['question'] or match.gold_answers!=[row['answer']]:
                raise ValueError(f'Frozen query/gold differs from input: {sid}')
        examples.extend(values)
    chosen_ids={e.sample_id for e in examples}
    for memory_id in {e.memory_id for e in examples}:
        mid=store.histories.get((memory_id,))
        if mid is None:raise ValueError('Frozen source memory is missing')
        memories[memory_id]=deepcopy(mid)
        bid=store.snapshot['memory_to_bank'][memory_id]
        store.hierarchical_bank(memory_id,store.snapshot['directory_version'])
        bank=deepcopy(store.hierarchies.get((bid,)))
        bank['query_memberships']=[m for m in bank.get('query_memberships',[]) if m['sample_id'] in chosen_ids]
        if not bank['query_memberships']:raise ValueError('Frozen source lost selected memberships')
        if any(store.snapshot['query_to_bank'][m['sample_id']]!=bid for m in bank['query_memberships']):
            raise ValueError('Frozen source membership crosses its signed query-to-bank mapping')
        _match_history(mid,bank);validate_bank(bank)
        banks[bid]=bank
    provenance.update(frozen_root=str(root),frozen_manifest_sha256=file_digest(root/SNAPSHOT_FILE),
                      directory_version=store.snapshot['directory_version'],
                      selected_bank_hashes={bid:digest(bank) for bid,bank in sorted(banks.items())})
    return examples,memories,banks,deepcopy(store.snapshot['construction_identity']),Counter()


def _patches(root, kind, source_root, source_identity, db):
    if root is None:return {},None
    root=Path(root).resolve()
    if kind=='fallback':
        manifest_name,patch_name,marker_key='repair_manifest.json','repair_patches.jsonl','selection_sha256'
    else:
        manifest_name,patch_name,marker_key='pending_manifest.json','pending_parent_patches.jsonl','manifest_sha256'
    manifest=json.loads((root/manifest_name).read_text())
    marker=json.loads((root/'complete.json').read_text())
    if (Path(manifest['source_root']).resolve()!=source_root
            or manifest['source_identity_sha256']!=digest(source_identity)
            or marker[marker_key]!=digest(manifest)
            or marker['patches_sha256']!=file_digest(root/patch_name)):
        raise ValueError(f'Incomplete, changed, or foreign {kind} repair overlay: {root}')
    expected={(r['bank_id'],r['parent_id']) if kind=='fallback'
              else (r['session']['bank_id'],r['session']['parent_id']) for r in manifest['selected']}
    result={}
    with (root/patch_name).open() as stream:
        for line in stream:
            patch=json.loads(line);key=(patch['bank_id'],patch['parent_id'])
            if key in result:raise ValueError('Duplicate repair patch')
            row=db.execute('SELECT snapshot,status,directory FROM sessions WHERE bank_id=? AND parent_id=?',key).fetchone()
            if row is None:raise ValueError('Repair patch source session is absent')
            session=json.loads(row[0]);directory=json.loads(row[2]) if row[2] is not None else None
            if kind=='fallback':
                if row[1]!='fallback':raise ValueError('Fallback patch source state changed')
                apply_repair_patch(directory,patch,session,source_identity)
            else:
                apply_pending_parent_patch(directory,patch,session,source_identity,source_status=row[1])
            result[key]=patch
    if set(result)!=expected or len(result)!=marker['n']:
        raise ValueError('Repair overlay does not cover its complete selection')
    names=[manifest_name,'complete.json',patch_name,'run_manifest.json']
    if (root/'length_acceptance.json').exists():names.append('length_acceptance.json')
    proof={'root':str(root),'patches':len(result),'files':{n:file_digest(root/n) for n in names}}
    return result,proof


def _locomo_journal_source(spec, prepared, provenance, splits, backup_timeout):
    """Bind query rows to shared LoCoMo histories, then freeze each bank once."""
    import pyarrow.parquet as pq

    source = Path(spec['directory_root']).resolve()
    examples, memories, banks = [], {}, {}
    applied, source_bank_hashes = Counter(), {}
    with journal_snapshot(source, backup_timeout) as db:
        identity_row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if identity_row is None:
            raise ValueError('Directory journal has no construction identity')
        construction = json.loads(identity_row[0])
        if json.loads((source/'run_manifest.json').read_text()) != construction:
            raise ValueError('Build journal construction identity differs from its manifest')
        # LoCoMo bank IDs describe histories, not one row/file. Check the input
        # fingerprints explicitly before trusting their many-query memberships.
        sources = [item for item in construction.get('datasets', [])
                   if item.get('spec', {}).get('name') == spec['name']]
        if len(sources) != 1 or sources[0]['spec'].get('format') != 'locomo_parquet':
            raise ValueError('LoCoMo journal lacks its input Parquet provenance')
        files = sources[0].get('files', [])
        for split in splits:
            matches = [item for item in files if item.get('split') == split]
            if len(matches) != 1 or matches[0].get('sha256') != provenance['files'][split]['sha256']:
                raise ValueError(f'LoCoMo directory input Parquet differs for {split}')
        fallback, fallback_proof = _patches(spec.get('fallback_repairs'), 'fallback', source, construction, db)
        pending, pending_proof = _patches(spec.get('pending_repairs'), 'pending', source, construction, db)
        if fallback.keys() & pending.keys():
            raise ValueError('Conflicting repair overlays')

        source_banks, memberships = {}, {}
        for bid, dataset, payload in db.execute(
                'SELECT bank_id,dataset,payload FROM banks WHERE dataset=?', (spec['name'],)):
            bank = json.loads(payload)
            if bank.get('bank_id') != bid or bank.get('dataset') != dataset:
                raise ValueError('Journal bank identity mismatch')
            source_banks[bid] = bank
            for member in bank.get('query_memberships', []):
                sid = member.get('sample_id')
                if not isinstance(sid, str) or sid in memberships:
                    raise ValueError('Duplicate or invalid hierarchy query membership')
                memberships[sid] = (bid, member)

        selected_members, memory_banks = {}, {}
        for split in splits:
            file = provenance['files'][split]
            count = 0
            for batch in pq.ParquetFile(file['path']).iter_batches(batch_size=1):
                for raw in batch.to_pylist():
                    index = count
                    count += 1
                    example = adapt_row(raw, spec['name'], split, index)
                    expected = {'sample_id': example.sample_id, 'split': split, 'row_index': index,
                                'original_question_id': raw['question_id']}
                    match = memberships.get(example.sample_id)
                    if match is None:
                        raise ValueError(f'Missing selected directory bank: {example.sample_id}')
                    bid, member = match
                    if member != expected:
                        raise ValueError('Selected query has a mismatched hierarchy membership')
                    mem = _memory(clean_history(raw))
                    example.memory_id = mid = mem['memory_id']
                    if mid in memories and memories[mid] != mem:
                        raise ValueError('Conflicting memory contents')
                    if mid in memory_banks and memory_banks[mid] != bid:
                        raise ValueError('One history maps to multiple directory banks')
                    memories[mid], memory_banks[mid] = mem, bid
                    selected_members.setdefault(bid, []).append(expected)
                    examples.append(example)
            _check_input_counts(provenance['files'], split, count)

        for bid, members in selected_members.items():
            original = source_banks[bid]
            bank = deepcopy(original)
            if any(item.get('status') == 'failed' for item in bank.get('issues', [])):
                raise ValueError('Selected bank contains a source adaptation failure')
            bank['sessions'], bank['directories'], originals = [], [], []
            member_splits = {member['split'] for member in members}
            for pid, sha, snapshot, status, directory in db.execute(
                    'SELECT parent_id,source_hash,snapshot,status,directory FROM sessions '
                    'WHERE bank_id=? ORDER BY parent_id', (bid,)):
                session = json.loads(snapshot)
                value = json.loads(directory) if directory is not None else None
                if session['source_hash'] != sha or session['bank_id'] != bid or session['parent_id'] != pid:
                    raise ValueError('Selected journal session source changed')
                originals.append({'parent_id': pid, 'source_hash': sha, 'status': status,
                                  'directory_sha256': digest(value)})
                key = (bid, pid)
                repair_kind = None
                if key in fallback:
                    value = apply_repair_patch(value, fallback[key], session, construction)
                    repair_kind = 'fallback'
                elif key in pending:
                    value = apply_pending_parent_patch(value, pending[key], session, construction,
                                                       source_status=status)
                    repair_kind = 'pending'
                elif status != 'complete' or value is None:
                    raise ValueError(f'Selected session has no complete directory/repair: {bid}/{pid} ({status})')
                if repair_kind:
                    for split in member_splits:
                        applied[f'{split}/{repair_kind}'] += 1
                bank['sessions'].append(session)
                bank['directories'].append(value)
            source_bank_hashes[bid] = digest({'bank': original, 'sessions': originals})
            # Source-map addresses remain immutable; only the memberships of
            # the selected train/validation queries enter the published bank.
            bank['query_memberships'] = members
            validate_bank(bank)
            banks[bid] = bank
        for mid, bid in memory_banks.items():
            _match_history(memories[mid], banks[bid])
    provenance.update(directory_root=str(source), construction_identity_sha256=digest(construction),
                      selected_bank_hashes=source_bank_hashes,
                      repair_overlays={'fallback': fallback_proof, 'pending': pending_proof})
    return examples, memories, banks, construction, applied


def _journal_source(spec, prepared, provenance, splits, backup_timeout):
    if spec['name'] == 'locomo':
        return _locomo_journal_source(spec, prepared, provenance, splits, backup_timeout)
    import pyarrow.parquet as pq
    if 'split_journals' in spec:
        return _split_journal_source(spec, provenance, splits)
    source=Path(spec['directory_root']).resolve()
    examples=[];memories={};banks={}; applied=Counter(); source_bank_hashes={}
    with journal_snapshot(source,backup_timeout) as db:
        identity_row=db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if identity_row is None:raise ValueError('Directory journal has no construction identity')
        construction=json.loads(identity_row[0])
        if json.loads((source/'run_manifest.json').read_text())!=construction:
            raise ValueError('Build journal construction identity differs from its manifest')
        fallback,fallback_proof=_patches(spec.get('fallback_repairs'),'fallback',source,construction,db)
        pending,pending_proof=_patches(spec.get('pending_repairs'),'pending',source,construction,db)
        if fallback.keys() & pending.keys():raise ValueError('Conflicting repair overlays')
        for split in splits:
            file=provenance['files'][split]
            rows=pq.ParquetFile(file['path'])
            count=0
            for batch in rows.iter_batches(batch_size=1):
                for raw in batch.to_pylist():
                    index=count;count+=1
                    example=adapt_row(raw,spec['name'],split,index)
                    mem=_memory(clean_history(raw));example.memory_id=mem['memory_id']
                    bid='bank_'+digest([spec['name'],split,file['sha256'],index])[:24]
                    saved=db.execute('SELECT dataset,payload FROM banks WHERE bank_id=?',(bid,)).fetchone()
                    if saved is None or saved[0]!=spec['name']:raise ValueError(f'Missing selected directory bank: {example.sample_id}')
                    bank=json.loads(saved[1])
                    expected={'sample_id':example.sample_id,'split':split,'row_index':index,'original_question_id':raw['question_id']}
                    if bank.get('bank_id')!=bid or bank.get('dataset')!=spec['name'] or bank.get('query_memberships')!=[expected]:
                        raise ValueError('Selected query has a mismatched hierarchy membership')
                    if any(i.get('status')=='failed' for i in bank.get('issues',[])):
                        raise ValueError('Selected bank contains a source adaptation failure')
                    bank['sessions']=[];bank['directories']=[]; originals=[]
                    for pid,sha,snapshot,status,directory in db.execute(
                        'SELECT parent_id,source_hash,snapshot,status,directory FROM sessions WHERE bank_id=? ORDER BY parent_id',(bid,)):
                        session=json.loads(snapshot);value=json.loads(directory) if directory is not None else None
                        if session['source_hash']!=sha or session['bank_id']!=bid or session['parent_id']!=pid:
                            raise ValueError('Selected journal session source changed')
                        originals.append({'parent_id':pid,'source_hash':sha,'status':status,'directory_sha256':digest(value)})
                        key=(bid,pid)
                        if key in fallback:
                            value=apply_repair_patch(value,fallback[key],session,construction);applied[f'{split}/fallback']+=1
                        elif key in pending:
                            value=apply_pending_parent_patch(value,pending[key],session,construction,source_status=status)
                            applied[f'{split}/pending']+=1
                        elif status!='complete' or value is None:
                            raise ValueError(f'Selected session has no complete directory/repair: {bid}/{pid} ({status})')
                        bank['sessions'].append(session);bank['directories'].append(value)
                    source_bank_hashes[bid]=digest({'bank':json.loads(saved[1]),'sessions':originals})
                    bank['query_memberships']=[expected]
                    validate_bank(bank);_match_history(mem,bank)
                    if mem['memory_id'] in memories and memories[mem['memory_id']]!=mem:
                        raise ValueError('Conflicting memory contents')
                    memories[mem['memory_id']]=mem;banks[bid]=bank;examples.append(example)
            _check_input_counts(provenance['files'],split,count)
    provenance.update(directory_root=str(source),construction_identity_sha256=digest(construction),
                      selected_bank_hashes=source_bank_hashes,
                      repair_overlays={'fallback':fallback_proof,'pending':pending_proof})
    return examples,memories,banks,construction,applied


def _split_journal_source(spec, provenance, splits):
    """Join disjoint committed session addresses across the completed M builds."""
    import pyarrow.parquet as pq
    from .reuse_adjacent_directories import source_connection

    banks, memories, examples, applied = {}, {}, [], Counter()
    identities, proofs = {}, {}
    for split in splits:
        file = provenance['files'][split]
        with ExitStack() as stack:
            shards = []
            for shard in spec['split_journals'][split]:
                root = Path(shard['directory_root']).resolve()
                db = stack.enter_context(closing(source_connection(root)))
                db.execute('BEGIN')
                identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
                if identity != json.loads((root/'run_manifest.json').read_text()):
                    raise ValueError(f'Journal identity mismatch: {root}')
                patches, proof = _patches(shard.get('fallback_repairs'), 'fallback', root, identity, db)
                identities[str(root)] = identity
                proofs[str(root)] = proof
                shards.append((db, identity, patches))
            rows = pq.ParquetFile(file['path']).iter_batches(batch_size=1)
            count = 0
            for index, batch in enumerate(rows):
                raw, = batch.to_pylist()
                count += 1
                example = adapt_row(raw, spec['name'], split, index)
                mem = _memory(clean_history(raw))
                example.memory_id = mem['memory_id']
                bid = 'bank_' + digest([spec['name'], split, file['sha256'], index])[:24]
                member = dict(sample_id=example.sample_id, split=split, row_index=index,
                              original_question_id=raw['question_id'])
                bank = dict(bank_id=bid, dataset=spec['name'], query_memberships=[member], issues=[], source_map=[])
                parents = {}
                for db, identity, patches in shards:
                    saved = db.execute('SELECT payload FROM banks WHERE bank_id=?', (bid,)).fetchone()
                    if not saved:
                        continue
                    original = json.loads(saved[0])
                    if (original['bank_id'] != bid or original['query_memberships'] != [member]
                            or original['dataset'] != spec['name']):
                        raise ValueError(f'Unexpected source membership for {bid}')
                    kept = set()
                    for pid, source_hash, snapshot, encoded, status in db.execute(
                            "SELECT parent_id,source_hash,snapshot,directory,status FROM sessions WHERE bank_id=? "
                            "AND status IN ('complete','fallback')", (bid,)):
                        if pid in parents:
                            raise ValueError(f'Overlapping committed sessions: {bid}/{pid}')
                        session, directory = json.loads(snapshot), json.loads(encoded)
                        if session['parent_id'] != pid or session['source_hash'] != source_hash:
                            raise ValueError(f'Session address or hash differs from journal: {bid}/{pid}')
                        if status == 'fallback':
                            directory = apply_repair_patch(directory, patches[(bid,pid)], session, identity)
                            applied[f'{split}/fallback'] += 1
                        parents[pid] = (session, directory)
                        kept.add(pid)
                    bank['source_map'].extend(r for r in original['source_map'] if r['parent_id'] in kept)
                ordered = [parents[k] for k in sorted(parents)]
                bank['sessions'] = [s for s,d in ordered]
                bank['directories'] = [d for s,d in ordered]
                validate_bank(bank)
                _match_history(mem, bank)
                banks[bid], memories[example.memory_id] = bank, mem
                examples.append(example)
            _check_input_counts(provenance['files'], split, count)
    provenance.update(repair_overlays=proofs, selected_bank_hashes={bid:digest(b) for bid,b in banks.items()})
    return examples, memories, banks, identities, applied


def prepare_mixed(config, *, check_only=False, backup_timeout=120):
    data=config['data'];sources=data['sources'];splits=data['splits']; specs=data.get('source_specs',[])
    if (data.get('directory_protocol')!=PROTOCOL or data.get('layout','combined')!='combined'
            or splits!=SPLITS or not sources or len(set(sources))!=len(sources)
            or [s['name'] for s in specs]!=sources):
        raise ValueError('Mixed freeze requires ordered unique source_specs and train/validation only')
    target=Path(data['prepared_root']).resolve()
    for spec in specs:
        if sum(k in spec for k in ('frozen_root','directory_root','split_journals')) != 1:
            raise ValueError('Each source needs exactly one frozen_root or directory_root')
        artifacts = [spec, *(shard for shards in spec.get('split_journals', {}).values() for shard in shards)]
        for artifact in artifacts:
            for key in ('input_root','frozen_root','directory_root','fallback_repairs','pending_repairs'):
                if key in artifact:
                    old=Path(artifact[key]).resolve()
                    if target==old or old in target.parents or target in old.parents:
                        raise ValueError('Mixed output must be independent of every source artifact')
    all_examples=[];all_memories={};all_banks={}; identities={};provenance={};constructions={};reports={}
    for spec in specs:
        name=spec['name'];identities[name],proof=_inputs(spec,splits)
        loader=_frozen_source if 'frozen_root' in spec else _journal_source
        args=(spec,identities[name],proof,splits)
        values=loader(*args) if 'frozen_root' in spec else loader(*args,backup_timeout)
        examples,memories,banks,construction,applied=values
        for mid,mem in memories.items():
            if mid in all_memories and all_memories[mid]!=mem:raise ValueError('Conflicting cross-source memory ID')
            all_memories[mid]=mem
        for bid,bank in banks.items():
            if bid in all_banks and all_banks[bid]!=bank:raise ValueError('Conflicting cross-source bank ID')
            all_banks[bid]=bank
        all_examples.extend(examples);provenance[name]=proof;constructions[name]=construction
        reports[name]={'queries':dict(Counter(e.split for e in examples)),'banks':len(banks),
                       'sessions':sum(len(b['sessions']) for b in banks.values()),'repairs_applied':dict(applied)}
        if spec.get('expected_counts') and reports[name]['queries']!=spec['expected_counts']:
            raise ValueError(f'Expected source query counts differ: {name}')
    if len({e.sample_id for e in all_examples})!=len(all_examples):raise ValueError('Duplicate mixed query IDs')
    query_map={};memory_map={}
    memberships={}
    for bid,bank in all_banks.items():
        for member in bank['query_memberships']:
            if member['sample_id'] in memberships:raise ValueError('Duplicate mixed bank membership')
            memberships[member['sample_id']]=bid
    for example in all_examples:
        bid=memberships.get(example.sample_id)
        if bid is None or all_banks[bid]['dataset']!=example.source:raise ValueError('Mixed query bank mapping is incomplete')
        if example.memory_id in memory_map and memory_map[example.memory_id]!=bid:
            raise ValueError('One history maps to multiple directory banks')
        memory_map[example.memory_id]=bid;query_map[example.sample_id]=bid
    input_hashes={name:identity['input_manifest_sha256'] for name,identity in identities.items()}
    identity={'layout':'mixed_sources','sources':sources,'splits':splits,'limit_per_source_split':None,
              'adapter_version':ADAPTER_VERSION,'storage_format':Store.FORMAT,'datasets':identities,
              'input_roots':{s['name']:str(Path(s['input_root']).resolve()) for s in specs},
              'input_manifest_sha256_by_source':input_hashes,'input_manifest_sha256':digest(input_hashes)}
    proof={'protocol':VERSION,'sources':provenance}
    for spec in specs:
        _,current=_inputs(spec,splits)
        original=provenance[spec['name']]
        if any(current[k]!=original[k] for k in ('input_root','manifest_sha256','files')):
            raise ValueError('Input changed during mixed preparation')
        if 'frozen_root' in original and file_digest(Path(original['frozen_root'])/SNAPSHOT_FILE)!=original['frozen_manifest_sha256']:
            raise ValueError('Frozen source changed during mixed preparation')
        for overlay in original.get('repair_overlays',{}).values():
            if overlay and any(file_digest(Path(overlay['root'])/name)!=sha for name,sha in overlay['files'].items()):
                raise ValueError('Repair overlay changed during mixed preparation')
    report={'event':'mixed_hierarchical_prepared','prepared_root':str(target),'by_source':reports,
            'examples':dict(Counter(e.split for e in all_examples)),'banks':len(all_banks),
            'sessions':sum(len(b['sessions']) for b in all_banks.values()),'test_queries_included':0}
    if target.exists():
        frozen=HierarchicalTrainingStore(target)
        if frozen.snapshot['prepare_identity']!=identity or frozen.snapshot['provenance']!=proof:
            raise ValueError('Existing mixed snapshot has different inputs/repairs/provenance')
        frozen.validate_directories(splits)
        return {**report,'directory_version':frozen.snapshot['directory_version'],'reused':True,'check_only':check_only}
    if check_only:return {**report,'status':'inputs_verified','reused':False,'check_only':True}
    target.parent.mkdir(parents=True,exist_ok=True)
    with file_lock(target.parent/'.prepare_mixed.lock'):
        if target.exists():raise ValueError('Mixed output appeared during preparation; rerun to validate it')
        staging=Path(tempfile.mkdtemp(prefix='.freeze-mixed-',dir=target.parent))
        try:
            store=Store(staging)
            try:
                for mid,mem in sorted(all_memories.items()):
                    if store.add_memory(mem['records'])!=mid:raise ValueError('Frozen memory ID changed')
                for example in sorted(all_examples,key=lambda e:e.sample_id):store.add_example(example)
                with (staging/BANK_FILE).open('w') as stream:
                    for bid,bank in sorted(all_banks.items()):stream.write(canonical(bank)+'\n')
                frozen={'schema_version':SNAPSHOT_FORMAT,'directory_protocol':PROTOCOL,'sources':sources,'splits':splits,
                        'prepare_identity':identity,'construction_identity':{'kind':'mixed_hierarchy.v1','by_source':constructions},
                        'provenance':proof,'memory_to_bank':memory_map,'query_to_bank':query_map,
                        'banks':{bid:{'sha256':digest(bank),'dataset':bank['dataset'],'sessions':len(bank['sessions'])}
                                 for bid,bank in all_banks.items()},
                        'files':{name:file_digest(staging/name) for name in
                                 [BANK_FILE,'memory_records.jsonl',*[f'{s}_examples.jsonl' for s in splits]]}}
                version=snapshot_version(frozen);frozen['directory_version']=version
                atomic_json(staging/SNAPSHOT_FILE,frozen)
                store.metadata.update(prepare_identity=identity,prepare_complete=True,directory_protocol=PROTOCOL,
                    directory_version=version,directory_empty_pending=0,data={'directory_protocol':PROTOCOL,'sources':sources,'splits':splits})
                atomic_json(store.manifest_path,store.metadata)
            finally:store.close()
            report.update(directory_version=version,reused=False)
            atomic_json(staging/'hierarchical_preparation_report.json',report)
            checked=HierarchicalTrainingStore(staging)
            try:checked.validate_directories(splits)
            finally:checked.close()
            staging.rename(target)
        finally:
            if staging.exists():shutil.rmtree(staging)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--backup-timeout',type=float,default=120)
    args=parser.parse_args()
    print(canonical(prepare_mixed(load_config(args.config),check_only=args.check_only,backup_timeout=args.backup_timeout)),flush=True)


if __name__=='__main__':main()
