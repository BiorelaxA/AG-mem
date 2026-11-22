"""Repair placeholder entries into a small, source-bound directory overlay.

The source journal is read-only. Child IDs and boundaries remain fixed; valid
child text remains byte-for-byte equal. Only placeholder children and parent
summaries are generated, without embeddings or boundary review. Consumers must
apply repair_patches.jsonl against its exact recorded source hashes.
"""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time

from . import build_topic_event_directories as topic
from .build_adjacent_directories import AdjacentBuildStore
from .core import InfrastructureError, atomic_json, canonical, digest, file_digest
from .execution import bounded_results
from .hierarchical.binding import bind_directory, validate_directory
from .hierarchical.client import WriterPool
from .hierarchical.records import validate_session
from .hierarchical.writer import token_count, validate_build_result
from .jobs import file_lock
from .pipeline import announce
from .reuse_adjacent_directories import source_connection, source_lock


VERSION = 'placeholder_directory_repair_overlay.v1'
DEFAULT_CONFIG = 'configs/repair_longmemeval_s_fallbacks.yaml'
DEFAULT_SOURCE = 'runs/parent_child_directories/qwen27b_adjacent_qwen4b_31535150_incremental_v2'
PLACEHOLDER = 'Automatic summary unavailable; read the original source span.'
GUIDANCE = '''\n\nRepair output discipline:
Keep the JSON concise and factual. Use short phrases, without an introduction or
repetition. Aim for at most 12 tokens in title and three quarters of the allowed content
budget, while retaining key facts. The hard limits in limits are mandatory.
Do not fill the budget just because space remains. Child entities must be unique.
Do not repeat a request to read the source or the automatic-summary placeholder.
'''


def is_placeholder(entry):
    return entry.get('content', '').strip() == PLACEHOLDER


def _read_source(source, datasets, splits):
    """One consistent read transaction; caller retains the source builder lock."""
    with closing(source_connection(source)) as db:
        db.execute('BEGIN')
        row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row is None:
            raise ValueError('Source has no construction identity')
        identity = json.loads(row[0])
        if json.loads((source/'run_manifest.json').read_text()) != identity:
            raise ValueError('Source journal and construction manifest differ')
        banks = {bid: json.loads(payload) for bid, dataset, payload in db.execute(
            'SELECT bank_id,dataset,payload FROM banks') if dataset in datasets}
        selected = []
        for bid, pid, sha, snapshot, encoded, stats in db.execute(
                "SELECT bank_id,parent_id,source_hash,snapshot,directory,result_stats "
                "FROM sessions WHERE status='fallback' ORDER BY bank_id,parent_id"):
            bank = banks.get(bid)
            if bank is None or not any(m['split'] in splits for m in bank['query_memberships']):
                continue
            session, directory = json.loads(snapshot), json.loads(encoded)
            validate_session(session)
            validate_directory(directory, session)
            if (session['bank_id'] != bid or session['parent_id'] != pid or session['source_hash'] != sha
                    or directory['build']['status'] != 'fallback'):
                raise ValueError('Source fallback address/hash/status differs from its journal')
            stats = json.loads(stats or '{}')
            if any(stats.get(k, 0) for k in ('segmentation_review_failures', 'segmentation_review_unavailable',
                                            'boundary_review_failures')):
                raise ValueError(f'Fallback requires boundary repair, not placeholder repair: {bid}/{pid}')
            bad_children = [c['id'] for c in directory['children'] if is_placeholder(c)]
            if not is_placeholder(directory) and not bad_children:
                raise ValueError(f'Fallback has no recognized placeholder: {bid}/{pid}')
            selected.append({'dataset': bank['dataset'], 'bank': bank, 'session': session,
                             'directory': directory, 'placeholder_children': bad_children})
        db.rollback()
    return identity, selected


def _selection_manifest(source, identity, selected, datasets, splits):
    return {'protocol': VERSION, 'artifact_kind': 'partial_directory_repair_overlay',
            'source_root': str(source), 'source_identity': identity, 'source_identity_sha256': digest(identity),
            'datasets': sorted(datasets), 'splits': sorted(splits),
            'boundary_policy': 'preserve_source_child_ids_and_spans',
            'valid_children_policy': 'preserve_source_child_text_and_entities',
            'source_is_read_only': True,
            'selected': [{'dataset': r['dataset'], 'bank_id': r['session']['bank_id'],
                          'parent_id': r['session']['parent_id'], 'source_hash': r['session']['source_hash'],
                          'before_status': 'fallback',
                          'before_directory_sha256': digest(r['directory']),
                          'before_directory_version': r['directory']['directory_version'],
                          'source_bank_sha256': digest(r['bank']),
                          'placeholder_children': r['placeholder_children']}
                         for r in selected]}


class ConciseRepairWriter(topic.TopicEventDirectoryWriter):
    def __init__(self, tokenizer, settings, complete):
        super().__init__(tokenizer, settings, complete)
        for kind in ('child', 'parent'):
            self.adjacent_prompts[kind] += GUIDANCE

    def repair(self, session, previous):
        self.prepare(session)
        children, parent_inputs = [], []
        for child in previous['children']:
            start, end = child['source']['start_block'], child['source']['end_block']
            entry = {k: deepcopy(child[k]) for k in ('title', 'content', 'entities')}
            if is_placeholder(child):
                entry, failed = self._summarize_range(start, end, 'child', [])
                if failed or is_placeholder(entry):
                    raise InfrastructureError(f'Child placeholder repair remains invalid: {child["id"]}')
            children.append({'end_block': end, 'start_reason': child['start_reason'], **entry})
            parent_inputs.append({'start_block': start, 'end_block': end, **entry})
            self.stats['child_entries'] += 1
        parent, failed = self._summarize_range(1, len(session['blocks']), 'parent', parent_inputs)
        if failed or is_placeholder(parent):
            raise InfrastructureError('Parent placeholder repair remains invalid')
        result = {'parent': parent, 'children': children, 'status': 'complete',
                  'mode': 'windowed' if self._windowed or previous['build']['mode'] == 'windowed' else 'single_session',
                  'stats': deepcopy(self.stats), 'traces': deepcopy(self.traces)}
        return validate_build_result(result, session, self.tokenizer, self.settings)


def validate_replacement(previous, replacement, session, *, count_tokens=None):
    validate_directory(previous, session, count_tokens=count_tokens)
    validate_directory(replacement, session, count_tokens=count_tokens)
    if replacement['build']['status'] != 'complete':
        raise ValueError('Only a fully validated complete repair may be published')
    if len(previous['children']) != len(replacement['children']):
        raise ValueError('Repair changed child count')
    for old, new in zip(previous['children'], replacement['children']):
        if any(old[k] != new[k] for k in ('id', 'parent_id', 'order', 'source', 'start_reason')):
            raise ValueError('Repair changed an immutable child ID/boundary')
        if not is_placeholder(old) and old != new:
            raise ValueError('Repair changed a valid child entry')
        if is_placeholder(new):
            raise ValueError('Repair still contains a child placeholder')
    if is_placeholder(replacement):
        raise ValueError('Repair still contains a parent placeholder')


def apply_repair_patch(base_directory, patch, session, source_identity=None):
    """Return a checked replacement without mutating the base or its journal.

    A frozen training snapshot can call this before accepting the directory.
    Passing the base journal's construction identity additionally verifies its
    provenance; exact directory/snapshot hashes are mandatory in all cases.
    This overlay is partial and never replaces an entire bank on its own.
    """
    if not isinstance(patch, dict) or patch.get('protocol') != VERSION:
        raise ValueError('Unsupported directory repair patch')
    validate_session(session)
    validate_directory(base_directory, session)
    expected = {'bank_id': session['bank_id'], 'parent_id': session['parent_id'],
                'source_hash': session['source_hash'], 'before_status': 'fallback',
                'before_directory_sha256': digest(base_directory),
                'before_directory_version': base_directory['directory_version']}
    if base_directory['build']['status'] != 'fallback' or any(patch.get(k) != v for k,v in expected.items()):
        raise ValueError('Repair patch does not match the exact source fallback')
    if source_identity is not None and patch.get('source_identity_sha256') != digest(source_identity):
        raise ValueError('Repair patch belongs to another source construction identity')
    replacement = patch.get('directory')
    if not isinstance(replacement, dict) or patch.get('after_directory_sha256') != digest(replacement):
        raise ValueError('Repair patch replacement hash differs')
    validate_replacement(base_directory, replacement, session)
    return deepcopy(replacement)


def _publish(store, selected, manifest, execution):
    report = store.export(execution)
    patches, missing = [], []
    for r, before in zip(selected, manifest['selected']):
        session = r['session']
        status, directory = store.committed(session['bank_id'], session['parent_id'])
        if status != 'complete' or directory is None:
            missing.append({'bank_id': session['bank_id'], 'parent_id': session['parent_id'], 'status': status})
            continue
        validate_replacement(r['directory'], directory, session)
        patches.append({'protocol': VERSION, **before, 'source_root': manifest['source_root'],
                        'source_identity_sha256': manifest['source_identity_sha256'],
                        'after_directory_sha256': digest(directory), 'directory': directory})
    temporary = store.root/'repair_patches.jsonl.tmp'
    with temporary.open('w') as out:
        for row in patches:
            out.write(canonical(row)+'\n')
    temporary.replace(store.root/'repair_patches.jsonl')
    summary = {'protocol': VERSION, 'source_root': manifest['source_root'], 'output_root': str(store.root),
               'selected': len(selected), 'repaired': len(patches), 'remaining': len(missing),
               'status': 'ready' if not missing else 'incomplete', 'missing': missing,
               'requires_base_directory': True, 'source_modified': False,
               'selection_sha256': digest(manifest), 'execution': execution, 'states': report['states']}
    atomic_json(store.root/'repair_report.json', summary)
    marker = store.root/'complete.json'
    if not missing:
        atomic_json(marker, {'selection_sha256': digest(manifest), 'n': len(patches),
                             'patches_sha256': file_digest(store.root/'repair_patches.jsonl')})
    elif marker.exists():
        marker.unlink()
    return summary


def run(args, *, tokenizer=None, pool_factory=WriterPool, writer_factory=ConciseRepairWriter):
    config, base_identity = topic.load_recipe(args.config, args.workers,
        output_root=getattr(args, 'output_root', None), allow_missing_embedding=True)
    if config.get('preserve_existing_directories', False):
        raise ValueError('A repair overlay must use a separate, non-preservation recipe')
    source = Path(args.source_root).resolve()
    output = Path(config['output_root']).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Repair overlay output must be separate from its source')
    datasets = args.datasets or ['split_longmemeval_s']
    splits = args.splits or ['train', 'validation', 'test']
    if (not datasets or len(set(datasets)) != len(datasets) or len(set(splits)) != len(splits)
            or not set(splits).issubset({'train', 'validation', 'test'})):
        raise ValueError('Select unique datasets and splits')
    specs = {s['name']: s for s in config['datasets']}
    if not set(datasets).issubset(specs):
        raise ValueError('Repair datasets must be present in the recipe')
    limit = getattr(args, 'limit_sessions', None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit_sessions must be a positive integer')
    with source_lock(source):
        source_identity, selected = _read_source(source, datasets, splits)
        manifest = _selection_manifest(source, source_identity, selected, datasets, splits)
        identity = {**deepcopy(base_identity), 'repair_overlay': {
            'version': VERSION, 'selection_sha256': digest(manifest),
            'prompt_guidance': GUIDANCE, 'prompt_guidance_sha256': digest(GUIDANCE),
            'boundary_policy': manifest['boundary_policy'],
            'implementation_sha256': file_digest(Path(__file__))}}
        manifest_path = output/'repair_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Source fallback selection changed; choose a fresh overlay output_root')
        saved_identity = output/'run_manifest.json'
        if saved_identity.exists() and json.loads(saved_identity.read_text()) != identity:
            raise ValueError('Repair construction identity changed; choose a fresh overlay output_root')
        preview = {'source_root': str(source), 'output_root': str(output), 'selected': len(selected),
                   'placeholder_children': sum(len(r['placeholder_children']) for r in selected),
                   'datasets': datasets, 'splits': splits, 'selection': 'existing_placeholder_fallback_only',
                   'requires_embedding': False, 'resegments_source': False, 'limit_sessions': limit}
        announce('topic_event_fallback_repair_plan', **preview)
        if args.dry_run:
            return preview
        endpoints = args.base_urls or ([args.base_url] if getattr(args, 'base_url', None) else None)
        if not endpoints or len(set(endpoints)) != len(endpoints):
            raise ValueError('Provide unique --base-urls or --base-url for the Qwen Writer')
        from .policy import load_tokenizer
        tokenizer = tokenizer or load_tokenizer(config['model_path'])
        count_tokens = lambda text: token_count(tokenizer, text)
        with file_lock(output/'.build.lock'):
            store = AdjacentBuildStore(output, identity)
            pool, counters = None, Counter()
            execution = {'kind': VERSION, 'started_at': time.time(), 'run_status': 'running'}
            try:
                atomic_json(manifest_path, manifest)
                pending = []
                for r in selected:
                    session, old = r['session'], r['directory']
                    validate_session(session, count_tokens=count_tokens)
                    validate_directory(old, session, count_tokens=count_tokens)
                    bank = deepcopy(r['bank'])
                    bank['source_map'] = [x for x in bank.get('source_map', [])
                                          if x.get('parent_id') == session['parent_id']]
                    store.add_bank(bank)
                    store.add_session(session)
                    status, directory = store.committed(session['bank_id'], session['parent_id'])
                    if status == 'complete':
                        validate_replacement(old, directory, session, count_tokens=count_tokens)
                        counters['resumed'] += 1
                    else:
                        pending.append(r)
                selected_work = pending[:limit] if limit else pending
                if selected_work:
                    pool = pool_factory(endpoints, config['workers'], config.get('service', {}), store)

                def repair(r):
                    session, previous = r['session'], r['directory']
                    writer = writer_factory(tokenizer, {**config['writer'],
                        'modality_policy': specs[r['dataset']].get('modality_policy', 'existing_captions')},
                        pool.for_session(session))
                    try:
                        result = writer.repair(session, previous)
                        validate_build_result(result, session, tokenizer, config['writer'])
                        if result['status'] != 'complete':
                            raise InfrastructureError('Repair remained fallback')
                        directory = bind_directory(session, result, identity['writer'], identity['tokenizer_id'],
                                                   count_tokens=count_tokens, prompt_version=topic.PROMPT_VERSION)
                        validate_replacement(previous, directory, session, count_tokens=count_tokens)
                        return {'directory': directory, 'stats': result.get('stats', {})}
                    except Exception as exc:
                        return {'error_type': type(exc).__name__, 'reason': str(exc)[:1600]}
                    finally:
                        for trace in writer.traces:
                            store.trace({'event': 'placeholder_repair_writer_trace',
                                         'bank_id': session['bank_id'], 'parent_id': session['parent_id'], **trace})

                for r, result in bounded_results(repair, selected_work, config['workers']):
                    if 'directory' in result:
                        store.commit(r['session'], result['directory'], result['stats'])
                        counters['repaired_now'] += 1
                    else:
                        store.failed(r['session'], result['error_type'], result['reason'])
                        counters['failed_now'] += 1
                    announce('topic_event_fallback_repair_progress', bank_id=r['session']['bank_id'],
                             parent_id=r['session']['parent_id'], status='complete' if 'directory' in result else 'failed',
                             **dict(counters), **{k:result[k] for k in ('error_type','reason') if k in result})
                execution.update(run_status='finished', **dict(counters))
            except BaseException:
                execution.update(run_status='interrupted', **dict(counters))
                raise
            finally:
                try:
                    if pool is not None:
                        pool.close()
                finally:
                    try:
                        report = _publish(store, selected, manifest, execution)
                    finally:
                        store.close()
        announce('topic_event_fallback_repair_finished', **{k:report[k] for k in ('status','selected','repaired','remaining','output_root')})
        if report['remaining'] and limit is None:
            raise InfrastructureError('Some fallback repairs remain incomplete; source directories are preserved; rerun to resume')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--source-root', default=DEFAULT_SOURCE)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--splits', nargs='+', choices=['train','validation','test'])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--output-root')
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument('--base-url')
    urls.add_argument('--base-urls', nargs='+')
    parser.add_argument('--limit-sessions', type=int)
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
