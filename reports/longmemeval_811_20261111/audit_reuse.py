"""Read-only coverage audit for the 8:1:1 LongMemEval resplit.

No directory or source data is rewritten. Counts are per query-bank/session,
not globally unique session text and not predicted model request counts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq


def load_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metadata(root):
    result = {}
    for source in ['split_longmemeval_s', 'split_longmemeval_m']:
        for split in ['train', 'validation', 'test']:
            rows = pq.read_table(root / source / f'{split}.parquet',
                                columns=['question_id', 'haystack_session_ids']).to_pylist()
            for index, row in enumerate(rows):
                ids = row['haystack_session_ids']
                if isinstance(ids, str):
                    ids = json.loads(ids)
                if not isinstance(ids, list):
                    raise ValueError('Session IDs must decode to a list')
                result[source, split, index] = {'question_id': row['question_id'], 'sessions': len(ids)}
    return result


def aggregate(rows):
    counts = Counter()
    for row in rows:
        counts.update(row['session_states'])
    return {'queries': len(rows),
            'retained_queries': sum(r['old_split'] == r['new_split'] for r in rows),
            'moved_queries': sum(r['old_split'] != r['new_split'] for r in rows),
            'sessions': sum(r['sessions'] for r in rows),
            'session_states': {k: counts[k] for k in ['complete', 'empty', 'fallback', 'failed', 'pending', 'unregistered']},
            'fully_complete_queries': sum(r['fully_complete'] for r in rows),
            'queries_needing_new_sessions': sum(r['new_sessions_to_generate'] > 0 for r in rows),
            'queries_needing_only_repair': sum(r['new_sessions_to_generate'] == 0 and r['sessions_to_retry'] > 0 for r in rows),
            'new_sessions_to_generate': sum(r['new_sessions_to_generate'] for r in rows),
            'sessions_to_retry': sum(r['sessions_to_retry'] for r in rows),
            'sessions_not_ready': sum(r['sessions_not_ready'] for r in rows)}


def old_flat(root):
    examples = load_jsonl(root / 'test_examples.jsonl')
    qids = {r['judge_metadata']['question_id']: r['memory_id'] for r in examples}
    expected = {}
    with (root / 'memory_records.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            expected[row['memory_id']] = {r['id'] for r in row['records']}
    latest = {}
    with (root / 'directories.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            latest[row['memory_id'], row['version'], row['id']] = row
    versions = {key[1] for key in latest}
    if len(versions) != 1:
        raise ValueError(f'Expected one directory version in {root}; found {versions}')
    valid = {(mid, rid) for (mid, version, rid), row in latest.items()
             if row.get('writer_status') != 'empty_pending' and row.get('content', '').strip()}
    return qids, expected, valid, next(iter(versions))


def run(args):
    mappings = load_jsonl(args.data_root / 'row_mapping.jsonl')
    source = metadata(args.old_root)
    snapshot = json.loads(args.snapshot.read_text())
    banks = snapshot['banks']
    per_query = []
    for mapping in mappings:
        ds = mapping['source']
        if ds not in {'split_longmemeval_s', 'split_longmemeval_m'}:
            continue
        original = source[ds, mapping['old_split'], mapping['old_row_index']]
        if original['question_id'] != mapping['question_id']:
            raise ValueError('Row map differs from original question ID')
        n = original['sessions']
        bank = banks.get(mapping['old_bank_id'])
        if bank is not None:
            if bank['dataset'] != ds or not any(q['original_question_id'] == mapping['question_id']
                    and q['split'] == mapping['old_split'] and q['row_index'] == mapping['old_row_index']
                    for q in bank['query_memberships']):
                raise ValueError('Mapped bank provenance differs from source row')
        states = Counter()
        for index in range(1, n + 1):
            entry = (bank or {}).get('sessions', {}).get(f'p{index:04d}')
            state = 'unregistered' if entry is None else entry['status']
            if state in {'complete', 'empty'} and not entry['has_directory']:
                raise ValueError('Completed session has no persisted directory')
            states[state] += 1
        new = states['pending'] + states['unregistered']
        retry = states['fallback'] + states['failed']
        per_query.append({**mapping, 'sessions': n, 'session_states': dict(states),
                          'fully_complete': n == states['complete'] + states['empty'],
                          'new_sessions_to_generate': new, 'sessions_to_retry': retry,
                          'sessions_not_ready': new + retry})
    summaries = {}
    for ds in ['split_longmemeval_s', 'split_longmemeval_m']:
        rows = [r for r in per_query if r['source'] == ds]
        summaries[ds] = {'all': aggregate(rows),
                         'by_split': {split: aggregate([r for r in rows if r['new_split'] == split])
                                      for split in ['train', 'validation', 'test']},
                         'newly_added': {split: aggregate([r for r in rows if r['new_split'] == split
                                                            and r['old_split'] != split])
                                         for split in ['train', 'validation']}}
    prepared = {
        'qwen3.5-27B': Path('runs/evaluation/format05_best_final/prepared_test/split_longmemeval_s'),
        'gpt-5-mini': Path('runs/evaluation/longmemeval_gpt5mini/prepared_test/split_longmemeval_s'),
    }
    flat = {label: old_flat(root) for label, root in prepared.items()}
    test_rows = [r for r in per_query if r['source'] == 'split_longmemeval_s' and r['new_split'] == 'test']
    flat_queries = []
    for row in test_rows:
        qid = row['question_id']
        expected, union = None, set()
        counts = {}
        for label, (qids, memories, valid, version) in flat.items():
            mid = qids[qid]
            ids = memories[mid]
            if len(ids) != row['sessions']:
                raise ValueError('Flat original session count differs from resplit history')
            if expected is None:
                expected = {(mid, rid) for rid in ids}
            elif expected != {(mid, rid) for rid in ids}:
                raise ValueError('Flat Writer roots disagree on original history addresses')
            good = valid & expected
            counts[label] = len(good)
            union.update(good)
        flat_queries.append({'question_id': qid, 'new_row_index': row['new_row_index'],
                             'old_row_index': row['old_row_index'], 'sessions': row['sessions'],
                             'available_by_writer': counts, 'available_union': len(union),
                             'missing_union': len(expected - union), 'complete_union': union == expected})
    flat_summary = {
        'queries': len(flat_queries), 'sessions': sum(r['sessions'] for r in flat_queries),
        'by_writer': {label: {'available_sessions': sum(r['available_by_writer'][label] for r in flat_queries),
                             'fully_complete_queries': sum(r['available_by_writer'][label] == r['sessions'] for r in flat_queries),
                             'source_root': str(root), 'directory_version': flat[label][3]}
                      for label, root in prepared.items()},
        'available_union': sum(r['available_union'] for r in flat_queries),
        'new_sessions_to_generate': sum(r['missing_union'] for r in flat_queries),
        'fully_complete_queries': sum(r['complete_union'] for r in flat_queries),
        'queries_needing_new_sessions': sum(r['missing_union'] > 0 for r in flat_queries),
    }
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'directory_snapshot_time': snapshot['taken_at'],
              'directory_snapshot_source': snapshot['source_journal'],
              'data_root': str(args.data_root.resolve()), 'old_root': str(args.old_root.resolve()),
              'counting_unit': 'one directory session per query-bank; underlying source sessions may overlap',
              'reused_materialized': False,
              'notes': ['Only resplit data has been materialized. Existing directories require explicit source-verified rebinding before use.',
                        'Complete directories are reusable; fallback/failed are repairs, pending/unregistered are newly generated sessions.',
                        'The hierarchical audit uses the adjacent Qwen4B protocol only; older non-clustered parent-child directories are not interchangeable.',
                        'Future exact model request counts may be lower due to compatible content caches and higher due to multiple children/retries.'],
              'hierarchical': summaries,
              'single_layer_orm_new_s_test': flat_summary}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'coverage_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'per_query_coverage.jsonl').open('w') as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    with (args.output / 'single_layer_test_coverage.jsonl').open('w') as f:
        for row in flat_queries:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/processed/longmemeval_811_seed42_20261111'))
    parser.add_argument('--old-root', type=Path, default=Path('data/processed/query_dedup_val10_seed42_20261018'))
    parser.add_argument('--snapshot', type=Path, default=Path('reports/longmemeval_811_20261111/directory_status_snapshot.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/longmemeval_811_20261111'))
    run(parser.parse_args())
