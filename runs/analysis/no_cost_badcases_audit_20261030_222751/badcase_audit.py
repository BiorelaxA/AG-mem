"""Read-only coverage audit. Annotation coverage is a diagnostic, not a causal label."""
from collections import Counter
from pathlib import Path
import importlib.util
import json
import re
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SNAP = json.loads((OUT / 'snapshot.json').read_text())
RUN = ROOT / SNAP['run']
CONFIG = json.loads((RUN / 'run.json').read_text())['config']
RAW = Path(CONFIG['data']['input_root'])
PREP = Path(CONFIG['data']['prepared_root'])

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def clean(s):
    return re.sub(r'^(?:answer_)+', '', str(s))

meta = {}
record_maps = {}
for source in CONFIG['data']['sources']:
    pf = pq.ParquetFile(RAW / source / 'train.parquet')
    cols = ['question_id', 'question', 'answer', 'answer_session_ids']
    if 'question_type' in pf.schema_arrow.names:
        cols.append('question_type')
    raw = pf.read(columns=cols).to_pylist()
    examples = {x['sample_id']: x for x in read_jsonl(PREP / source / 'train_examples.jsonl')}
    for i, x in enumerate(raw):
        sid = f'{source}/train/{i:06d}'
        ex = examples[sid]
        assert ex['query'] == x['question'] and ex['gold_answers'] == [x['answer']]
        gold = x['answer_session_ids']
        if isinstance(gold, str):
            gold = json.loads(gold)
        meta[sid] = {**x, 'source': source, 'memory_id': ex['memory_id'],
                     'gold_sessions': sorted({clean(v) for v in gold})}
    # Existing offsets are read without instantiating a Store or rewriting indexes.
    idx = json.loads((PREP / source / 'memory_records.index.json').read_text())
    offsets = {key[0]: offset for key, offset in idx['offsets']}
    with (PREP / source / 'memory_records.jsonl').open('rb') as f:
        for mid in {x['memory_id'] for x in examples.values()}:
            f.seek(offsets[mid]); memory = json.loads(f.readline())
            assert memory['memory_id'] == mid
            record_maps[(source, mid)] = {x['id']: clean(x['source_session_id']) for x in memory['records']}

metrics = read_jsonl(OUT / 'candidate_metrics.jsonl')
all_groups, bad_groups = [], []
for m in metrics:
    d = json.loads((RUN / 'candidate_rollouts' / f"candidate-{m['candidate_id']:08d}.json").read_text())
    md = meta[m['sample_id']]
    mapping = record_maps[(md['source'], md['memory_id'])]
    gold_s = set(md['gold_sessions'])
    gold_ids = {k for k, v in mapping.items() if v in gold_s}
    assert {mapping[k] for k in gold_ids} == gold_s
    rows = []
    for r in d['rollouts']:
        steps = r['reader_steps']
        actual_ids = {v for s in steps for v in s.get('new_ids', [])}
        assert actual_ids <= mapping.keys()
        actual_s = {mapping[v] for v in actual_ids}
        hits = actual_s & gold_s
        requested, excluded = set(), set()
        for s in steps:
            a = s.get('action') or {}
            if not a.get('stop', True):
                ids = a.get('ids', [])
                requested.update(ids)
                excluded.update(ids[CONFIG['retrieval']['max_ids_per_call']:])
        category = ('empty' if not actual_s else 'no_annotated_hit' if not hits
                    else 'full_annotated_coverage' if gold_s <= actual_s else 'partial_annotated_coverage')
        flags = [flag for s in steps for flag in s.get('format_violations', [])]
        pred = r['task_agent_output']
        rows.append({'rollout': r['rollout_index'], 'judge': r['judge_score'], 'reward': r['reward'],
                     'prediction': pred, 'answer_f1': r['outcome_f1'], 'category': category,
                     'actual_ids': sorted(actual_ids), 'actual_sessions': sorted(actual_s),
                     'hit_sessions': sorted(hits), 'missing_sessions': sorted(gold_s - actual_s),
                     'calls': sum(bool(s.get('read_call_consumed')) for s in steps),
                     'fresh_reads': len(actual_ids), 'termination': r['termination_reason'],
                     'format_flags': flags, 'format_count': r['format_violation_count'],
                     'requested_gold_never_read': sorted((requested & gold_ids) - actual_ids),
                     'overlimit_gold_never_read': sorted((excluded & gold_ids) - actual_ids),
                     'explicit_unknown_answer': bool(re.search(r"\bi (?:do not|don't|don’t) know\b|not enough information|insufficient information", pred, re.I)),
                     'actions': [s.get('action') for s in steps]})
    assert len(rows) == 8
    if all(not r['actual_ids'] for r in rows):
        category = 'all8_empty'
    elif all(not r['hit_sessions'] for r in rows):
        category = 'none_of8_hit_annotated_support'
    elif any(r['category'] == 'full_annotated_coverage' for r in rows):
        category = 'at_least1_full_coverage_but_allwrong'
    else:
        category = 'some_partial_hit_none_full_coverage'
    g = {'candidate': m['candidate_id'], 'policy': m['behavior_policy_version'], 'selected': m['selected'],
         'accepted': m['accepted'], 'all_wrong': m['judge_all_wrong'], 'sample_id': m['sample_id'],
         **md, 'support_count': len(gold_s), 'gold_reader_ids': sorted(gold_ids),
         'coverage_group_category': category, 'rows': rows}
    all_groups.append(g)
    if m['judge_all_wrong']:
        assert all(r['judge'] == 0 for r in rows)
        bad_groups.append(g)

def summarize(groups):
    rows = [r for g in groups for r in g['rows']]
    return {'groups': len(groups), 'rollouts': len(rows),
            'sources': dict(Counter(g['source'] for g in groups)),
            'group_coverage_categories': dict(Counter(g['coverage_group_category'] for g in groups)),
            'support_count_histogram': dict(sorted(Counter(g['support_count'] for g in groups).items())),
            'rollout_coverage_categories': dict(Counter(r['category'] for r in rows)),
            'fresh_read_histogram': dict(sorted(Counter(r['fresh_reads'] for r in rows).items())),
            'termination': dict(Counter(r['termination'] for r in rows)),
            'explicit_unknown_answers': sum(r['explicit_unknown_answer'] for r in rows),
            'format_rollouts': sum(bool(r['format_flags']) for r in rows),
            'groups_with_format': sum(any(r['format_flags'] for r in g['rows']) for g in groups),
            'format_decisions_by_type': dict(Counter(f for r in rows for f in r['format_flags'])),
            'groups_with_overlimit_gold_omitted': sum(any(r['overlimit_gold_never_read'] for r in g['rows']) for g in groups),
            'rollouts_with_overlimit_gold_omitted': sum(bool(r['overlimit_gold_never_read']) for r in rows),
            'groups_all8_full_coverage': sum(all(r['category'] == 'full_annotated_coverage' for r in g['rows']) for g in groups),
            'max_answer_f1': max((r['answer_f1'] for r in rows), default=None)}

report = {'captured_at': SNAP['captured_at'], 'definition': 'All-wrong means all 8 Judge labels equal 0. Support coverage is descriptive annotation matching, not proven minimal evidence or a causal explanation.',
          'total_candidate_groups': len(metrics), 'all_wrong': summarize(bad_groups),
          'by_policy': {str(v): summarize([g for g in bad_groups if g['policy'] == v]) for v in sorted({g['policy'] for g in all_groups})},
          'by_source': {s: summarize([g for g in bad_groups if g['source'] == s]) for s in CONFIG['data']['sources']},
          'suspected_high_f1_judge_disagreements': [
              {'candidate': g['candidate'], 'sample_id': g['sample_id'], 'question': g['question'], 'gold': g['answer'],
               'max_f1': max(r['answer_f1'] for r in g['rows']),
               'answers': sorted({r['prediction'] for r in g['rows']})}
              for g in bad_groups if any(r['answer_f1'] >= .8 for r in g['rows'])]}
(OUT / 'allwrong_coverage_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
(OUT / 'allwrong_groups.json').write_text(json.dumps(bad_groups, ensure_ascii=False, indent=2)+'\n')
(OUT / 'all_candidate_support_audit.json').write_text(json.dumps(all_groups, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
