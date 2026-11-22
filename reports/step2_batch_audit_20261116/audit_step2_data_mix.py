import collections
import json
import pathlib
import random
import statistics

import pyarrow.parquet as pq

ROOT = pathlib.Path('/data/home/wtma/AG-mem')
PREP = ROOT / 'runs/locomo_lmes_context3_open1_nav06/prepared'
NAMES = ['navneg', 'nav06']
RUNS = {name: ROOT / f'runs/locomo_lmes_context3_open1_{name}/train' for name in NAMES}
BATCHES = ['batch-00000000', 'batch-00000001']
def jl(p):
    return [json.loads(line) for line in open(p)]
examples = jl(PREP / 'train_examples.jsonl')
by_id = {e['sample_id']: e for e in examples}
metrics = {n: [r for r in jl(p/'candidate_metrics.jsonl') if r['batch_id'] in BATCHES] for n,p in RUNS.items()}
ids = {r['sample_id'] for rows in metrics.values() for r in rows}
raw = {}
paths = {'locomo':'locomo_query9010_seed42_20261102', 'split_longmemeval_s':'longmemeval_315_35_150_seed42_20261113'}
for source, directory in paths.items():
    for i,row in enumerate(pq.read_table(ROOT/f'data/processed/{directory}/{source}/train.parquet').to_pylist()):
        sid = f'{source}/train/{i:06d}'
        if sid in ids:
            raw[sid] = row

banks = {}
for line in open(PREP/'hierarchical_banks.jsonl'):
    bank = json.loads(line)
    memberships = {r['sample_id'] for r in bank['query_memberships']} & ids
    if not memberships:
        continue
    session_map = {r['parent_id']:r['original_session_id'] for r in bank['source_map']}
    for sid in memberships:
        banks[sid] = {'bank_id':bank['bank_id'], 'session_map':session_map,
                      'gold_parents': sorted(p for p,s in session_map.items() if s in raw[sid]['answer_session_ids'])}

questions = {}
for sid in sorted(ids):
    row = raw[sid]
    questions[sid] = {**by_id[sid], 'question_id':row['question_id'],
        'question_type':row.get('question_type'), 'locomo_official_category':None,
        'abs':row['question_id'].endswith('_abs'),
        'annotated_answer_session_count':len(row['answer_session_ids']),
        'answer_session_ids':row['answer_session_ids'],
        'total_session_count':len(row['haystack_session_ids']),
        'question_form_time_heuristic':row['question'].lower().startswith(('when ', 'what month ', 'which event ')),
        **banks[sid]}

result = {'scope':'read-only first two completed sample/train batches; no model calls',
          'category_limit':'LoCo prepared judge_metadata is empty and source Parquet has no category column; no official class inferred.',
          'coverage_limit':'Annotated answer session overlap is a session-level proxy, not evidence completeness or causality.',
          'questions':questions, 'runs':{}, 'comparisons':{}}

for name, path in RUNS.items():
    cand = []
    for row in metrics[name]:
        sid = row['sample_id']; gold = set(questions[sid]['gold_parents'])
        rolls = json.load(open(path/f'candidate_rollouts/candidate-{row["candidate_id"]:08d}.json'))['rollouts']
        rr = []
        for r in rolls:
            read_ids = sorted({i for step in r['reader_steps'] for i in step.get('new_ids', [])})
            parents = {i.split(':')[0] for i in read_ids}
            rr.append({'rollout_index':r['rollout_index'], 'J':r['judge_score'], 'F1':r['outcome_f1'],
                       'q':(r['outcome_reward']+1)/2, 'R':r['reward'],
                       'outcome_reward':r['outcome_reward'], 'read_cost':r['read_cost'],
                       'format_reward':r['format_reward'], 'read_ids':read_ids,
                       'annotated_answer_parent_hits':sorted(gold&parents),
                       'annotated_answer_session_recall':len(gold&parents)/len(gold) if gold else None,
                       'empty_evidence':not read_ids, 'answer':r['task_agent_output'],
                       'termination_reason':r['termination_reason'],
                       'actions':[s.get('action') for s in r['reader_steps']]})
        cand.append({'metrics':row, 'rollouts':rr})
    train = [r for r in jl(path/'train_metrics.jsonl') if r['batch_id'] in BATCHES]
    sampling = [r for r in jl(path/'sampling_metrics.jsonl') if r['batch_id'] in BATCHES]
    summaries=[]
    for b in BATCHES:
        rows=[r for r in cand if r['metrics']['batch_id']==b]
        for selected in [False,True]:
            items=[r for r in rows if r['metrics']['selected'] or not selected]
            ms=[r['metrics'] for r in items]
            qs=[questions[r['sample_id']] for r in ms]
            rs=[r for x in items for r in x['rollouts']]
            summaries.append({'batch':b,'population':'selected' if selected else 'candidate',
                'group_count':len(ms), 'source_counts':dict(collections.Counter(r['source'] for r in ms)),
                'question_types':dict(collections.Counter(q['question_type'] or 'locomo_unavailable' for q in qs)),
                'abs_count':sum(q['abs'] for q in qs),
                'annotated_multisession_count':sum(q['annotated_answer_session_count']>1 for q in qs),
                'time_question_heuristic_count':sum(q['question_form_time_heuristic'] for q in qs),
                'mean_answer_session_count':statistics.mean(q['annotated_answer_session_count'] for q in qs),
                'mean_total_session_count':statistics.mean(q['total_session_count'] for q in qs),
                'J':statistics.mean(r['J'] for r in rs),'F1':statistics.mean(r['F1'] for r in rs),
                'q':statistics.mean(r['q'] for r in rs),'R':statistics.mean(r['R'] for r in rs),
                'all_wrong_count':sum(r['judge_all_wrong'] for r in ms),
                'all_correct_count':sum(r['judge_all_correct'] for r in ms),
                'success_1_or_2_count':sum(r['judge_accuracy'] in [.125,.25] for r in ms),
                'success_histogram':dict(sorted(collections.Counter(round(r['judge_accuracy']*8) for r in ms).items())),
                'empty_evidence_rate':statistics.mean(r['empty_evidence'] for r in rs),
                'mean_annotated_session_recall':statistics.mean(r['annotated_answer_session_recall'] for r in rs if r['annotated_answer_session_recall'] is not None),
                'read_cost':statistics.mean(r['read_cost'] for r in rs),
                'format_reward':statistics.mean(r['format_reward'] for r in rs)})
    result['runs'][name]={'summaries':summaries,'train':train,'sampling':sampling,'candidates':cand}

ordered=list(examples);random.Random(42).shuffle(ordered)
result['comparisons']['shuffle42_matches_first29'] = all([r['sample_id'] for r in metrics[n]]==[e['sample_id'] for e in ordered[:29]] for n in NAMES)
result['comparisons']['candidate_sequence_equal'] = [r['sample_id'] for r in metrics['navneg']]==[r['sample_id'] for r in metrics['nav06']]
result['comparisons']['selected_sequence_equal'] = [r['sample_id'] for r in metrics['navneg'] if r['selected']]==[r['sample_id'] for r in metrics['nav06'] if r['selected']]
result['comparisons']['matching_judge_counts_step2'] = sum(a['judge_accuracy']==b['judge_accuracy'] for a,b in zip(metrics['navneg'][12:],metrics['nav06'][12:]))
out=pathlib.Path('/tmp/audit_step2_data_mix.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(out)
print(json.dumps(result['comparisons'],ensure_ascii=False))
for name,r in result['runs'].items():
    print(name)
    for s in r['summaries']: print(json.dumps(s,ensure_ascii=False))
print('QUESTIONS')
for r in metrics['navneg']:
    q=questions[r['sample_id']]
    print(r['candidate_id'],q['sample_id'],'selected',r['selected'],'sessions',q['annotated_answer_session_count'],'/',q['total_session_count'],'gold_parents',q['gold_parents'],'gold',q['gold_answers'])
