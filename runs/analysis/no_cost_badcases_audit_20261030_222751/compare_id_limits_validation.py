"""Read-only paired comparison of ID limits on fixed validation queries."""
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / 'runs/locomo_longmemeval/experiments'
PREFIX = 'locomo_s_dapo_mixed_v2_micro2_parallel4_reader_ids_v2_format_count'
RUNS = {'max4_cost': '', 'max2_cost': '_max2', 'max1_cost': '_max1',
        'max1_no_cost': '_max1_no_call_cost'}

def readj(path):
    return json.loads(path.read_text())

def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

report = {'runs': {}, 'paired_comparisons': {}, 'notes': [
    'All comparisons use the same fixed 20 validation queries (10 per source), verified below.',
    'Outcome_reward is mean(2*Judge-1); it does not include read cost or format penalties.',
    'Historical training runs are not a controlled same-checkpoint inference ablation. The ID limit changes prompts, sampled trajectories, selected groups and resulting model weights.',
    'max4_cost has only validation16; validation32 must not be substituted with another recipe or validation25.',
    'Same model fingerprints refer to recorded metadata fingerprints, not full shard hashing.'
]}
episodes = {}
reference_config = None
reference_ids = None
reference_initial_states = {}
for label, suffix in RUNS.items():
    runpath = BASE / (PREFIX + suffix) / 'train'
    run = readj(runpath / 'run.json'); config = run['config']
    examples = {}
    for source in config['data']['sources']:
        for row in jsonl(Path(config['data']['prepared_root']) / source / 'validation_examples.jsonl'):
            examples[row['sample_id']] = row
    selected_training = {key: config['training'].get(key) for key in (
        'learning_rate','group_size','candidate_groups','target_groups','max_num_gen_batches',
        'filter_metric','mini_batch_groups','micro_batch_rollouts','clip_ratio_low','clip_ratio_high',
        'overlong_filter','epochs','validation_every','parameterization','optimizer','kl_coefficient')}
    judge = run['scoring_protocol']['judge']
    comparable = {'seed':config['seed'],'data':config['data'], 'reader':config['reader'],
                  'directory_version':run['directory_version'], 'prepare_identity':run['prepare_identity'],
                  'task_agent':config['task_agent'],'training':selected_training,
                  'format_penalty':config['reward']['format_penalty'],
                  'format_penalty_mode':config['reward']['format_penalty_mode'],
                  'formula':config['reward']['formula'],
                  'reader_fingerprint':run['reader']['fingerprint'],
                  'task_fingerprint':run['task']['fingerprint'],
                  'task_input_contract':run['task_input_contract'],
                  'judge_templates':judge['template_sha256'],
                  'judge_protocols':judge['protocols'],
                  'judge_temperature':judge['temperature'], 'judge_seed':judge['seed']}
    if reference_config is None: reference_config = comparable
    record = {'path':str(runpath.relative_to(ROOT)), 'max_ids_per_call':config['retrieval']['max_ids_per_call'],
              'max_read_calls':config['retrieval']['max_read_calls'],
              'read_call_cost':config['reward']['read_call_cost'],
              'reader_input_contract':run['reader_input_contract'],
              'comparable_config':comparable,
              'config_equal_to_max4':{key: val == reference_config[key] for key,val in comparable.items()},
              'validation':{}}
    episodes[label] = {}
    for step in (16,32,48,64):
        directory=runpath/'validation'/f'step-{step:08d}'
        if not (directory/'metrics.json').exists(): continue
        metric=readj(directory/'metrics.json'); raw=jsonl(directory/'episodes.jsonl')
        ids=sorted(row['sample_id'] for row in raw)
        if reference_ids is None:reference_ids=ids
        assert ids == reference_ids
        assert len(ids)==len(set(ids))==20
        rows={}
        for row in raw:
            sid=row['sample_id']; ex=examples[sid]
            initial_state=next(m['content'] for m in row['decisions'][0]['messages'] if m['role']=='user')
            reference_initial_states.setdefault(sid,initial_state)
            assert initial_state == reference_initial_states[sid], (label,step,sid,'different query/observation/directory initial state')
            rows[sid]={'sample_id':sid,'query':ex['query'],'gold':ex['gold_answers'],
                       'judge':row['judge_score'],'prediction':row['prediction'],
                       'read_calls':row['n_read_calls'],'unique_ids':len({e['id'] for e in row['evidence']}),
                       'evidence_ids':[e['id'] for e in row['evidence']],
                       'source_session_ids':[e['source_session_id'] for e in row['evidence']],
                       'format_count':row['format_violation_count'],
                       'actions':[d.get('action') for d in row['decisions']],
                       'termination_reason':row['termination_reason']}
        correct=sum(row['judge_score'] for row in raw)
        assert abs(metric['pooled']['outcome_reward']-(2*correct/20-1))<1e-12
        episodes[label][step]=rows
        record['validation'][str(step)]={'correct':int(correct),'n':20,
              **{k:metric['pooled'][k] for k in ['judge_accuracy','outcome_reward','reward','read_calls',
                    'unique_ids_read','answer_f1','format_failure_rate','format_violation_count',
                    'repeated_read_rate','empty_evidence_rate']},
              'by_source':{source:{'correct':int(round(x['judge_accuracy']*x['n'])),
                            **{k:x[k] for k in ['n','judge_accuracy','outcome_reward','reward','read_calls','unique_ids_read','answer_f1']}}
                          for source,x in metric['by_source'].items()},
              'call_histogram':dict(Counter(row['read_calls'] for row in rows.values())),
              'unique_id_histogram':dict(Counter(row['unique_ids'] for row in rows.values())),
              'all_same_query_ids':ids==reference_ids,
              'all_same_initial_user_states':True}
    report['runs'][label]=record

for oldlabel,newlabel,step in [('max4_cost','max1_no_cost',16),('max2_cost','max1_no_cost',16),
                             ('max2_cost','max1_no_cost',32),('max2_cost','max1_no_cost',48),
                             ('max1_cost','max1_no_cost',16),('max1_cost','max1_no_cost',32),
                             ('max1_cost','max1_no_cost',48)]:
    old,new=episodes[oldlabel][step],episodes[newlabel][step]
    changed=[]
    for sid in sorted(old):
        assert old[sid]['query']==new[sid]['query'] and old[sid]['gold']==new[sid]['gold']
        if old[sid]['judge']!=new[sid]['judge']:
            changed.append({'sample_id':sid,'query':old[sid]['query'],'gold':old[sid]['gold'],
                            'old':old[sid],'new':new[sid]})
    report['paired_comparisons'][f'{oldlabel}_to_{newlabel}_step{step}']={
        'lost_correct':sum(x['old']['judge']==1 for x in changed),
        'gained_correct':sum(x['new']['judge']==1 for x in changed),
        'same_answers':sum(old[sid]['prediction']==new[sid]['prediction'] for sid in old),
        'same_evidence':sum(old[sid]['evidence_ids']==new[sid]['evidence_ids'] for sid in old),
        'changed_questions':changed}

path=OUT/'id_limit_validation_comparison.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for label,run in report['runs'].items():
    print(label,'config_diffs',[k for k,v in run['config_equal_to_max4'].items() if not v])
    for step,row in run['validation'].items():
        print(step,{k:row[k] for k in ['correct','outcome_reward','reward','read_calls','unique_ids_read']})
for key,value in report['paired_comparisons'].items():
    print(key,{k:v for k,v in value.items() if k!='changed_questions'})
    if key=='max2_cost_to_max1_no_cost_step32':
        for row in value['changed_questions']:
            print(row['sample_id'],row['query'],'gold',row['gold'],'OLD',row['old']['prediction'],row['old']['source_session_ids'],
                  'NEW',row['new']['prediction'],row['new']['source_session_ids'])
print(path)
