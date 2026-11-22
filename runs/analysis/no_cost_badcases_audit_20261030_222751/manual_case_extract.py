import sys,json,re
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path.cwd()))
import pyarrow.parquet as pq
from agmem.store import Store
out=Path('runs/analysis/no_cost_badcases_audit_20261030_222751')
run=Path(json.loads((out/'snapshot.json').read_text())['run']);conf=json.loads((run/'run.json').read_text())['config'];rawroot=Path(conf['data']['input_root']);prep=Path(conf['data']['prepared_root'])
meta={};examples={};stores={}
for source in conf['data']['sources']:
 rows=pq.read_table(rawroot/source/'train.parquet',columns=['question_id','question','answer','answer_session_ids']).to_pylist()
 meta.update({f'{source}/train/{i:06d}':r for i,r in enumerate(rows)})
 examples.update({r['sample_id']:r for r in map(json.loads,(prep/source/'train_examples.jsonl').read_text().splitlines())})
 stores[source]=Store(prep/source,memory_cache_size=12)
normal=lambda s:re.sub(r'^(?:answer_)+','',s)
ms=[json.loads(l) for l in (out/'candidate_metrics.jsonl').read_text().splitlines()]
results=[]
for m in ms:
 if not m['judge_all_wrong'] or m['behavior_policy_version']<32:continue
 sid=m['sample_id'];source=sid.split('/')[0];raw=meta[sid];e=examples[sid];records=stores[source]._records(e['memory_id']);by_id={r['id']:r for r in records};gold=set(map(normal,raw['answer_session_ids']));goldids={r['id'] for r in records if normal(r['source_session_id']) in gold};assert len(goldids)==len(gold)
 d=json.loads((run/'candidate_rollouts'/f"candidate-{m['candidate_id']:08d}.json").read_text())
 cases=[]
 for row in d['rollouts']:
  ids=[v for s in row['reader_steps'] for v in s.get('new_ids',[])];hits=set(ids)&goldids
  cases.append({'rollout':row['rollout_index'],'ids':ids,'gold_hits':list(hits),'covered_all':hits==goldids,'answer':row['task_agent_output'],'judge':row['judge_score'],'steps':row['reader_steps'],'termination_reason':row['termination_reason']})
 record={'candidate':m['candidate_id'],'policy':m['behavior_policy_version'],'sample_id':sid,**raw,'gold_directory_ids':list(goldids),'question_date':e.get('observation'),'rows':cases,'records':[{k:v for k,v in r.items()} for r in records if r['id'] in goldids|{x for c in cases for x in c['ids']}]}
 results.append(record)
 print(m['candidate_id'],sid,'support',len(gold),'full',sum(x['covered_all'] for x in cases),'hit',sum(bool(x['gold_hits']) for x in cases),'Q',raw['question'],'GOLD',raw['answer'],'PRED',list({x['answer'] for x in cases}))
(out/'badcase_manual_extracted.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
