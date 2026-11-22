from pathlib import Path
import json,gzip,collections
P=Path('/data/home/wtma/AG-mem/runs/locomo_lmes_context3_open1_navneg/train')
metrics=[json.loads(l) for l in (P/'candidate_metrics.jsonl').read_text().splitlines()]
train=[json.loads(l) for l in (P/'train_metrics.jsonl').read_text().splitlines()]
completed={d['batch_id'] for d in train}

def compact_row(r):
 steps=r['reader_steps']; ids={x for s in steps for x in s.get('new_ids',[]) if ':c' in x}
 attempts={x for s in steps for x in (s.get('action') or {}).get('ids',[]) if ':c' in x}
 return dict(sample_id=r['sample_id'], J=r['judge_score'], A=r['advantage'], ids=sorted(ids), parents=sorted({x.split(':')[0] for x in ids}), attempts=sorted(attempts), reads=sum(bool(s.get('read_call_consumed')) for s in steps), navs=sum(s.get('action_kind')=='navigation' for s in steps), segments=sum(s.get('action_kind')=='segment' for s in steps), positive_read_decisions=sum(s.get('training_advantage',r['advantage'])>0 and bool(s.get('new_ids')) for s in steps),truncated=any(s.get('truncated',False) for s in steps))

def full_row(r):
 ids={e['id'] for e in r['evidence'] if ':c' in e['id']}
 attempts={x for s in r['decisions'] for x in (s.get('action') or {}).get('ids',[]) if ':c' in x}
 positive_steps=[s for s,a in zip(r['decisions'],r['action_advantages']['per_decision']) if a['advantage']>0 and s['completion']['token_ids']]
 positive_read_ids={x for s in positive_steps for x in (s.get('tool_result') or {}).get('new_ids',[]) if ':c' in x}
 return dict(sample_id=r['sample_id'],J=r['judge_score'], A=r['advantage'],ids=sorted(ids), parents=sorted({e.get('parent_id') or e['id'].split(':')[0] for e in r['evidence'] if ':c' in e['id']}), attempts=sorted(attempts),reads=r['n_read_calls'],truncated=any(s['completion'].get('truncated',False) for s in r['decisions']),tokens=sum(len(s['completion']['token_ids']) for s in r['decisions']),positive_decisions=len(positive_steps),positive_read_ids=sorted(positive_read_ids),positive_nav=sum((s.get('tool_result') or {}).get('action_kind')=='navigation' for s in positive_steps),positive_stop=sum((s.get('action') or {}).get('stop',False) for s in positive_steps),question=r['decisions'][0]['messages'][-1]['content'][:1000],prediction=r['prediction'])

def stats(rows):
 n=len(rows);o={'n':n,'J1':sum(r['J']==1 for r in rows),'J0':sum(r['J']==0 for r in rows),'reads_total':sum(r['reads'] for r in rows),'J1_reads_ge2':sum(r['J']==1 and r['reads']>=2 for r in rows),'J1_children_hist':dict(sorted(collections.Counter(len(r['ids']) for r in rows if r['J']==1).items())),'child2':sum(len(r['ids'])>=2 for r in rows),'J1_child2':sum(r['J']==1 and len(r['ids'])>=2 for r in rows),'J1_parent2':sum(r['J']==1 and len(r['parents'])>=2 for r in rows),'J0_child2':sum(r['J']==0 and len(r['ids'])>=2 for r in rows),'J0_attempt_child2':sum(r['J']==0 and len(r['attempts'])>=2 for r in rows),'J0_parent2':sum(r['J']==0 and len(r['parents'])>=2 for r in rows),'J1_child2_Apos':sum(r['J']==1 and len(r['ids'])>=2 and r['A']>0 for r in rows),'J1_parent2_Apos':sum(r['J']==1 and len(r['parents'])>=2 and r['A']>0 for r in rows)}
 if rows and 'tokens' in rows[0]:
  eligible=lambda r: not r['truncated'] and r['tokens']>0
  for pred,label in [(lambda r:len(r['ids'])>=2,'child2'),(lambda r:len(r['parents'])>=2,'parent2')]:
   good=[r for r in rows if r['J']==1 and pred(r) and r['A']>0 and eligible(r)]
   o[f'J1_{label}_Apos_unmasked']=len(good)
   o[f'J1_{label}_Apos_unmasked_groups']=len({r['sample_id'] for r in good})
   o[f'J1_{label}_unmasked_groups']=len({r['sample_id'] for r in rows if r['J']==1 and pred(r) and eligible(r)})
   o[f'J1_{label}_Apos_unmasked_positive_children']=sum(len(r['positive_read_ids']) for r in good)
  o['masked']=sum(r['truncated'] for r in rows)
  o['J1_child2_nonpositive_details']=[r for r in rows if r['J']==1 and len(r['ids'])>=2 and r['A']<=0]
 return o

out={'completed_train_rows':len(train),'batches':{},'selected_details':{},'in_progress':{},'candidate_group_details':[]}
for batch in sorted({m['batch_id'] for m in metrics}):
 ms=[m for m in metrics if m['batch_id']==batch];allrows=[];gs=[]
 for m in ms:
  d=json.loads((P/'candidate_rollouts'/f"candidate-{m['candidate_id']:08d}.json").read_text())
  rows=[compact_row(r) for r in d['rollouts']];allrows+=rows
  gs.append({'candidate_id':m['candidate_id'],'sample_id':m['sample_id'],'accepted':m['accepted'],'selected':m['selected'],'all_wrong':m['judge_all_wrong'],'all_correct':m['judge_all_correct'],**stats(rows)})
 cand={'groups':len(ms),**stats(allrows),'groups_with_J1_child2':sum(g['J1_child2']>0 for g in gs),'groups_with_J1_parent2':sum(g['J1_parent2']>0 for g in gs),'groups_with_J1_child2_Apos':sum(g['J1_child2_Apos']>0 for g in gs),'groups_with_J1_parent2_Apos':sum(g['J1_parent2_Apos']>0 for g in gs),'allwrong_groups':sum(g['all_wrong'] for g in gs),'allwrong_with_child2_groups':sum(g['all_wrong'] and g['child2']>0 for g in gs),'allwrong_with_attempt_child2_groups':sum(g['all_wrong'] and g['J0_attempt_child2']>0 for g in gs),'allwrong_child2_rows':sum(g['child2'] for g in gs if g['all_wrong']),'allwrong_attempt_child2_rows':sum(g['J0_attempt_child2'] for g in gs if g['all_wrong']),'allwrong_parent2_rows':sum(g['J0_parent2'] for g in gs if g['all_wrong']),'rejected_J1_child2':sum(g['J1_child2'] for g in gs if not g['selected']),'rejected_groups_with_J1_child2':sum(g['J1_child2']>0 for g in gs if not g['selected'])}
 out['candidate_group_details'] +=[{'batch':batch,**g} for g in gs]
 if batch not in completed: out['in_progress'][batch]=cand;continue
 idx=int(batch.split('-')[1]);p=P/'groups'/f'group-{idx:08d}.json.gz';d=json.load(gzip.open(p,'rt'));rows=[full_row(r) for r in d['rollouts']]
 out['batches'][batch]={'candidate':cand,'selected':stats(rows)}
 out['selected_details'][batch]=rows
Path('/tmp/audit_multiread_supply.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ['selected_details','candidate_group_details']},ensure_ascii=False,indent=2))
