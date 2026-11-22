"""Read-only final-boundary causal trace accounting, not counting rejected attempts."""
import collections, datetime, json, pathlib, sqlite3
from agmem.core import digest
from agmem.hierarchical.writer import public_session
OUT=pathlib.Path('reports/directory_semantic_audit_20261111')
cn=sqlite3.connect('file:runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1/build_journal.sqlite3?mode=ro',uri=True)
co=sqlite3.connect('file:runs/parent_child_directories/qwen27b_child300_v1/build_journal.sqlite3?mode=ro',uri=True)
ident=json.loads(cn.execute("select value from metadata where key='identity'").fetchone()[0])
r=cn.execute("select value from metadata where key='parent_budget_migrations'").fetchone()
versions=[ident]+[m['from_identity'] for m in json.loads(r[0] if r else '[]')]
versions=list({digest(v):v for v in versions}.values())
metadata=[];boundaries=[];failures=[]
banks=cn.execute("select bank_id from banks where dataset='locomo' order by bank_id").fetchall()
for (bank,) in banks:
 rows=cn.execute("select parent_id,source_hash,status,snapshot,directory from sessions where bank_id=? order by parent_id",(bank,)).fetchall()
 for pid,sh,status,sjs,djs in rows:
  oldrow=co.execute('select source_hash,status,directory from sessions where bank_id=? and parent_id=?',(bank,pid)).fetchone()
  if status!='complete' or not oldrow or oldrow[1]!='complete' or sh!=oldrow[0]:raise RuntimeError('Unpaired '+bank+pid)
  s=json.loads(sjs);n=json.loads(djs);o=json.loads(oldrow[2]);nspans=[(x['source']['start_block'],x['source']['end_block']) for x in n['children']];oldbounds={x['source']['end_block'] for x in o['children'][:-1]}
  candidates=[]
  for v in versions:
   key=digest({'input':public_session(s,'existing_captions'),'construction':v,'modality_policy':'existing_captions'})
   found=cn.execute('select payload from segment_cache where key=?',(key,)).fetchone()
   if found:
    seg=json.loads(found[0]);matches=[(x['start_block'],x['end_block']) for x in seg['spans']]==nspans
    candidates.append({'key':key,'parent_budget':v['writer_settings']['parent_content_tokens'],'matches':matches,'seg':seg})
  exact=[x for x in candidates if x['matches']]
  selected=exact[0] if exact else None
  m={'bank_id':bank,'parent_id':pid,'old_children':len(o['children']),'new_children':len(n['children']), 'children_increased':len(n['children'])>len(o['children']),'candidate_cache_count':len(candidates),'exact_cache_count':len(exact), 'selected_key':selected['key'] if selected else None,'selected_parent_budget':selected['parent_budget'] if selected else None}
  metadata.append(m)
  if not selected:failures.append(m)
  trace=selected['seg']['traces'] if selected else []
  for l,r in zip(nspans,nspans[1:]):
   # Use only matching final span membership; last event if duplicates.
   matched=[(i,t) for i,t in enumerate(trace) if t.get('event')=='adjacent_boundary_kept' and t.get('left')==list(l) and t.get('right')==list(r)]
   event_index,t=matched[-1] if matched else (None,{})
   score=t.get('score');cohesion=t.get('cohesion')
   row={'bank_id':bank,'parent_id':pid,'boundary_after_block':l[1],'left':list(l),'right':list(r),'old_has_boundary':l[1] in oldbounds,'added_vs_old':l[1] not in oldbounds,'children_increased':m['children_increased'],'reason':t.get('reason','missing_final_pair_event'),'score':score,'score_bin':('missing' if score is None else 'below_0.55' if score<0.55 else '0.55_to_below_0.78' if score<0.78 else 'at_least_0.78'),'cohesion':cohesion,'cohesion_below_0.4':cohesion is not None and cohesion<0.4,'matched_event_count':len(matched),'event_index':event_index,'cache_key':m['selected_key'],'review':t.get('review')}
   boundaries.append(row)
cn.close();co.close()

def summarize(rr):
 c=collections.Counter(r['reason'] for r in rr);s=collections.Counter(r['score_bin'] for r in rr)
 detail={}
 for reason in sorted(c):
  x=[r for r in rr if r['reason']==reason]
  detail[reason]={'count':len(x),'percent':100*len(x)/len(rr) if rr else 0,'score_bins':dict(collections.Counter(r['score_bin'] for r in x)), 'cohesion_known':sum(r['cohesion'] is not None for r in x),'cohesion_below_0.4':sum(r['cohesion_below_0.4'] for r in x)}
 return {'boundary_count':len(rr),'sessions':len({(r['bank_id'],r['parent_id']) for r in rr}),'reasons':dict(c),'reason_details':detail,'score_bins':dict(s),'cohesion_known':sum(r['cohesion'] is not None for r in rr),'cohesion_below_0.4':sum(r['cohesion_below_0.4'] for r in rr),'missing_final_pair_events':c['missing_final_pair_event']}
report={'timestamp':datetime.datetime.now().astimezone().isoformat(),'method':'For each adjacent pair of final child spans, choose LAST adjacent_boundary_kept event whose left and right span memberships exactly equal those final spans. Count final boundaries once, never all rejected attempts. Final session-end excluded. Compare old boundary end_block indices on same source_hash.',
'sessions':len(metadata),'old_children':sum(m['old_children'] for m in metadata),'new_children':sum(m['new_children'] for m in metadata),'sessions_children_increased':sum(m['children_increased'] for m in metadata),'cache_missing_or_spans_mismatch':failures,'cache_identity_parent_budget_counts':dict(collections.Counter(m['selected_parent_budget'] for m in metadata)),
'all_new_final_boundaries':summarize(boundaries),'added_final_boundaries_vs_old':summarize([r for r in boundaries if r['added_vs_old']]),'added_final_boundaries_in_more_children_sessions':summarize([r for r in boundaries if r['added_vs_old'] and r['children_increased']]),
'all_final_boundaries_in_more_children_sessions':summarize([r for r in boundaries if r['children_increased']]),
'notes':['Reasons low_similarity, review_keep_boundary, review_budget_exhausted and token_budget are distinct. Score bin alone does not imply model review if the budget was exhausted.','Cohesion is absent for low_similarity/token_budget because code checks those first. Missing cohesion does not mean cohesion>=0.4.','More-children subset consists of130sessions, but some could have no added boundary only if not truly more children; report actual boundary-supporting session count.','Operational cause of retaining a boundary is observed; semantic suitability still needs raw-text review.']}
(OUT/'locomo_boundary_causes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (OUT/'locomo_boundary_causes.jsonl').open('w') as f:
 for r in boundaries:f.write(json.dumps(r,ensure_ascii=False)+'\n')
(OUT/'locomo_boundary_sessions.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
