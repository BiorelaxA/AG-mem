"""Read-only structural comparison; no generation or mutation of source directories."""
import collections, hashlib, json, sqlite3, statistics, pathlib, datetime
ROOT=pathlib.Path('/share/home/wtma/AG-mem')
OUT=ROOT/'reports/directory_semantic_audit_20261111'
newdb=ROOT/'runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1/build_journal.sqlite3'
olddb=ROOT/'runs/parent_child_directories/qwen27b_child300_v1/build_journal.sqlite3'
cn=sqlite3.connect('file:'+str(newdb)+'?mode=ro', uri=True)
co=sqlite3.connect('file:'+str(olddb)+'?mode=ro', uri=True)

def raw_hash(s, body_only=False):
 def record(r):return {k:v for k,v in r.items() if k not in ('record_id','seq')}
 p={'time':s['time'],'participants':s['participants'],'source_kind':s['source_kind'],
    'records':[record(x) for x in s['records']], 'headers':[record(x) for x in s['header_records']]}
 if body_only:p={k:v for k,v in p.items() if k in ('records','source_kind')}
 return hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def entry_stats(d,s):
 children=d['children']; nextblock=1;errs=[];sizes=[];tokens=[];boundaries=[]
 record_map={r['record_id']:r for r in s['records']}
 for x in children:
  src=x['source'];a,b=src['start_block'],src['end_block']; sizes.append(b-a+1);tokens.append(src['body_token_count'])
  if a!=nextblock or b<a or b>len(s['blocks']):errs.append('block_coverage')
  if src['block_count']!=b-a+1:errs.append('block_count')
  rids=[rid for bl in s['blocks'][a-1:b] for rid in bl['record_ids']]
  if not rids or src['start_record_id']!=rids[0] or src['end_record_id']!=rids[-1] or src['record_count']!=len(rids):errs.append('record_binding')
  nextblock=b+1;boundaries.append(b)
 if nextblock!=len(s['blocks'])+1:errs.append('trailing_coverage')
 if len(children) and boundaries:boundaries=boundaries[:-1]
 return {'children':len(children),'sizes':sizes,'tokens':tokens,'boundaries':boundaries,'errors':sorted(set(errs))}

def aggregate(rows):
 result={'sessions':len(rows),'banks':len({r['bank_id'] for r in rows})}
 for version in ('old','new'):
  sizes=[n for r in rows for n in r[version]['sizes']];tokens=[n for r in rows for n in r[version]['tokens']]
  cc=[r[version]['children'] for r in rows]
  result[version]={'children':len(sizes),'mean_children_per_session':statistics.mean(cc) if cc else None,
    'mean_blocks_per_child':statistics.mean(sizes) if sizes else None,'median_blocks_per_child':statistics.median(sizes) if sizes else None,
    'one_block_percent':100*sizes.count(1)/len(sizes) if sizes else None,'two_block_percent':100*sizes.count(2)/len(sizes) if sizes else None,
    'size_distribution':dict(sorted(collections.Counter(sizes).items())), 'max_blocks':max(sizes,default=0),'max_body_tokens':max(tokens,default=0),
    'over4096_children':sum(t>4096 for t in tokens), 'over16blocks_children':sum(t>16 for t in sizes),
    'median_body_tokens':statistics.median(tokens) if tokens else None,
    'coverage_error_sessions':sum(bool(r[version]['errors']) for r in rows)}
 changes=collections.Counter();removed=added=0
 for r in rows:
  o,n=r['old'],r['new'];ob,nb=set(o['boundaries']),set(n['boundaries'])
  changes['identical_boundaries' if ob==nb else ('fewer_children' if n['children']<o['children'] else 'more_children' if n['children']>o['children'] else 'same_count_different_boundaries')]+=1
  removed+=len(ob-nb);added+=len(nb-ob)
 result['boundary_changes']={**changes,'removed':removed,'added':added}
 return result

banks=cn.execute("select bank_id,dataset from banks where dataset in ('locomo','split_longmemeval_s') order by dataset,bank_id").fetchall()
rows=[];statuses=collections.defaultdict(collections.Counter); mismatch=[];stats=collections.defaultdict(collections.Counter)
for bi,(bank,ds) in enumerate(banks):
 nr=cn.execute('select parent_id,status,source_hash,directory,snapshot,result_stats from sessions where bank_id=? order by parent_id',(bank,)).fetchall()
 old={r[0]:r for r in co.execute('select parent_id,status,source_hash,directory from sessions where bank_id=?',(bank,)).fetchall()}
 for pid,status,sh,djs,sjs,stjs in nr:
  statuses[ds][status]+=1
  if status!='complete' or pid not in old or old[pid][1]!='complete':continue
  if sh!=old[pid][2]:mismatch.append([ds,bank,pid]);continue
  s=json.loads(sjs);nd=json.loads(djs);od=json.loads(old[pid][3]);st=json.loads(stjs or '{}')
  for k,v in st.items():
   if isinstance(v,(int,float)) and not isinstance(v,bool):stats[ds][k]+=v
  rows.append({'dataset':ds,'bank_id':bank,'parent_id':pid,'raw_hash':raw_hash(s),'body_hash':raw_hash(s,body_only=True),'blocks':len(s['blocks']),
    'body_tokens':sum(x['token_count'] for x in s['blocks']), 'old':entry_stats(od,s),'new':entry_stats(nd,s)})
 if bi%50==0:print('banks',bi+1,'paired sessions',len(rows),flush=True)
cn.close();co.close()
report={'timestamp':datetime.datetime.now().astimezone().isoformat(),'old_db':str(olddb),'new_db':str(newdb),
'criteria':'Both status complete, same (bank_id,parent_id,source_hash). Counts are built intersection, not all planned sessions.',
'unique_rule':'SHA256 of canonical raw records (without local IDs/seq), headers, source_kind, time and participants. For deduplicated metrics, lexicographically first (bank_id,parent_id) represents each unique raw session; identical text can have varied generated outputs.',
'body_only_unique_rule':'Records excluding local IDs/seq and source_kind only, ignore timestamps/participants/header metadata; same lexicographic representative rule.', 'source_hash_mismatches':mismatch,'datasets':{}}
for ds in ('locomo','split_longmemeval_s'):
 rr=[r for r in rows if r['dataset']==ds]; unique={}; variants=collections.defaultdict(set)
 for r in rr:
  unique.setdefault(r['raw_hash'],r)
  variants[r['raw_hash']].add(tuple(r['new']['boundaries']))
 body_unique={}
 for r in rr:body_unique.setdefault(r['body_hash'],r)
 report['datasets'][ds]={'current_status':dict(statuses[ds]), 'paired_instances':aggregate(rr), 'deduplicated_raw_sessions':aggregate(list(unique.values())), 'deduplicated_body_only':aggregate(list(body_unique.values())),
  'raw_sessions_with_multiple_new_boundary_patterns':sum(len(v)>1 for v in variants.values()),'segmentation_stats_sums':dict(stats[ds])}
(OUT/'corpus_stats.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (OUT/'paired_session_stats.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
for ds,s in report['datasets'].items():
 print(ds,'status',s['current_status'])
 for unit in ('paired_instances','deduplicated_raw_sessions','deduplicated_body_only'):
  a=s[unit];print(unit,'sessions',a['sessions'],'banks',a['banks'])
  for v in ('old','new'):print(v,{k:x for k,x in a[v].items() if k!='size_distribution'})
  print(a['boundary_changes'])
 print('boundary variants',s['raw_sessions_with_multiple_new_boundary_patterns']);print('seg stats',s['segmentation_stats_sums'])
