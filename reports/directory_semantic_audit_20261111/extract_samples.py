import sqlite3,json,random,pathlib,hashlib,datetime
out=pathlib.Path('reports/directory_semantic_audit_20261111');out.mkdir(exist_ok=True)
pnew='runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1/build_journal.sqlite3';pold='runs/parent_child_directories/qwen27b_child300_v1/build_journal.sqlite3'
c=sqlite3.connect('file:'+pnew+'?mode=ro',uri=True);c.execute('attach database ? as old',('file:'+pold+'?mode=ro',))
q="""select s.bank_id,s.parent_id,b.dataset,json_extract(s.directory,'$.source.body_token_count'),json_array_length(s.snapshot,'$.blocks'),json_array_length(s.directory,'$.children'),json_array_length(o.directory,'$.children'),json_extract(b.payload,'$.query_memberships') from sessions s join banks b using(bank_id) join old.sessions o on s.bank_id=o.bank_id and s.parent_id=o.parent_id and s.source_hash=o.source_hash where s.status='complete' and o.status='complete' and b.dataset in ('locomo','split_longmemeval_s')"""
rows=c.execute(q).fetchall();rng=random.Random(202611112131);chosen=[]
loc=[r for r in rows if r[2]=='locomo'];bybank={}
for r in loc:bybank.setdefault(r[0],[]).append(r)
for bank in sorted(bybank):chosen.append(rng.choice(sorted(bybank[bank])))
chosen+=rng.sample([r for r in loc if r not in chosen],2)
lme=sorted([r for r in rows if r[2]=='split_longmemeval_s'],key=lambda r:(r[3],r[0],r[1]));used=set();quartiles=[]
for i in range(4):
 pool=lme[len(lme)*i//4:len(lme)*(i+1)//4];rng.shuffle(pool);sel=[]
 for r in pool:
  if r[0] not in used:sel.append(r);used.add(r[0])
  if len(sel)==4:break
 chosen+=sel;quartiles.append({'quartile':i+1,'tokens_min':min(x[3] for x in pool),'tokens_max':max(x[3] for x in pool),'selected':[(x[0],x[1]) for x in sel]})
items=[]
def trim(d):
 return {k:d[k] for k in ('title','content','entities','children','build','source')}
for idx,r in enumerate(chosen):
 n=c.execute('select snapshot,directory,result_stats from sessions where bank_id=? and parent_id=?',r[:2]).fetchone();o=c.execute('select snapshot,directory from old.sessions where bank_id=? and parent_id=?',r[:2]).fetchone()
 snap=json.loads(n[0]);oldsnap=json.loads(o[0]);assert snap==oldsnap
 recs={x['record_id']:x for x in snap['records']};blocks=[{'index':b['index'],'token_count':b['token_count'],'messages':[recs[rid] for rid in b['record_ids']]} for b in snap['blocks']]
 item={'audit_id':('L' if r[2]=='locomo' else 'S')+str((idx+1) if r[2]=='locomo' else (idx-11)).zfill(2),'bank_id':r[0],'parent_id':r[1],'dataset':r[2],'body_tokens':r[3],'block_count':r[4],'memberships':json.loads(r[7]),'time':snap['time'],'participants':snap['participants'],'raw_blocks':blocks,'old':trim(json.loads(o[1])),'new':trim(json.loads(n[1])),'new_stats':json.loads(n[2]) if n[2] else {}}
 item['raw_content_hash']=hashlib.sha256(json.dumps([[x['speaker'],x['text']] for x in snap['records']],ensure_ascii=False).encode()).hexdigest();items.append(item)
for group,seq in [('locomo',items[:12]),('longmemeval_s_a',items[12::2]),('longmemeval_s_b',items[13::2])]:
 (out/(group+'_samples.json')).write_text(json.dumps(seq,ensure_ascii=False,indent=2))
 # Readable full evidence, no truncation.
 with (out/(group+'_samples.txt')).open('w') as f:
  for a in seq:
   f.write('\n===== '+a['audit_id']+' '+a['bank_id']+'/'+a['parent_id']+f" ({a['body_tokens']} tokens, {a['block_count']} blocks) =====\n")
   for version in ('old','new'):
    d=a[version];f.write('\n'+version.upper()+' PARENT '+d['title']+'\nSUMMARY '+d['content']+'\nENTITIES '+json.dumps(d['entities'],ensure_ascii=False)+'\n')
    for child in d['children']:
     s=child['source'];f.write(f"{version.upper()} {child['id']} BLOCK {s['start_block']}-{s['end_block']}: {child['title']}\nSUMMARY {child['content']}\nENTITIES {json.dumps(child['entities'],ensure_ascii=False)}\n")
   f.write('\nRAW BLOCKS\n')
   for b in a['raw_blocks']:
    f.write(f"[BLOCK {b['index']}] "+'\n'.join(x['speaker']+': '+x['text'] for x in b['messages'])+'\n')
summary={'time':datetime.datetime.now().isoformat(),'seed':202611112131,'new_database':pnew,'old_database':pold,'eligible_complete_pairs':{d:sum(r[2]==d for r in rows) for d in ('locomo','split_longmemeval_s')},'method':'LoCoMo one seeded random session per each of ten banks, plus two seeded random remaining sessions; LME-S four seeded sessions per body-token quartile with distinct banks. Complete on both sides, source_hash and exact snapshots equal. Qualitative sample not a population error-rate estimate.','longmemeval_token_quartiles':quartiles,'samples':[{k:a[k] for k in ('audit_id','bank_id','parent_id','dataset','body_tokens','block_count','raw_content_hash')} for a in items]}
(out/'sample_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));c.close()
print(json.dumps({k:v for k,v in summary.items() if k!='samples'},ensure_ascii=False,indent=2));print('samples',[(a['audit_id'],a['body_tokens'],len(a['old']['children']),len(a['new']['children'])) for a in items]);print('unique sampled raw',len({a['raw_content_hash'] for a in items}));print('chars',[(p.name,p.stat().st_size) for p in out.glob('*samples.txt')])
