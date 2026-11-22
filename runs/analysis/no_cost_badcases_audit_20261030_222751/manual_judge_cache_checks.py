import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import pyarrow.parquet as pq
from agmem.judges import prompt_for
from agmem.core import digest
out=Path(__file__).resolve().parent
run=json.load(open(Path(json.load(open(out/'snapshot.json'))['run'])/'run.json'))
prov=run['scoring_protocol']['judge'];scope='judge/'+digest(prov)
prep=Path(run['config']['data']['prepared_root']);rawroot=Path(run['config']['data']['input_root'])
cache={row['key']:row['response'] for row in map(json.loads,open(prep/'_runtime/completions.jsonl'))}
examples={}
for source in ['locomo','split_longmemeval_s']:
 raw=pq.read_table(rawroot/source/'train.parquet',columns=['question_id']+(['question_type'] if 'longmemeval' in source else [])).to_pylist()
 for i,row in enumerate(map(json.loads,open(prep/source/'train_examples.jsonl'))):
  row['judge_metadata']=raw[int(row['sample_id'].rsplit('/',1)[-1])]
  examples[row['sample_id']]=row
cs=json.load(open(out/'badcase_manual_extracted.json'));res=[]
for c in cs:
 if c['candidate'] not in [162,168,169,174,255,272,302,315]:continue
 ex=SimpleNamespace(**examples[c['sample_id']]);pr=[]
 for pred in dict.fromkeys(r['answer'] for r in c['rows']):
  for gold in ex.gold_answers:
   prompt,parse=prompt_for(ex,gold,pred,prov['protocols'][ex.source])
   req={'model':prov['model']['model'],'messages':[{'role':'user','content':prompt}],'temperature':0.0,'max_tokens':prov['max_output_tokens'],'seed':prov['seed'],'chat_template_kwargs':{'enable_thinking':False}}
   key=digest({'identity':prov['model'],'role':'judge','request':req,'cache_scope':scope})
   raw=cache.get(key)
   pr.append({'prediction':pred,'gold':gold,'cache_key':key,'raw_judge_response':raw,'parsed_judge_score':parse(raw) if raw is not None else None})
 res.append({'candidate':c['candidate'],'sample_id':c['sample_id'],'cached_judgments':pr})
 print(c['candidate'],[(a['prediction'],a['raw_judge_response']) for a in pr])
json.dump(res,open(out/'manual_judge_cache_checks.json','w'),indent=2,ensure_ascii=False)
