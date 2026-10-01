"""One separately logged re-observation of identified failed LLM jobs.

Only runs after the full collection pass. It never modifies old failed records,
never replaces successful observations, and stops on the first new failure.
"""
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import yaml
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger
from .api_pilot import messages
from .core import probability, llm_probability

BASE=Path(__file__).resolve().parents[2]


def reobserve(collection, output):
    manifest=json.loads((collection/'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] not in ['COLLECTED_WITH_MISSING_NOT_FROZEN','COLLECTED_NOT_FROZEN']:
        raise ValueError('Full collection pass must complete before targeted re-observation')
    prompt_hash=hashlib.sha256((BASE/'src/recovery/api_pilot.py').read_bytes()).hexdigest()
    if prompt_hash!=manifest['prompt_source_sha256']:raise ValueError('Frozen prompt changed')
    records=[json.loads(s) for s in (collection/'parsed.jsonl').read_text(encoding='utf8').splitlines()]
    if len(records)!=manifest['expected_calls']:raise ValueError('Full pass has pending jobs')
    failed=[r for r in records if r['status']!='ok']
    output.mkdir(exist_ok=False)
    spent=float(pd.read_csv(collection/'ledger.csv').paid_usd.sum())
    allowance=max(0.,min(.2,8.-spent))
    config=output/'budget.yaml';config.write_text(f'total_usd_hard_limit: {allowance:.9f}\n',encoding='utf8')
    ledger=Ledger(str(output/'ledger.csv'),str(config));os.environ['JEV_RUN_ID']=output.name
    clients={m:LLMClient(model=m,ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite'),
                        endpoint=manifest['endpoint']) for m in ['gpt-6-luna','deepseek-v4-flash']}
    data={};questions={};inputs={}
    for r in failed:
        ds,split,kind=r['dataset'],r['split'],r['kind']
        if (ds,split) not in data:
            path=BASE/(f'data/pools/{ds}_pool2000.jsonl' if split=='pool' else f'data/canonical/{ds}/test.jsonl')
            data[ds,split]={p['pair_id']:p for p in [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]}
            inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        if (ds,kind) not in questions:
            path=BASE/f"questions/{ds}/{'holistic' if kind=='H' else 'decomposed'}.yaml"
            q=yaml.safe_load(path.read_text(encoding='utf8'));questions[ds,kind]={'H1_noul':q['H1_noul']} if kind=='H' else q
            inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for rel,digest in inputs.items():
        if manifest['input_sha256'][rel]!=digest:raise ValueError('Collection input changed')
    output_manifest=dict(status='TARGETED_REOBSERVATION_RUNNING',result_freeze=False,expected=len(failed),
        new_usd_limit=allowance,collection_sha256=hashlib.sha256((collection/'parsed.jsonl').read_bytes()).hexdigest(),
        prompt_source_sha256=prompt_hash,input_sha256=inputs,note='New observation of identified failure; historical billing remains unknown')
    (output/'manifest.json').write_text(json.dumps(output_manifest,indent=2),encoding='utf8')
    (output/'executed_source.py').write_bytes(Path(__file__).read_bytes());completed=0;error=None
    for original in failed:
        job={k:original[k] for k in ['dataset','split','pair_id','model','kind','label']}
        pair=data[job['dataset'],job['split']][job['pair_id']];q=questions[job['dataset'],job['kind']]
        if pair['label']!=job['label']:raise ValueError('Label changed')
        try:
            response,meta=clients[job['model']].ask(messages(pair,q,job['kind']=='D'),
                max_tokens=700 if 'luna' in job['model'] else 400,
                reasoning_effort='none' if 'luna' in job['model'] else None,
                bypass_cache=original['status']=='parse_failed',context=dict(purpose='targeted_failure_reobservation',**{k:v for k,v in job.items() if k!='label'}))
            obj=json.loads(response['choices'][0]['message']['content'])
            if job['kind']=='D':values={qid:probability(obj[qid]) for qid in q};answer=None;conflict=False
            else:
                values={'p_yes':llm_probability(obj)};answer=str(obj.get('answer','')).lower().strip()
                if answer not in ['yes','no']:raise ValueError('Expected yes/no')
                conflict=(answer=='yes')!=(values['p_yes']>=.5)
            r=dict(**job,status='ok',values=values,meta=meta,explicit_answer=answer,answer_probability_conflict=conflict,
                response_model=response.get('model'),provenance='new_targeted_reobservation',previous_status=original['status'])
        except Exception as e:
            r=dict(**job,status='repair_failed',values={},error_type=type(e).__name__,previous_status=original['status'])
            error=type(e).__name__
        with (output/'parsed.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        completed+=1
        if error:break
    output_manifest.update(status='REOBSERVED_NOT_FROZEN' if not error else 'STOPPED_REQUIRES_ATTENTION',completed=completed,error_type=error,cost=ledger.summary())
    output_manifest['output_sha256']={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (output/'manifest.json').write_text(json.dumps(output_manifest,indent=2),encoding='utf8')
    for c in clients.values():c.http.close()
    if not failed:(output/'parsed.jsonl').write_text('',encoding='utf8')
    print(json.dumps(output_manifest,indent=2))
    if error:raise RuntimeError('Targeted re-observation failed; do not replay automatically')

