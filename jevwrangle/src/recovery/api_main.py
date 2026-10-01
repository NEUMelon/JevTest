"""Resume-safe, auditable LLM H/D collection with a fixed validated prompt.

Run after the design pilot passed. A transport failure remains missing rather than
silently being replayed. No predictions or labels are used to adjust prompts.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import httpx
import yaml
from src.clients.llm import LLMClient
from src.clients.ledger import Ledger
from src.recovery.api_pilot import messages
from src.recovery.core import probability,llm_probability

BASE=Path(__file__).resolve().parents[2]
if (BASE/'.runtime_dependencies').exists():sys.path.insert(0,str(BASE/'.runtime_dependencies'))


def run(out,workers,endpoint,http2=False):
    pilot=json.loads((BASE/'runs/api_protocol_pilot_20260930/summary.json').read_text(encoding='utf8'))
    if pilot['calls']!=64 or pilot['conflicts']!=0:raise ValueError('Design pilot gate failed')
    out.mkdir(parents=True,exist_ok=True);(out/'features').mkdir(exist_ok=True)
    config=out/'budget.yaml'
    if not config.exists():config.write_text('total_usd_hard_limit: 8.0\n',encoding='utf8')
    ledger=Ledger(str(out/'ledger.csv'),str(config))
    os.environ['JEV_RUN_ID']=out.name
    clients={m:LLMClient(model=m,cache_path=str(BASE/'cache/calls_v2.sqlite'),ledger=ledger,endpoint=endpoint)
             for m in ['gpt-6-luna','deepseek-v4-flash']}
    for client in clients.values():
        client.http.close()
        client.http=httpx.Client(timeout=60,trust_env=False,http2=http2,
            limits=httpx.Limits(max_connections=workers,max_keepalive_connections=workers,keepalive_expiry=30),
            headers={'Authorization':'Bearer '+client.api_key,'Content-Type':'application/json'})
    source=Path(__file__);sourcehash=hashlib.sha256(source.read_bytes()).hexdigest()
    prompt_hash=hashlib.sha256((BASE/'src/recovery/api_pilot.py').read_bytes()).hexdigest()
    snapshots={}
    for rel in ['src/recovery/api_main.py','src/recovery/api_pilot.py','src/clients/evidence.py','src/clients/llm.py','src/clients/ledger.py']:
        sourcefile=BASE/rel;digest=hashlib.sha256(sourcefile.read_bytes()).hexdigest()
        dest=out/'source_snapshot'/digest/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.write_bytes(sourcefile.read_bytes())
        snapshots[rel]=digest
    manifest_path=out/'manifest.json'
    history=[]
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf8'))
        if old['prompt_source_sha256']!=prompt_hash:raise ValueError('Prompt changed; use a separate run')
        history=old.get('previous_attempts',[])+[{k:v for k,v in old.items() if k!='previous_attempts'}]
    manifest=dict(status='RUNNING_NOT_FROZEN',workers=workers,endpoint=endpoint,concurrency_deviation='Collection concurrency differs from plan 6; use separate controlled runs for E6 latency' if workers!=6 else None,
                  source_sha256=sourcehash,prompt_source_sha256=prompt_hash,api_budget_new_usd=8,
                  expected_calls=66924,http2=http2,source_snapshot_sha256=snapshots,
                  pricing_sources=['https://aihubmix.com/models?lang=en','https://api.aihubmix.com/model/deepseek-v4-flash'])
    manifest['previous_attempts']=history
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    parsed=out/'parsed.jsonl';rows={};lock=threading.Lock()
    def identity(r):return (r['dataset'],r['split'],r['pair_id'],r['model'],r['kind'])
    if parsed.exists():
        for s in parsed.read_text(encoding='utf8').splitlines():
            r=json.loads(s);rows[identity(r)]=r
    jobs=[]
    input_hashes={}
    for ds in ['wa','ag','da','ab']:
        h=yaml.safe_load((BASE/f'questions/{ds}/holistic.yaml').read_text(encoding='utf8'))
        d=yaml.safe_load((BASE/f'questions/{ds}/decomposed.yaml').read_text(encoding='utf8'))
        for rel in [f'questions/{ds}/holistic.yaml',f'questions/{ds}/decomposed.yaml']:
            input_hashes[rel]=hashlib.sha256((BASE/rel).read_bytes()).hexdigest()
        for split,path in [('pool',BASE/f'data/pools/{ds}_pool2000.jsonl'),('test',BASE/f'data/canonical/{ds}/test.jsonl')]:
            pairs=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
            input_hashes[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            for p in pairs:
                for model in clients:
                    for kind,q in [('H',{'H1_noul':h['H1_noul']}),('D',d)]:
                        job=dict(dataset=ds,split=split,pair_id=p['pair_id'],model=model,kind=kind,label=p['label'])
                        # Unknown-billing failures are not automatically replayed.
                        if identity(job) in rows:continue
                        jobs.append((job,p,q))
    if history and history[-1].get('input_sha256') and history[-1]['input_sha256']!=input_hashes:
        raise ValueError('Dataset or question inputs changed; use a separate run')
    manifest['input_sha256']=input_hashes
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    def one(args):
        job,p,q=args;client=clients[job['model']]
        for attempt in range(3):
            response,meta=client.ask(messages(p,q,job['kind']=='D'),max_tokens=700 if 'luna' in job['model'] else 400,
                reasoning_effort='none' if 'luna' in job['model'] else None,bypass_cache=attempt>0,
                context={k:v for k,v in job.items() if k!='label'})
            try:
                obj=json.loads(response['choices'][0]['message']['content'])
                if job['kind']=='D':values={qid:probability(obj[qid]) for qid in q};answer=None;conflict=False
                else:
                    values={'p_yes':llm_probability(obj)};answer=str(obj.get('answer','')).lower().strip()
                    if answer not in ['yes','no']:raise ValueError('Expected yes/no decision')
                    conflict=(answer=='yes')!=(values['p_yes']>=.5)
                return dict(**job,status='ok',values=values,explicit_answer=answer,answer_probability_conflict=conflict,
                            meta=meta,response_model=response.get('model'),parse_attempts=attempt+1)
            except (ValueError,TypeError,KeyError):
                if attempt==2:return dict(**job,status='parse_failed',values={},meta=meta)
    def save():
        df=pd.DataFrame([{**{k:v for k,v in r.items() if k not in ['values','meta']},**r['values'],
                          'cache_key':r.get('meta',{}).get('key'),'request_id':r.get('meta',{}).get('request_id')}
                         for r in rows.values()])
        if not df.empty:
            for keys,g in df.groupby(['dataset','split','model','kind']):
                g.sort_values('pair_id').to_csv(out/'features'/('_'.join(keys)+'.csv'),index=False)
        summary=dict(status='RUNNING_NOT_FROZEN',completed=len(rows),expected=66924,
                     failed=sum(r['status']!='ok' for r in rows.values()),
                     conflicts=sum(bool(r.get('answer_probability_conflict')) for r in rows.values()),cost=ledger.summary())
        (out/'status.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(f'API_MAIN queued={len(jobs)} already_recorded={len(rows)} workers={workers}',flush=True)
    # Keep only workers requests in flight, so interruption does not leave a huge
    # executor queue draining paid calls in the background.
    iterator=iter(jobs);pending={};fatal=None;completed=0;consecutive_failures=0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in range(workers):
            job=next(iterator,None)
            if job:pending[ex.submit(one,job)]=job[0]
        while pending:
            done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for f in done:
                job=pending.pop(f)
                try:
                    r=f.result();consecutive_failures=0
                except Exception as e:
                    r=dict(**job,status='request_failed',values={},error_type=type(e).__name__)
                    consecutive_failures+=1
                    if type(e).__name__=='BudgetExceededError' or consecutive_failures>=3:
                        fatal=fatal or type(e).__name__
                rows[identity(r)]=r
                with parsed.open('a',encoding='utf8') as writer:writer.write(json.dumps(r,ensure_ascii=False)+'\n')
                completed+=1
                if completed%200==0:save();print(f'API_MAIN completed={len(rows)}/66924 spent_estimated=${ledger.summary()["total_paid"]:.4f}',flush=True)
                if (out/'STOP').exists():fatal=fatal or 'MaintenanceStop'
                if fatal is None:
                    nextjob=next(iterator,None)
                    if nextjob:pending[ex.submit(one,nextjob)]=nextjob[0]
    save()
    failed=sum(r['status']!='ok' for r in rows.values())
    manifest.update(status='STOPPED_REQUIRES_ATTENTION' if fatal else 'COLLECTED_WITH_MISSING_NOT_FROZEN' if failed else 'COLLECTED_NOT_FROZEN',completed=len(rows),fatal_error_type=fatal)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    status=json.loads((out/'status.json').read_text(encoding='utf8'))
    status.update(status=manifest['status'],fatal_error_type=fatal)
    (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    for client in clients.values():client.http.close()
    print(json.dumps(manifest,indent=2),flush=True)
    if fatal:raise RuntimeError('Collection stopped after request failure; raw events retained. Inspect before resume.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--endpoint',choices=['https://aihubmix.com/v1/chat/completions','https://api.inferera.com/v1/chat/completions'],default='https://aihubmix.com/v1/chat/completions')
    ap.add_argument('--http2',action='store_true',help='Opt in; Windows shared HTTP/2 connections previously failed with WinError 10035')
    a=ap.parse_args();run(a.output.resolve(),a.workers,a.endpoint,a.http2)
