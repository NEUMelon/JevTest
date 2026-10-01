"""Comparable H/D latency from AutoDL; key arrives only via encrypted stdin."""
import concurrent.futures as cf,json,os,sys,ssl
from pathlib import Path
import numpy as np,pandas as pd,yaml
import httpx
from src.clients.ledger import Ledger
from src.clients.llm import LLMClient
from src.clients.systemone import SystemOneClient
from .api_pilot import messages
BASE=Path(__file__).resolve().parents[2]
def main():
    secret=json.load(sys.stdin);os.environ['AIHUBMIX_API_KEY']=secret.pop('api_key');del secret
    out=BASE/'runs'/os.environ.get('JEV_COST_PROBE_OUT','remote_cost_probe_20260930');out.mkdir(exist_ok=False)
    (out/'budget.yaml').write_text('total_usd_hard_limit: 0.03\n');ledger=Ledger(str(out/'ledger.csv'),str(out/'budget.yaml'))
    os.environ['JEV_RUN_ID']=out.name
    clients={'jev-1.13':SystemOneClient(ledger=ledger,cache_path=str(BASE/'cache/cost_probe.sqlite')),
             **{m:LLMClient(model=m,ledger=ledger,cache_path=str(BASE/'cache/cost_probe.sqlite')) for m in ['gpt-6-luna','deepseek-v4-flash']}}
    # Official alternate gateway is reachable from this AutoDL instance.
    # Use the system trust store, which includes its configured certificate CA;
    # certificate verification remains enabled. No global network settings.
    import src.clients.systemone as systemone
    systemone.URL='https://api.inferera.com/v1/systemone'
    for model,client in clients.items():
        if model!='jev-1.13':client.endpoint='https://api.inferera.com/v1/chat/completions'
        client.http.close();client.http=httpx.Client(timeout=60,trust_env=False,verify=ssl.create_default_context(),
            headers={'Authorization':'Bearer '+client.api_key,'Content-Type':'application/json'})
    jobs=[]
    for ds in ['wa','ag','da','ab']:
        pairs=[json.loads(s) for s in (BASE/f'data/design/{ds}.jsonl').read_text().splitlines()][:12]
        h=yaml.safe_load((BASE/f'questions/{ds}/holistic.yaml').read_text());d=yaml.safe_load((BASE/f'questions/{ds}/decomposed.yaml').read_text())
        for p in pairs:
            for model in clients:
                for kind,q in [('H',{'H1_noul':h['H1_noul']}),('D',d)]:jobs.append((ds,p,model,kind,q))
    # Fixed shuffled order avoids measuring one model only during a later
    # gateway load interval; concurrency stays at the plan's six requests.
    import random;random.Random(0).shuffle(jobs)
    def one(job):
        ds,p,model,kind,q=job;context=dict(dataset=ds,pair_id=p['pair_id'],kind=kind,experiment='E6_design',split='design')
        if model=='jev-1.13':response,meta=clients[model].ask({k:p[k] for k in ['record_a','record_b']},q,bypass_cache=True,context=context)
        else:response,meta=clients[model].ask(messages(p,q,kind=='D'),max_tokens=700 if 'luna' in model else 400,
                reasoning_effort='none' if 'luna' in model else None,bypass_cache=True,context=context)
        return dict(**context,model=model,response_model=response.get('model'),latency_ms=meta['latency_ms'],
                    paid_usd=meta['paid_usd'],in_tokens=meta['in_tokens'],out_tokens=meta['out_tokens'])
    rows=[];failures=[]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        pending={ex.submit(one,j):j for j in jobs}
        for future in cf.as_completed(pending):
            try:rows.append(future.result())
            except Exception as error:failures.append(dict(model=pending[future][2],error_type=type(error).__name__,error=str(error)[:200]))
    pd.DataFrame(rows).to_csv(out/'per_request.csv',index=False)
    summaries=[]
    for (model,kind),group in pd.DataFrame(rows).groupby(['model','kind']):
        summaries.append(dict(model=model,task=kind,n=len(group),cost_per_1000_decisions=1000*group.paid_usd.mean(),
            latency_p50_ms=float(group.latency_ms.quantile(.5)),latency_p95_ms=float(group.latency_ms.quantile(.95))))
    pd.DataFrame(summaries).to_csv(out/'matched_task_cost_latency.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_DESIGN_PROBE' if not failures else 'PARTIAL_DESIGN_PROBE',
        concurrency=6,location='AutoDL via official AIHubMix alternate api.inferera.com, while GPU training is active',n_expected=len(jobs),n_success=len(rows),
        cost=ledger.summary(),failures=failures,cache='bypassed deliberately for latency; every attempt billed',
        limitations='Small fixed design sample; H and D are different tasks, compare only within task; gateway load affects latency'),indent=2))
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
