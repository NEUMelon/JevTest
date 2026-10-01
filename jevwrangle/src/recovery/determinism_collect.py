"""Fresh, cache-bypassed repeat observations under the remaining night budget."""
import concurrent.futures as cf,hashlib,json,os,time
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,psutil,yaml
import httpx
from sklearn.model_selection import train_test_split
from src.clients.systemone import SystemOneClient
from src.clients.ledger import Ledger,BudgetExceededError
from .core import noul
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'runs/determinism_fresh_20260930'

def main():
    # Only run after the other API collectors have exited; never race ledgers.
    for process in psutil.process_iter(['pid','name','cmdline']):
        args=process.info['cmdline'] or []
        if process.pid!=os.getpid() and any(x in args for x in ['src.recovery.overnight_api','src.recovery.overnight_finish']):
            raise RuntimeError('Another paid API phase is still running')
    final=json.loads((BASE/'runs/overnight_finalization_20260930/status.json').read_text(encoding='utf8'))
    remote=float(pd.read_csv(BASE/'runs/remote_cost_probe_inferera_20260930/ledger.csv').paid_usd.sum())
    spent=final['api_main_spent']+final['repair_spent']+final['route_pilot_spent']+remote
    cap=min(.11,3.05-spent)
    if cap<=0:raise RuntimeError('No budget remains for fresh repeat evidence')
    OUT.mkdir(exist_ok=False);(OUT/'budget.yaml').write_text(f'total_usd_hard_limit: {cap}\n',encoding='utf8')
    ledger=Ledger(str(OUT/'ledger.csv'),str(OUT/'budget.yaml'));os.environ['JEV_RUN_ID']=OUT.name
    client=SystemOneClient(ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite'))
    jobs=[];selected={};hashes={}
    for ds in ['wa','ag','da','ab']:
        path=BASE/f'data/canonical/{ds}/test.jsonl';hashes[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        pairs=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
        chosen,_=train_test_split(pairs,train_size=200,stratify=[p['label'] for p in pairs],random_state=20260930)
        qpath=BASE/f'questions/{ds}/holistic.yaml';hashes[qpath.relative_to(BASE).as_posix()]=hashlib.sha256(qpath.read_bytes()).hexdigest()
        definition=yaml.safe_load(qpath.read_text(encoding='utf8'))['H1_noul'];selected[ds]=[r['pair_id'] for r in chosen]
        for repeat in range(5):
            for pair in chosen:jobs.append((ds,repeat,pair,{'H1_noul':definition}))
    meta=dict(status='FROZEN_BEFORE_REPEATS',selection=selected,seed=20260930,n_per_dataset=200,n_repeats=5,
        question_scope='H1_noul only; identical transmitted body for all five repeats; not a claim about batched multi-question independence',
        cache_bypass=True,concurrency=6,night_spent_before=spent,new_cap=cap,known_global_cap=3.05,
        amendment='200 instead of preregistered 500 per dataset because of hard balance; stratified sample frozen before any new repeat outcome',
        input_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2));(OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    def one(job):
        ds,repeat,pair,q=job;identity=dict(dataset=ds,repeat=repeat,pair_id=pair['pair_id'],label=pair['label'])
        response,info=client.ask({k:pair[k] for k in ['record_a','record_b']},q,bypass_cache=True,
            context={'experiment':'E5_fresh_repeats',**identity})
        if info['cache_hit']:raise ValueError('Repeat came from cache')
        return dict(**identity,status='ok',p_yes=noul(response['answers']['H1_noul']),response_model=response.get('model'),meta=info)
    iterator=iter(jobs);pending={};observed=[];stopped=False;started=time.perf_counter()
    consecutive_failures=0;stop_reason=None
    with cf.ThreadPoolExecutor(max_workers=6) as executor,(OUT/'parsed.jsonl').open('a',encoding='utf8') as target:
        def refill():
            while not stopped and len(pending)<6:
                try:job=next(iterator)
                except StopIteration:return
                pending[executor.submit(one,job)]=job
        refill()
        while pending:
            ready,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
            for future in ready:
                job=pending.pop(future)
                try:r=future.result()
                except BudgetExceededError:
                    stopped=True;stop_reason='LOCAL_USAGE_ESTIMATE_CAP';continue
                except Exception as error:
                    ds,repeat,pair,q=job;r=dict(dataset=ds,repeat=repeat,pair_id=pair['pair_id'],label=pair['label'],status='failed',error_type=type(error).__name__)
                    consecutive_failures+=1
                    # An estimated ledger cannot override the gateway's actual
                    # wallet refusal. Drain at most the already submitted tail.
                    status_code=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
                    balance_blocked=status_code in (402,403) and 'balance' in error.response.text.lower()
                    if balance_blocked or consecutive_failures>=6:
                        stopped=True
                        stop_reason='GATEWAY_WALLET_REFUSAL' if balance_blocked else 'SIX_CONSECUTIVE_FAILURES'
                else:
                    consecutive_failures=0
                observed.append(r);target.write(json.dumps(r)+'\n');target.flush()
                if len(observed)%100==0:print('FRESH_REPEATS',len(observed),ledger.summary()['total_paid'],flush=True)
            refill()
    results=[];cases=[];frame=pd.DataFrame(observed)
    for ds in ['wa','ag','da','ab']:
        good=frame[(frame.dataset==ds)&(frame.status=='ok')];complete=0
        for pair_id,g in good.groupby('pair_id'):
            if len(g)!=5 or set(g['repeat'])!=set(range(5)):continue
            if g.meta.map(lambda m:m['request_id']).nunique()!=5 or g.meta.map(lambda m:m['key']).nunique()!=1:raise ValueError('Repeat request identities do not prove fresh identical requests')
            p=g.p_yes.values;complete+=1
            cases.append(dict(dataset=ds,pair_id=pair_id,repeat_count=5,p_std=float(p.std()),p_range=float(p.max()-p.min()),
                binary_flip=int(len(set(p>=.5))>1),response_model_variants=g.response_model.nunique()))
        selected_cases=[r for r in cases if r['dataset']==ds]
        results.append(dict(dataset=ds,n_expected=200,n_complete=complete,status='COMPLETE_FRESH_REPEATS' if complete==200 else 'BLOCKED_INCOMPLETE',
            mean_std=float(np.mean([r['p_std'] for r in selected_cases])) if complete else None,
            max_range=max([r['p_range'] for r in selected_cases]) if complete else None,
            fraction_any_probability_change=float(np.mean([r['p_range']>0 for r in selected_cases])) if complete else None,
            fraction_binary_flip=float(np.mean([r['binary_flip'] for r in selected_cases])) if complete else None))
    pd.DataFrame(cases).to_csv(OUT/'case_variability.csv',index=False);pd.DataFrame(results).to_csv(OUT/'repeat_summary.csv',index=False)
    meta.update(status='COMPLETE_FRESH_REPEATS_NOT_FROZEN' if all(r['n_complete']==200 for r in results) else 'PARTIAL_FRESH_REPEATS',
        observed=len(observed),paid=ledger.summary()['total_paid'],seconds=time.perf_counter()-started,
        stop_reason=stop_reason,cost_basis='usage_times_configured_price_estimate_not_account_invoice',
        response_models=dict(Counter(r['response_model'] for r in observed if r['status']=='ok')),
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2));(OUT/'status.json').write_text(json.dumps(meta,indent=2))
    print('FRESH_REPEATS_COMPLETE',meta['status'],meta['paid'],flush=True)

if __name__=='__main__':main()
