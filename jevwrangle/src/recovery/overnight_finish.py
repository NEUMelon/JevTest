"""One-shot post-collection repair and evaluation; no spending while main runs."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import pandas as pd,yaml,psutil
from src.clients.ledger import Ledger,BudgetExceededError
from src.clients.llm import LLMClient
from src.clients.systemone import SystemOneClient
from .api_pilot import messages
from .core import llm_probability,probability,noul,choice
from .llm_aggregate import aggregate
from .stability_recompute import main as stability
from src.experiments.e5_stability import make_perturbed_requests
BASE=Path(__file__).resolve().parents[2]

def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()] if p.exists() else []
def spent(p):return float(pd.read_csv(p).paid_usd.sum()) if p.exists() else 0.

def main(pid):
    while psutil.pid_exists(pid):time.sleep(15)
    original=BASE/'runs/overnight_api_20260930';manifest=json.loads((original/'manifest.json').read_text(encoding='utf8'))
    out=BASE/'runs/overnight_finalization_20260930';out.mkdir(exist_ok=False)
    budget=max(0.,min(.12,3.05-spent(original/'ledger.csv')-spent(BASE/'runs/overnight_route_pilot_20260930/ledger.csv')))
    (out/'budget.yaml').write_text(f'total_usd_hard_limit: {budget}\n')
    ledger=Ledger(str(out/'ledger.csv'),str(out/'budget.yaml'));os.environ['JEV_RUN_ID']=out.name
    rows=read(original/'parsed.jsonl');snap=out/'evaluation_collection';snap.mkdir()
    repaired=[];jev=SystemOneClient(ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite'))
    llms={m:LLMClient(model=m,ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite')) for m in ['gpt-6-luna','deepseek-v4-flash']}
    pairs={};qh={};qd={};par={}
    for ds in ['wa','ag','da','ab']:
        pairs[ds]={p['pair_id']:p for p in read(BASE/f'data/pools/{ds}_pool2000.jsonl')+read(BASE/f'data/canonical/{ds}/test.jsonl')}
        qh[ds]=yaml.safe_load((BASE/f'questions/{ds}/holistic.yaml').read_text(encoding='utf8'));qd[ds]=yaml.safe_load((BASE/f'questions/{ds}/decomposed.yaml').read_text(encoding='utf8'))
        par[ds]=yaml.safe_load((BASE/f'questions/{ds}/paraphrase.yaml').read_text(encoding='utf8'))
    failures=[]
    for row in rows:
        if row['status']=='ok' or row['model']=='decision-model-preview':continue
        ds=row['dataset'];pair=pairs[ds][row['pair_id']]
        try:
            if row['model'] in llms:
                questions={'H1_noul':qh[ds]['H1_noul']} if row['kind']=='H' else qd[ds]
                response,meta=llms[row['model']].ask(messages(pair,questions,row['kind']=='D'),
                    max_tokens=700 if row['model']=='gpt-6-luna' else 400,
                    reasoning_effort='none' if row['model']=='gpt-6-luna' else None,
                    # A complete HTTP body can still be malformed JSON and
                    # already cached. Re-observe that failed parse once rather
                    # than reading the same invalid body forever.
                    bypass_cache=row.get('error_type') in ['JSONDecodeError','KeyError','ValueError','TypeError'],
                    context={'repair_of_failed_job':row})
                obj=json.loads(response['choices'][0]['message']['content'])
                values={'p_yes':llm_probability(obj)} if row['kind']=='H' else {q:probability(obj[q]) for q in questions}
            else:
                state,q=make_perturbed_requests(pair,qh[ds],qd[ds],par[ds],ds)[row['kind']]
                if row['kind']=='P2':q={'H1_noul':q['H1_noul'],**{k:q[k] for k in qd[ds]}}
                if row['kind']=='P5':
                    reverse=copy.deepcopy(qh[ds]['H1_choice']);reverse['criteria']=dict(reversed(list(reverse['criteria'].items())))
                    q={'H1_choice':reverse}
                response,meta=jev.ask(state,q,context={'repair_of_failed_job':row})
                values={qid:choice(response['answers'][qid]) if definition['type']=='choice' else noul(response['answers'][qid]) for qid,definition in q.items()}
            attempt=dict(**{k:v for k,v in row.items() if k not in ['status','values','meta','response_model']},status='ok',values=values,meta=meta,response_model=response.get('model'),repair_observation=True)
            repaired.append(attempt);row.update(status='ok',values=values,meta=meta,response_model=response.get('model'),repair_observation=True)
        except BudgetExceededError:break
        except Exception as error:failures.append(dict(identity=row,error=str(error)[:300]))
    (out/'repair_observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in repaired))
    repairs_dir=BASE/'runs/stability_api_repairs_20260930';repairs_dir.mkdir(exist_ok=True)
    observations=repairs_dir/'parsed.jsonl'
    with observations.open('a') as target:
        for r in repaired:
            if r['experiment']=='E5':target.write(json.dumps(r)+'\n')
    # Original journal remains immutable; evaluation uses a reconciled snapshot.
    (snap/'parsed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['experiment']=='EM' and r['model'] in llms))
    (snap/'manifest.json').write_text(json.dumps(dict(**manifest,reconciliation_sha256=hashlib.sha256((original/'parsed.jsonl').read_bytes()).hexdigest(),
        repair_source=str(out/'repair_observations.jsonl')),indent=2))
    aggregate(snap,out/'llm_evaluation',models=[('gpt-6-luna','L1'),('deepseek-v4-flash','L2')],protocol=manifest['protocol'])
    initial=out/'stability_before_repairs';stability(initial)
    todo=read_missing=json.loads((initial/'missing.json').read_text(encoding='utf8'))
    existing={(r['dataset'],r['pair_id'],r['kind']) for r in read(observations)}
    with observations.open('a') as target:
        for job in todo:
            if (job['dataset'],job['pair_id'],job['kind']) in existing or job['kind'] in ['P2','P5']:continue
            ds=job['dataset'];pair=pairs[ds][job['pair_id']]
            state,q=make_perturbed_requests(pair,qh[ds],qd[ds],par[ds],ds)[job['kind']]
            # Historical full question set identifies cached responses. Only
            # Noul outputs are used here, but send the same old context.
            try:
                response,meta=jev.ask(state,q,context={'experiment':'E5_missing_evidence_repair',**job})
                values={qid:noul(response['answers'][qid]) for qid in ['H1_noul',*qd[ds]]}
                target.write(json.dumps(dict(**job,status='ok',values=values,model='jev-1.13',meta=meta))+'\n');target.flush()
            except BudgetExceededError:break
            except Exception as error:failures.append(dict(identity=job,error=str(error)[:200]))
    stability(out/'stability_evaluation')
    from .adult_complete import complete as adult_zero_shot
    adult_zero_shot(jev,out/'adult_zero_shot')
    report=dict(status='FINALIZED_COMPLETE_GROUPS_ONLY_NOT_FROZEN',api_main_spent=spent(original/'ledger.csv'),repair_spent=ledger.summary()['total_paid'],
        route_pilot_spent=spent(BASE/'runs/overnight_route_pilot_20260930/ledger.csv'),repair_cap=budget,failures=failures,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);a=ap.parse_args();main(a.wait_pid)
