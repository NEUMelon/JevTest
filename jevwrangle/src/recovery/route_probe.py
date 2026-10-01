"""Validate the documented AIHubMix alternate endpoint on design data."""
import json
import os
import statistics
from pathlib import Path
import yaml
from src.clients.ledger import Ledger
from src.clients.llm import LLMClient
from .api_pilot import messages
from .core import probability,llm_probability
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/api_route_probe_20260930'


def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'budget.yaml').write_text('total_usd_hard_limit: 0.02\n')
    ledger=Ledger(str(OUT/'ledger.csv'),str(OUT/'budget.yaml'));os.environ['JEV_RUN_ID']=OUT.name
    rows=[]
    pairs=[json.loads(s) for s in (BASE/'data/design/wa.jsonl').read_text(encoding='utf8').splitlines()][:4]
    h=yaml.safe_load((BASE/'questions/wa/holistic.yaml').read_text(encoding='utf8'))
    d=yaml.safe_load((BASE/'questions/wa/decomposed.yaml').read_text(encoding='utf8'))
    for model in ['gpt-6-luna','deepseek-v4-flash']:
        client=LLMClient(model=model,ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite'),endpoint='https://api.inferera.com/v1/chat/completions')
        for p in pairs:
            for kind,q in [('H',{'H1_noul':h['H1_noul']}),('D',d)]:
                r,meta=client.ask(messages(p,q,kind=='D'),max_tokens=700 if 'luna' in model else 400,
                                 reasoning_effort='none' if 'luna' in model else None,
                                 context=dict(dataset='wa',split='design',pair_id=p['pair_id'],kind=kind))
                a=json.loads(r['choices'][0]['message']['content'])
                if kind=='D':decoded={k:probability(a[k]) for k in q};conflict=False
                else:decoded={'p_yes':llm_probability(a)};conflict=(a['answer']=='yes')!=(decoded['p_yes']>=.5)
                rows.append(dict(model=model,kind=kind,decoded=decoded,conflict=conflict,meta=meta,response_model=r.get('model')))
                (OUT/'parsed.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        client.http.close()
    summary=dict(status='PASSED' if not any(r['conflict'] for r in rows) else 'FAILED',calls=len(rows),
                 mean_latency_ms=statistics.mean(r['meta'].get('latency_ms',0) for r in rows),
                 cost=ledger.summary(),versions=sorted({r['response_model'] for r in rows}),
                 endpoint='https://api.inferera.com/v1/chat/completions')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
