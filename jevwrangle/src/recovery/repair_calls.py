"""Re-observe only the seven known missing/conflicting Jev requests."""
import json
import os
from pathlib import Path
from src.clients.systemone import SystemOneClient
from src.clients.ledger import Ledger
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/api_jev_repairs_20260930'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    requests=[json.loads(s) for s in (BASE/'runs/recovery_20260930/repair_requests.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(requests)==7
    (OUT/'budget.yaml').write_text('total_usd_hard_limit: 0.02\n',encoding='utf8')
    ledger=Ledger(str(OUT/'ledger.csv'),str(OUT/'budget.yaml'))
    os.environ['JEV_RUN_ID']='api_jev_repairs_20260930'
    client=SystemOneClient(model='jev-1.13',cache_path=str(BASE/'cache/calls_v2.sqlite'),ledger=ledger)
    observations=[]
    for item in requests:
        payload=item['payload']
        response,meta=client.ask(payload['state'],payload['questions'],context={k:v for k,v in item.items() if k!='payload'})
        observations.append(dict(**{k:v for k,v in item.items() if k!='payload'},response=response,meta=meta))
        (OUT/'observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2),encoding='utf8')
        if response.get('model')!='typesafe/jev-1.13-20260917':
            raise ValueError('Returned version differs from historical fixed version; stop and keep observations separate')
        print('REPAIRED',item['dataset'],item['split'],item['pair_id'],meta['key'],flush=True)
    client.http.close()
    (OUT/'summary.json').write_text(json.dumps(dict(n_requests=len(observations),cost=ledger.summary(),
        pricing_status='Estimated with frozen 2026-09-29 rate; not a billing receipt',
        status='NEW_OBSERVATIONS_NOT_ORIGINAL_EVIDENCE'),indent=2),encoding='utf8')


if __name__=='__main__':main()
