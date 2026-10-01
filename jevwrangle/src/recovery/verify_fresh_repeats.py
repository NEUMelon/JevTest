"""Independent offline verification of the partial fresh-request experiment."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[2]

def main():
    source=BASE/'runs/determinism_fresh_20260930'
    out=BASE/'runs/determinism_verified_20260930'
    out.mkdir(exist_ok=False)
    meta=json.loads((source/'manifest.json').read_text(encoding='utf8'))
    for relative,expected in meta['input_sha256'].items():
        assert hashlib.sha256((BASE/relative).read_bytes()).hexdigest()==expected,relative
    for relative,expected in meta['outputs'].items():
        assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==expected,relative
    assert hashlib.sha256((source/'executed_source.py').read_bytes()).hexdigest()==meta['source_sha256']
    requests={};responses={};http=Counter();balance=[]
    for line in (source/'api_events.jsonl').open(encoding='utf8'):
        event=json.loads(line);rid=event['request_id']
        if event['event']=='request':
            assert rid not in requests
            body=json.dumps(event['payload'],ensure_ascii=False,separators=(',',':'))
            key=hashlib.sha256((event['endpoint']+'\n'+body).encode('utf8')).hexdigest()
            assert key==event['cache_key'] and event['bypass_cache'] is True
            assert event['requested_model']=='jev-1.13'
            requests[rid]=event
        elif event['event']=='response':
            assert rid not in responses,'No retry expected for this completed observation'
            responses[rid]=event;http[event['http_status']]+=1
            if event['http_status']==403 and 'balance is insufficient' in json.dumps(event['response']).lower():balance.append(event)
        else:
            raise AssertionError('Unexpected event '+event['event'])
    rows=[json.loads(line) for line in (source/'parsed.jsonl').open(encoding='utf8')]
    assert len(requests)==len(responses)==len(rows)==4000
    good=[r for r in rows if r['status']=='ok']
    assert len(good)==http[200]==948 and len(balance)==http[403]==3052
    label_lookup={}
    for ds in ['wa','ag','da','ab']:
        label_lookup[ds]={r['pair_id']:r['label'] for r in [json.loads(line) for line in (BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()]}
    groups={}
    for row in good:
        rid=row['meta']['request_id'];request=requests[rid];response=responses[rid]
        assert row['meta']['cache_hit'] is False
        assert row['meta']['key']==request['cache_key']==response['cache_key']
        assert response['http_status']==200
        assert row['response_model']==response['response_model']=='typesafe/jev-1.13-20260917'
        assert row['pair_id'] in meta['selection'][row['dataset']]
        assert row['label']==label_lookup[row['dataset']][row['pair_id']]
        assert request['context']['repeat']==response['context']['repeat']==row['repeat']
        probability=float(response['response']['answers']['H1_noul']['noul'])
        assert probability==row['p_yes'] and 0<=probability<=1
        groups.setdefault((row['dataset'],row['pair_id']),[]).append(row)
    cases=[];examples=[]
    for (ds,pair_id),group in groups.items():
        if len(group)!=5:continue
        group=sorted(group,key=lambda row:row['repeat'])
        assert [r['repeat'] for r in group]==list(range(5))
        ids=[r['meta']['request_id'] for r in group]
        assert len(set(ids))==5
        bodies=[json.dumps(requests[rid]['payload'],ensure_ascii=False,separators=(',',':')) for rid in ids]
        assert len(set(bodies))==1
        service_ids=[responses[rid]['response']['id'] for rid in ids]
        assert len(set(service_ids))==5
        p=np.array([r['p_yes'] for r in group]);p_range=float(np.ptp(p))
        record=dict(dataset=ds,pair_id=pair_id,label=group[0]['label'],p_std=float(p.std()),p_range=p_range,binary_flip=int(len(set(p>=.5))>1),probabilities=p.tolist())
        cases.append(record)
        examples.append(dict(**record,request_ids=ids,service_response_ids=service_ids,cache_key=group[0]['meta']['key']))
    assert len(cases)==148 and all(r['dataset']=='wa' for r in cases)
    result=dict(dataset='wa',n_expected=200,n_complete=148,mean_std=float(np.mean([r['p_std'] for r in cases])),max_range=max(r['p_range'] for r in cases),
        n_any_probability_change=sum(r['p_range']>0 for r in cases),n_binary_flip=sum(r['binary_flip'] for r in cases))
    result.update(fraction_any_probability_change=result['n_any_probability_change']/148,fraction_binary_flip=result['n_binary_flip']/148)
    original=pd.read_csv(source/'repeat_summary.csv').query('dataset == "wa"').iloc[0]
    for key in ['mean_std','max_range','fraction_any_probability_change','fraction_binary_flip']:
        assert abs(result[key]-original[key])<1e-12,key
    maximum=max(examples,key=lambda r:r['p_range']);flip=next(r for r in examples if r['binary_flip'])
    (out/'counterexamples.json').write_text(json.dumps([maximum,flip],indent=2),encoding='utf8')
    pd.DataFrame(cases).to_csv(out/'verified_cases.csv',index=False)
    prior=json.loads((BASE/'runs/overnight_finalization_20260930/status.json').read_text(encoding='utf8'))
    usage_estimate=prior['api_main_spent']+prior['repair_spent']+prior['route_pilot_spent']+float(pd.read_csv(BASE/'runs/remote_cost_probe_inferera_20260930/ledger.csv').paid_usd.sum())+meta['paid']
    first=min(balance,key=lambda r:r['time'])
    report=dict(status='VERIFIED_PARTIAL_FRESH_REPEATS_NOT_FROZEN',summary=result,http_status_counts=dict(http),
        first_balance_refusal_utc=first['time'],balance_refused_requests=len(balance),n_requests_after_first_balance_refusal=sum(r['time']>first['time'] for r in requests.values()),
        usage_price_estimate_usd=usage_estimate,actual_account_debit='UNAVAILABLE',inference='Strict endpoint determinism refuted by same-body counterexamples; rates apply only to 148 complete WA cases, not the full selected sample or other datasets.',
        execution_failure='Original collector failed to stop at the first wallet refusal; original executed_source and rejected HTTP responses are retained. Future source is patched; no rerun is authorized with the empty wallet.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest())
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    report['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'status.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
