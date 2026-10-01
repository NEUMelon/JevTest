"""Trace legacy ED LLM probabilities to cached responses, without defaults."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from .core import probability, metrics
from src.experiments.e7_error_detection import load_ed_dataset

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/ed_llm_verify_20260930'


def main():
    OUT.mkdir(exist_ok=False);features=OUT/'features';features.mkdir();inputs={};summaries=[];issues=[]
    db=sqlite3.connect((BASE/'cache/calls.sqlite').as_uri()+'?mode=ro',uri=True)
    def track(p):inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    for ds in ['ho','ad','fl']:
        track(BASE/f'data/canonical/{ds}/test.jsonl')
        groups=load_ed_dataset(ds,'test')[:200]
        for model in ['gpt-6-luna','deepseek-v4-flash']:
            path=BASE/f'runs/e7_llm_{ds}_{model}.csv';track(path)
            legacy=pd.read_csv(path).set_index('cell_id',verify_integrity=True);rows=[]
            for group in groups:
                cols=[c['col'] for c in group['cells']]
                prompt=(f"You are a database error detection system.\n"
                    f"Given the following database row:\n{json.dumps(group['row'],indent=2)}\n\n"
                    f"For each of the following columns: {cols}\n"
                    f"Determine if the cell value contains an error (typo, invalid format, missing placeholder, or inconsistent value).\n"
                    f"Respond with a JSON object where keys are the column names and values are error probabilities between 0.0 (clean) and 1.0 (erroneous).\n"
                    f"Example: {{\"col_name\": 0.05}}\nJSON:")
                payload=dict(model=model,messages=[dict(role='user',content=prompt)],temperature=0.,
                    max_tokens=600 if 'luna' in model else 500,response_format={'type':'json_object'})
                if 'luna' in model:payload['reasoning_effort']='low'
                key=hashlib.sha256(json.dumps(['aihubmix',payload],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                hit=db.execute('SELECT response FROM calls WHERE key=?',(key,)).fetchone();obj=None;response=None;decode_error=None
                if hit:
                    response=json.loads(hit[0])
                    try:
                        content=response['choices'][0]['message']['content'];match=re.search(r'\{.*\}',content,re.DOTALL)
                        obj=json.loads(match.group(0) if match else content)
                        if not isinstance(obj,dict):raise ValueError('Expected column probability mapping')
                    except (ValueError,TypeError,KeyError,IndexError) as e:decode_error=type(e).__name__
                for c in group['cells']:
                    old=legacy.loc[c['cell_id']]
                    if int(old.label_error)!=c['label_error']:raise ValueError('Legacy ED label mismatch')
                    p=np.nan;status='cache_missing' if not hit else 'parse_failed' if obj is None else 'column_missing'
                    if obj is not None and c['col'] in obj:
                        try:p=probability(obj[c['col']]);status='verified_cache'
                        except (ValueError,TypeError):status='invalid_probability'
                    changed=bool(np.isfinite(p) and abs(p-float(old.p_error))>1e-9)
                    r=dict(cell_id=c['cell_id'],table_row_id=group['table_row_id'],col=c['col'],label=c['label_error'],
                        probability=p,legacy_probability=float(old.p_error),status=status,legacy_probability_mismatch=changed,
                        cache_key=key,response_model=response.get('model') if response else None)
                    rows.append(r)
                    if status!='verified_cache' or changed:issues.append(dict(dataset=ds,model=model,**r,parse_error=decode_error))
            frame=pd.DataFrame(rows)
            if set(frame.cell_id)!=set(legacy.index):raise ValueError('Legacy LLM cohort differs from first 200 rows')
            frame.to_csv(features/f'{ds}_{model}.csv',index=False)
            good=np.isfinite(frame.probability)
            summaries.append(dict(dataset=ds,model=model,n_expected=len(frame),n_verified=int(good.sum()),
                n_changed=int(frame.legacy_probability_mismatch.sum()),status='COMPLETE_DESCRIPTIVE_NOT_FROZEN' if good.all() else 'PARTIAL_BLOCKED',
                **(metrics(frame.label.values[good],frame.probability.values[good]) if good.any() else {}),
                note='Legacy first-200-row cohort, no fallback probabilities, not full-test/literature-matched evaluation'))
    db.close();track(BASE/'cache/calls.sqlite');track(BASE/'src/experiments/e7_error_detection.py');track(Path(__file__))
    pd.DataFrame(summaries).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'issues.json').write_text(json.dumps(issues,indent=2),encoding='utf8')
    manifest=dict(status='DESCRIPTIVE_NOT_FROZEN',result_freeze=False,input_sha256=inputs,
        legacy_cache_caveat='Original cache may overwrite repeated identical requests; matching raw probabilities supports but does not restore missing original attempt logs',
        output_sha256={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(pd.DataFrame(summaries)[['dataset','model','n_expected','n_verified','n_changed','f1','status']].to_string(index=False))


if __name__=='__main__':main()
