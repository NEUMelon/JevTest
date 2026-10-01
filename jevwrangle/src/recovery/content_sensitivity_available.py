"""Offline interim diagnostic while GPU transport is unavailable; no final claim."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from .core import metrics

BASE=Path(__file__).resolve().parents[2]

def main():
    out=BASE/'runs/em_content_sensitivity_available_20261001'
    out.mkdir(exist_ok=False)
    mask_path=BASE/'runs/em_content_sensitivity_frozen_20260930/manifest.json'
    mask=json.loads(mask_path.read_text(encoding='utf8'))
    inputs={mask_path.relative_to(BASE).as_posix():hashlib.sha256(mask_path.read_bytes()).hexdigest()}
    for rel,expected in mask['input_sha256'].items():
        if hashlib.sha256((BASE/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen canonical input changed')
    rows=[]
    for ds in ['wa','ag','da','ab']:
        truth={r['pair_id']:r['label'] for r in map(json.loads,(BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines())}
        keep=set(mask['datasets'][ds]['keep_ids'])
        jobs=[]
        for folder,systems in [('recovery_repaired_20260930',['J-D+LR','J-D+C+LR']),('magellan_v2_20260930',['M2+best']),('overnight_finalization_20260930/llm_evaluation',['L2-D+LR'])]:
            for system in systems:
                for seed in range(10):jobs.append((BASE/f'runs/{folder}/predictions/{ds}_{system}_b200_s{seed}.csv',system,seed,None))
        for p in (BASE/'runs/gpu_backup_20260930/ditto').glob('*/manifest.json'):
            m=json.loads(p.read_text(encoding='utf8'));j=m['job']
            if j['dataset']==ds and j['budget']==200:
                jobs.append((p.parent/'test_predictions.csv','Ditto',j['seed'],m['threshold']))
        for p,system,seed,t in jobs:
            f=pd.read_csv(p).set_index('pair_id',verify_integrity=True)
            if set(f.index)!=set(truth) or any(truth[k]!=v for k,v in zip(f.index,f.label)):raise ValueError('Full canonical ID/label mismatch')
            threshold=float(f.threshold.iloc[0]) if t is None else t
            if not np.array_equal(f.prediction,(f.probability>=threshold).astype(int)):raise ValueError('Frozen prediction mismatch')
            g=f.loc[[k for k in f.index if k in keep]]
            full=metrics(f.label,f.probability,threshold);reduced=metrics(g.label,g.probability,threshold)
            rows.append(dict(dataset=ds,system=system,budget=200,seed=seed,n_original=len(f),n_retained=len(g),f1_original=full['f1'],f1_retained=reduced['f1'],delta_f1=reduced['f1']-full['f1'],tp=reduced['tp'],fp=reduced['fp'],fn=reduced['fn'],tn=reduced['tn']))
            inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    if len(rows)!=172:raise ValueError('Expected all 172 available b200 groups')
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics_by_seed.csv',index=False)
    frame.groupby(['dataset','system']).agg(f1_original_mean=('f1_original','mean'),f1_retained_mean=('f1_retained','mean'),delta_f1_mean=('delta_f1','mean'),n_seeds=('seed','count'),n_retained=('n_retained','first')).reset_index().to_csv(out/'summary.csv',index=False)
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(status='INTERIM_AVAILABLE_SYSTEMS_ONLY_GPU_PENDING',n_prediction_groups=len(rows),inputs=inputs,interpretation=mask['interpretation'],missing='Qwen/NLI unavailable locally; Luna omitted from this full-test diagnostic. Full final diagnostic remains pending.',outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2),encoding='utf8')
    print(frame.groupby(['dataset','system']).delta_f1.mean().to_string())

if __name__=='__main__':main()
