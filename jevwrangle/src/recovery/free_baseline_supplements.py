"""Align new baseline to frozen Luna IDs and content-dedup diagnostic masks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .statistics import paired_bootstrap

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'runs/free_completion_20261001'
OUT=ROOT/'baseline_population_supplements'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def counts(f):
    y=f.label.to_numpy();p=f.prediction.to_numpy()
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,n_test=len(f),f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def main():
    OUT.mkdir(exist_ok=False)
    source=ROOT/'magellan_author_library'
    assert json.loads((ROOT/'baseline_independent_verification/manifest.json').read_text())['n_groups']==224
    mask_path=BASE/'runs/em_content_sensitivity_frozen_20260930/manifest.json'
    masks=json.loads(mask_path.read_text(encoding='utf8'))['datasets']
    inputs={str(mask_path):sha(mask_path)};matched=[];sensitive=[];comparisons=[]
    for ds in ['wa','ag','da','ab']:
        luna_path=BASE/f'runs/overnight_finalization_20260930/llm_evaluation/predictions/{ds}_L1-D+LR_b200_s0.csv'
        luna=pd.read_csv(luna_path);inputs[str(luna_path)]=sha(luna_path)
        assert len(luna)==500 and luna.pair_id.is_unique
        keep=set(masks[ds]['keep_ids'])
        for system in ['M3-author+LR','M3-author+best']:
            for budget in [50,200,1000,'full']:
                for seed in range(3 if budget=='full' else 5 if budget==1000 else 10):
                    p=source/f'predictions/{ds}_{system}_b{budget}_s{seed}.csv'
                    f=pd.read_csv(p);inputs[str(p)]=sha(p)
                    assert f.pair_id.is_unique
                    original=counts(f)
                    filtered=f[f.pair_id.isin(keep)]
                    assert set(filtered.pair_id)==keep
                    c=counts(filtered)
                    sensitive.append(dict(dataset=ds,system=system,budget=budget,seed=seed,
                        full_test_f1=original['f1'],delta_filtered_minus_full=c['f1']-original['f1'],
                        protocol='POST_AUDIT_TEST_FILTER_ONLY_NO_RETRAIN',**c))
                    if budget in [50,200] and seed<3:
                        subset=f.set_index('pair_id').loc[luna.pair_id].reset_index()
                        assert subset.label.tolist()==luna.label.tolist()
                        matched.append(dict(dataset=ds,system=system,budget=budget,seed=seed,
                            protocol='FROZEN_LUNA500_SEEDS_0_1_2',**counts(subset)))
            for budget in [50,200]:
                pairs=[]
                for seed in range(10):
                    a=source/f'predictions/{ds}_{system}_b{budget}_s{seed}.csv'
                    b=BASE/f'runs/recovery_repaired_20260930/predictions/{ds}_J-D+LR_b{budget}_s{seed}.csv'
                    inputs[str(b)]=sha(b)
                    fa,fb=[pd.read_csv(p).set_index('pair_id').loc[sorted(keep)] for p in [a,b]]
                    assert fa.label.tolist()==fb.label.tolist()
                    pairs.append((fa.label.to_numpy(),fa.prediction.to_numpy(),fb.prediction.to_numpy()))
                comparisons.append(dict(dataset=ds,budget=budget,system_A=system,system_B='J-D+LR',
                    protocol='POST_AUDIT_CONTENT_FILTER_ONLY_DESCRIPTIVE',**paired_bootstrap(pairs)))
    assert len(matched)==48 and len(sensitive)==224 and len(comparisons)==16
    mf=pd.DataFrame(matched);sf=pd.DataFrame(sensitive)
    mf.to_csv(OUT/'matched500_by_seed.csv',index=False)
    mf.groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),n_seeds=('seed','count'),n_test=('n_test','first')).reset_index().to_csv(OUT/'matched500_summary.csv',index=False)
    sf.to_csv(OUT/'content_sensitivity_by_seed.csv',index=False)
    sf.groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),full_test_f1_mean=('full_test_f1','mean'),delta_mean=('delta_filtered_minus_full','mean'),n_seeds=('seed','count'),n_test=('n_test','first')).reset_index().to_csv(OUT/'content_sensitivity_summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'content_filtered_paired_bootstrap.csv',index=False)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    inputs[str(BASE/'src/recovery/statistics.py')]=sha(BASE/'src/recovery/statistics.py')
    m=dict(status='BASELINE_POPULATION_ALIGNMENT_COMPLETE',n_matched500_groups=48,n_content_groups=224,n_filtered_comparisons=16,
        paid_api_calls=0,gpu_calls=0,inputs=inputs,
        limitations=['Content mask is post-audit diagnostic; test population changes and training stays unchanged.',
                     'No causal proof of no leakage or pretraining exposure; fixed mask applied to all systems.',
                     'Luna comparison uses only same frozen500 IDs and same first three seeds.'])
    m['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8')
    print(json.dumps({k:m[k] for k in ['status','n_matched500_groups','n_content_groups','n_filtered_comparisons']}))


if __name__=='__main__':main()
