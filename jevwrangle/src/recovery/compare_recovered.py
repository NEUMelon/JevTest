"""Compare newly completed systems on aligned IDs, with crossed bootstrap."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .statistics import paired_bootstrap
BASE=Path(__file__).resolve().parents[2]

@threadpool_limits.wrap(limits=1)
def interaction_bootstrap(predictions,replicates=2000,seed=20260930):
    # Per seed: y, J-H, J-D, comparator-H, comparator-D. Shared row draws
    # preserve all four decisions and correlation across training seeds.
    truth=np.asarray(predictions[0][0]);s=len(predictions);n=len(truth)
    if any(not np.array_equal(r[0],truth) for r in predictions):raise ValueError('Misaligned labels')
    matrix=np.stack([truth]+[np.asarray(r[k]) for r in predictions for k in [1,2,3,4]],axis=1)
    unique,counts=np.unique(matrix,axis=0,return_counts=True);rng=np.random.default_rng(seed)
    seed_weights=rng.multinomial(s,np.full(s,1/s),size=replicates)/s
    weights=rng.multinomial(n,counts/n,size=replicates);y=unique[:,0,None]
    sampled=[];original=[]
    for k in [1,2,3,4]:
        pred=unique[:,k::4];positive=(y==1)&(pred==1);falsepos=(y==0)&(pred==1);falseneg=(y==1)&(pred==0)
        tp=weights@positive;denom=2*tp+weights@falsepos+weights@falseneg
        scores=np.divide(200*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
        sampled.append(np.sum(scores*seed_weights,axis=1))
        tp=counts@positive;denom=2*tp+counts@falsepos+counts@falseneg
        original.append(float(np.divide(200*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0).mean()))
    delta=sampled[1]-sampled[0]-sampled[3]+sampled[2];point=original[1]-original[0]-original[3]+original[2]
    return dict(j_decomposition_gain=original[1]-original[0],comparator_decomposition_gain=original[3]-original[2],
        interaction=point,ci_low=float(np.percentile(delta,2.5)),ci_high=float(np.percentile(delta,97.5)),
        n_test=n,n_seeds=s,replicates=replicates,interpretation='Descriptive CI; bootstrap tail is not a formal p-value')

def main(out):
    out.mkdir(exist_ok=False);sources={};rows=[];interactions=[]
    jev=BASE/'runs/recovery_repaired_20260930/predictions'
    models=[('Q',BASE/'runs/overnight_synthesis_20260930/qwen_evaluation'),
            ('L1',BASE/'runs/overnight_finalization_20260930/llm_evaluation'),
            ('L2',BASE/'runs/overnight_finalization_20260930/llm_evaluation')]
    def load(p):
        sources[str(p.relative_to(BASE))]=hashlib.sha256(p.read_bytes()).hexdigest()
        f=pd.read_csv(p).set_index('pair_id',verify_integrity=True)
        if f.probability.isna().any():raise ValueError('Missing predictions')
        return f
    for ds in ['wa','ag','da','ab']:
        for budget,seeds in [(50,10),(200,10),(1000,5),('full',3)]:
            pairs=[];hybrid_pairs=[]
            for seed in range(seeds):
                a=load(jev/f'{ds}_J-D+LR_b{budget}_s{seed}.csv')
                b=load(BASE/f'runs/magellan_v2_20260930/predictions/{ds}_M2+best_b{budget}_s{seed}.csv').loc[a.index]
                if not np.array_equal(a.label,b.label):raise ValueError('String baseline label mismatch')
                pairs.append((a.label.values,a.prediction.values,b.prediction.values))
                hybrid=load(jev/f'{ds}_J-D+C+LR_b{budget}_s{seed}.csv').loc[a.index]
                if not np.array_equal(a.label,hybrid.label):raise ValueError('Hybrid feature label mismatch')
                hybrid_pairs.append((a.label.values,a.prediction.values,hybrid.prediction.values))
            rows.append(dict(dataset=ds,budget=budget,system_A='J-D+LR',system_B='Custom string features + CV-selected classifier',**paired_bootstrap(pairs)))
            rows.append(dict(dataset=ds,budget=budget,system_A='J-D+LR',system_B='J-D+C+LR',
                endpoint='Feature augmentation comparison; not a substitute for missing human semantic atomic gold',**paired_bootstrap(hybrid_pairs)))
        for short,folder in models:
            for budget in [50,200]:
                pairs=[];four=[]
                for seed in range(3 if short=='L1' else 10):
                    paths=[jev/f'{ds}_J-H1+LR_b{budget}_s{seed}.csv',jev/f'{ds}_J-D+LR_b{budget}_s{seed}.csv',
                           folder/f'predictions/{ds}_{short}-H1+LR_b{budget}_s{seed}.csv',folder/f'predictions/{ds}_{short}-D+LR_b{budget}_s{seed}.csv']
                    if not all(p.exists() for p in paths):break
                    h,d,mh,md=[load(p) for p in paths];ids=md.index
                    h,d,mh=h.loc[ids],d.loc[ids],mh.loc[ids]
                    if any(not np.array_equal(f.label,md.label) for f in [h,d,mh]):raise ValueError('Aligned IDs have conflicting labels')
                    y=md.label.values;pairs.append((y,d.prediction.values,md.prediction.values));four.append((y,h.prediction.values,d.prediction.values,mh.prediction.values,md.prediction.values))
                expected=3 if short=='L1' else 10
                if len(pairs)!=expected:continue
                rows.append(dict(dataset=ds,budget=budget,system_A='J-D+LR',system_B=short+'-D+LR',**paired_bootstrap(pairs)))
                if budget==200:interactions.append(dict(dataset=ds,comparator=short,**interaction_bootstrap(four)))
        for budget in [50,200,1000]:
            pairs=[]
            for seed in range(3):
                choices=[]
                for m in (BASE/'runs/gpu_backup_20260930/ditto').glob('*/manifest.json'):
                    meta=json.loads(m.read_text(encoding='utf8'));j=meta['job']
                    if (j['dataset'],j['budget'],j['seed'])==(ds,budget,seed) and meta['status']=='TRAINED_NOT_FROZEN':choices.append(m.parent/'test_predictions.csv')
                if len(choices)!=1:break
                a=load(jev/f'{ds}_J-D+LR_b{budget}_s{seed}.csv');b=load(choices[0]).loc[a.index]
                if not np.array_equal(a.label,b.label):raise ValueError('Ditto label mismatch')
                pairs.append((a.label.values,a.prediction.values,b.prediction.values))
            if len(pairs)==3:rows.append(dict(dataset=ds,budget=budget,system_A='J-D+LR',system_B='Ditto',**paired_bootstrap(pairs)))
    pd.DataFrame(rows).to_csv(out/'paired_system_comparisons.csv',index=False)
    pd.DataFrame(interactions).to_csv(out/'h2_interaction.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_GROUPS_ONLY_DESCRIPTIVE',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs=sources,bootstrap='Crossed seeds plus SAME paired rows shared by all seeds and systems',
        Holm='Not presented as whole preregistered family; future drift and human H5 endpoint not observed'),indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output.resolve())
