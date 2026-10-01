"""Paired hierarchical bootstrap and per-seed exact McNemar diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def f1_counts(counts,which):
    states=np.arange(8);truth=states//4;pred=(states//2)%2 if which=='A' else states%2
    tp=counts[..., (truth==1)&(pred==1)].sum(axis=-1)
    fp=counts[..., (truth==0)&(pred==1)].sum(axis=-1)
    fn=counts[..., (truth==1)&(pred==0)].sum(axis=-1)
    denom=2*tp+fp+fn
    return np.divide(200*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)


def paired_bootstrap(predictions,replicates=2000,seed=20260930):
    # Seeds and test rows are crossed: the SAME test item occurs under every
    # seed. One common row draw must be applied to all sampled seeds, rather
    # than pretending their test sets are independent nested samples.
    from threadpoolctl import threadpool_limits
    truth=np.asarray(predictions[0][0],dtype=int);n=len(truth);s=len(predictions)
    if any(not np.array_equal(truth,np.asarray(p[0])) for p in predictions):raise ValueError('Seeds must share aligned test labels')
    patterns=np.stack([truth]+[np.asarray(p[k],dtype=int) for p in predictions for k in [1,2]],axis=1)
    unique,counts=np.unique(patterns,axis=0,return_counts=True)
    rng=np.random.default_rng(seed)
    seed_weights=rng.multinomial(s,np.full(s,1/s),size=replicates)/s
    with threadpool_limits(limits=1):
        row_weights=rng.multinomial(n,counts/n,size=replicates)
        samples=[];original=[]
        for which in [1,2]:
            pred=unique[:,which::2];y=unique[:,0,None]
            tp=row_weights@((y==1)&(pred==1));fp=row_weights@((y==0)&(pred==1));fn=row_weights@((y==1)&(pred==0))
            denom=2*tp+fp+fn;f=np.divide(200*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
            samples.append(np.sum(f*seed_weights,axis=1))
            tp0=counts@((y==1)&(pred==1));fp0=counts@((y==0)&(pred==1));fn0=counts@((y==1)&(pred==0))
            original.append(float(np.mean(np.divide(200*tp0,2*tp0+fp0+fn0,out=np.zeros_like(tp0,dtype=float),where=(2*tp0+fp0+fn0)>0))))
    a,b=samples;originalA,originalB=original
    delta=b-a;low,high=np.percentile(delta,[2.5,97.5])
    return dict(f1_A=originalA,f1_B=originalB,delta_B_minus_A=originalB-originalA,
                delta_ci_low=float(low),delta_ci_high=float(high),
                f1_A_ci_low=float(np.percentile(a,2.5)),f1_A_ci_high=float(np.percentile(a,97.5)),
                f1_B_ci_low=float(np.percentile(b,2.5)),f1_B_ci_high=float(np.percentile(b,97.5)),
                replicates=replicates,n_seeds=s,n_test=n,
                bootstrap_design='crossed seed resampling plus common paired test-row resampling',
                bootstrap_tail_fraction=float(2*min(np.mean(delta<=0),np.mean(delta>=0))))


def main(run,output_name='statistics'):
    output=run/output_name;output.mkdir(exist_ok=False);summary=[];mcnemar=[];inputs={}
    for ds in ['wa','ag','da','ab']:
        for hypothesis,A,B,budgets in [('H1','J-H1+LR','J-D+LR',[200]),('H3','M+LR','J-D+LR',[50,200])]:
            for budget in budgets:
                predictions=[]
                for seed in range(10):
                    files=[run/f'predictions/{ds}_{s}_b{budget}_s{seed}.csv' for s in [A,B]]
                    a,b=[pd.read_csv(p) for p in files]
                    for p in files:inputs[str(p.relative_to(run))]=hashlib.sha256(p.read_bytes()).hexdigest()
                    assert a.pair_id.tolist()==b.pair_id.tolist() and a.label.tolist()==b.label.tolist()
                    assert (a.prediction>=0).all() and (b.prediction>=0).all()
                    y=a.label.values;pa=a.prediction.values;pb=b.prediction.values
                    predictions.append((y,pa,pb))
                    wrongA=pa!=y;wrongB=pb!=y
                    improved=int(np.sum(wrongA & ~wrongB));worsened=int(np.sum(~wrongA & wrongB));discord=improved+worsened
                    pvalue=float(binomtest(improved,discord,.5).pvalue) if discord else 1.
                    mcnemar.append(dict(hypothesis=hypothesis,dataset=ds,budget=budget,seed=seed,
                                        improved=improved,worsened=worsened,p_exact=pvalue,
                                        note='Tests equality of paired error rates, not equality of F1; do not pool repeated test items across seeds'))
                summary.append(dict(hypothesis=hypothesis,dataset=ds,budget=budget,system_A=A,system_B=B,
                                    **paired_bootstrap(predictions),status='UNFROZEN_DESCRIPTIVE',
                                    note='Bootstrap tail fraction is not labeled as a formal p-value; whole primary Holm family incomplete'))
    pd.DataFrame(summary).to_csv(output/'paired_hierarchical_bootstrap.csv',index=False)
    pd.DataFrame(mcnemar).to_csv(output/'mcnemar_by_seed.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(dict(status='UNFROZEN_DESCRIPTIVE',inputs=inputs,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=20260930,
        Holm_status='NOT_APPLIED: H2/H5/H6 primary comparisons unavailable; do not adjust a selected subset as the full preregistered family'),indent=2),encoding='utf8')
    print(pd.DataFrame(summary)[['hypothesis','dataset','budget','delta_B_minus_A','delta_ci_low','delta_ci_high']].to_string(index=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();main(a.run.resolve())
