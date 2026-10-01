"""Final local interval checks. No model calls, training, or threshold changes."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import beta

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/final_interval_review_20261001'

def exact(k,n):
    if not n:return np.nan,np.nan
    return (0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k)))

def main():
    OUT.mkdir(exist_ok=False);inputs={}
    def read(rel):
        p=BASE/rel;inputs[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p)
    root='runs/qwen_diagnostics_20260930'
    attacks=read(root+'/attacks.csv')
    ditto=read('runs/gpu_backup_20260930/qwen_recovery_20260930/ditto_injection_predictions.csv')
    code=read(root+'/code_injection/predictions.csv')
    cache={}
    def q(ds,system,experiment,variant):
        key=(ds,system,experiment,variant)
        if key not in cache:cache[key]=read(f'{root}/predictions/{ds}_{system}_{experiment}_{variant}.csv').set_index('pair_id',verify_integrity=True)
        return cache[key]
    for row in attacks.itertuples(index=False):
        if row.system.startswith('Q-'):
            f=q(row.dataset,row.system,row.experiment,row.variant)
            eligible=f.clean_prediction==f.label
            if row.denominator.startswith('Common'):
                other=q(row.dataset,'Q-D+LR' if row.system=='Q-H1' else 'Q-H1',row.experiment,row.variant).loc[f.index]
                if not np.array_equal(f.label,other.label):raise ValueError('Common attack label mismatch')
                eligible=eligible&(other.clean_prediction==other.label)
        else:
            frame=ditto if row.system=='Ditto-full' else code
            subset=frame[frame.dataset==row.dataset]
            clean=subset[subset.variant=='clean'].set_index('pair_id',verify_integrity=True)
            f=subset[subset.variant==row.variant].set_index('pair_id',verify_integrity=True).loc[clean.index]
            if not np.array_equal(f.label,clean.label):raise ValueError('Attack labels mismatch')
            eligible=clean.prediction==clean.label
        events=int(((f.prediction!=f.label)&eligible).sum());n=int(eligible.sum())
        if events!=row.n_events or n!=row.n_eligible or len(f)!=row.n_test:raise ValueError('Attack count does not reproduce')
    attacks['cp95_low'],attacks['cp95_high']=zip(*(exact(int(r.n_events),int(r.n_eligible)) for r in attacks.itertuples(index=False)))
    attacks['interval_note']='Two-sided exact binomial 95%; conditional on clean-correct subset; finite fixed benchmark, not universal attack distribution'
    attacks.to_csv(OUT/'attacks_exact_binomial.csv',index=False)
    stability=read(root+'/stability.csv')
    for i,r in stability.iterrows():
        f=read(f"{root}/predictions/{r.dataset}_{r.system}_{r.variant}.csv")
        events=int((f.prediction!=f.clean_prediction).sum());n=len(f)
        if abs(events/n-r.flip_rate)>1e-12 or n!=r.n_test:raise ValueError('Stability rate does not reproduce')
        stability.loc[i,'n_flip']=events
        stability.loc[i,'flip_cp95_low'],stability.loc[i,'flip_cp95_high']=exact(events,n)
    stability.to_csv(OUT/'qwen_flip_exact_binomial.csv',index=False)
    # These are count-based intervals; zero observed errors never establish immunity.
    selective=read(root+'/selective_thresholds.csv')
    for i,r in selective.iterrows():
        n=int(r.n_selected);k=round(r.accuracy*n) if n else 0
        if n and abs(k/n-r.accuracy)>1e-12:raise ValueError('Non-integer selected correct count')
        selective.loc[i,'accuracy_cp95_low'],selective.loc[i,'accuracy_cp95_high']=exact(k,n)
        selective.loc[i,'coverage_cp95_low'],selective.loc[i,'coverage_cp95_high']=exact(n,int(r.n_test))
    selective.to_csv(OUT/'selective_counts_exact_binomial.csv',index=False)
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(dict(status='COUNTS_REPRODUCED_NONDEGENERATE_BOUNDARY_INTERVALS',
        n_attack_rows=len(attacks),n_stability_rows=len(stability),inputs=inputs,
        note='Historical empirical bootstrap intervals are retained. Use these exact intervals for marginal binary rates; paired differences keep paired bootstrap. No universal population or independence claim.',
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}),indent=2),encoding='utf8')
    print(json.dumps(dict(status='VERIFIED',attacks=len(attacks),stability=len(stability))))

if __name__=='__main__':main()
