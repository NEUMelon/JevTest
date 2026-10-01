"""Recover confidence fields and selective decisions without extra API calls."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from .calibration_ci import metric_batch
from .core import probability

BASE = Path(__file__).resolve().parents[2]
RUN = BASE/'runs/recovery_repaired_20260930'
OUT = BASE/'runs/selective_recovery_20260930'


def threshold_intervals(correct, confidence, threshold):
    selected = confidence > threshold
    n = len(correct)
    counts = np.array([np.sum(selected & correct), np.sum(selected & ~correct), np.sum(~selected)])
    draws = np.random.default_rng(20260930).multinomial(n, counts/n, size=2000)
    retained = draws[:,0]+draws[:,1]
    accuracy = np.divide(draws[:,0], retained, out=np.full(2000,np.nan), where=retained>0)
    coverage = retained/n
    available = retained>0
    lo,hi = np.percentile(accuracy[available],[2.5,97.5]) if available.any() else [np.nan,np.nan]
    return dict(n_selected=int(selected.sum()),coverage=float(selected.mean()),
        accuracy=float(np.mean(correct[selected])) if selected.any() else np.nan,
        coverage_ci_low=float(np.percentile(coverage,2.5)),coverage_ci_high=float(np.percentile(coverage,97.5)),
        accuracy_ci_low=lo,accuracy_ci_high=hi,empty_bootstrap_draws=int(np.sum(~available)))


def main():
    OUT.mkdir(exist_ok=False);(OUT/'curves').mkdir();inputs={};rows=[];calibration=[];confidence_rows=[]
    def track(p):inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    track(Path(__file__));track(BASE/'src/recovery/calibration_ci.py');track(BASE/'src/recovery/core.py')
    db=sqlite3.connect((BASE/'cache/calls.sqlite').as_uri()+'?mode=ro',uri=True)
    repairs=BASE/'runs/api_jev_repairs_20260930/observations.json';track(repairs)
    repair_by_key={r['meta']['key']:r['response'] for r in json.loads(repairs.read_text(encoding='utf8'))}
    for ds in ['wa','ag','da','ab']:
        path=RUN/f'features/{ds}_test_H.csv';track(path);f=pd.read_csv(path)
        for _,r in f.iterrows():
            hit=db.execute('SELECT response FROM calls WHERE key=?',(r.cache_key,)).fetchone()
            response=json.loads(hit[0]) if hit else repair_by_key.get(r.cache_key)
            if response is None:raise ValueError('Raw confidence response unavailable')
            for system,qid,cols in [('J-H1s','H1_score',['score_0','score_1','score_2']),
                                    ('J-H1c','H1_choice',['choice_same','choice_different','choice_cannot_tell'])]:
                a=response['answers'][qid];confidence=probability(a['confidence']) if 'confidence' in a else np.nan
                pmatch=r.p_h1s if system=='J-H1s' else r.p_h1c
                confidence_rows.append(dict(dataset=ds,system=system,pair_id=r.pair_id,label=r.label,p_match=pmatch,
                    binary_decision_correct=int((pmatch>=.5)==r.label),reported_confidence=confidence,
                    max_option_probability=max(float(r[c]) for c in cols),cache_key=r.cache_key,
                    note='Correctness of binary match threshold decision, not correctness of an ordinal score or cannot_tell response'))
        for p in sorted((RUN/'predictions').glob(f'{ds}_J-*_b*_s*.csv')):
            track(p);pred=pd.read_csv(p)
            if not np.isfinite(pred.probability).all():raise ValueError('Incomplete selective predictions')
            correct=pred.prediction.values==pred.label.values
            # The probability assigned to the actual decision, including when
            # OOF threshold is not 0.5; max(p,1-p) could describe the other class.
            conf=np.where(pred.prediction.values==1,pred.probability.values,1-pred.probability.values)
            for t in [.95,.99]:
                rows.append(dict(artifact=p.stem,dataset=ds,confidence_source='probability_of_actual_binary_decision',threshold=t,
                    **threshold_intervals(correct,conf,t),status='DESCRIPTIVE_NOT_FROZEN'))
            order=np.argsort(-conf,kind='stable');counts=np.arange(1,len(conf)+1);sorted_conf=conf[order]
            endpoints=np.flatnonzero(np.r_[sorted_conf[:-1]!=sorted_conf[1:],True])
            cumulative=np.cumsum(~correct[order])
            pd.DataFrame(dict(confidence=sorted_conf[endpoints],n_selected=counts[endpoints],
                coverage=counts[endpoints]/len(conf),risk=cumulative[endpoints]/counts[endpoints])).to_csv(OUT/f'curves/{p.stem}.csv',index=False)
            if '+LR' in p.stem:
                m=metric_batch(pred.label.values,pred.probability.values)
                calibration.append(dict(artifact=p.stem,dataset=ds,**{k:float(v[0]) for k,v in m.items()},
                    status='POINT_ESTIMATE_NOT_FROZEN',note='Probability of match, independent of tuned decision threshold'))
    db.close()
    features=pd.DataFrame(confidence_rows);features.to_csv(OUT/'confidence_features.csv',index=False)
    confidence_metrics=[]
    for (ds,system),g in features.groupby(['dataset','system']):
        for col in ['reported_confidence','max_option_probability']:
            valid=np.isfinite(g[col]);y=g.loc[valid,'binary_decision_correct'].values;p=g.loc[valid,col].values
            m=metric_batch(y,p)
            confidence_metrics.append(dict(dataset=ds,system=system,confidence_source=col,n=len(p),n_missing=int((~valid).sum()),
                **{k:float(v[0]) for k,v in m.items()},
                target='Correctness of binary p(match)>=0.5; not ordinal/abstention correctness',status='DESCRIPTIVE_NOT_FROZEN'))
    pd.DataFrame(confidence_metrics).to_csv(OUT/'confidence_correctness_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'aggregator_calibration.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'selective_thresholds.csv',index=False)
    (OUT/'source_snapshot').mkdir()
    for name in ['selective.py','calibration_ci.py','core.py']:(OUT/'source_snapshot'/name).write_bytes((BASE/'src/recovery'/name).read_bytes())
    manifest=dict(status='DESCRIPTIVE_NOT_FROZEN',result_freeze=False,input_sha256=inputs,bootstrap_replicates=2000,
        seed=20260930,coverage_rule='confidence > threshold, tied ranks grouped in curves',
        caveat='Explicit confidence semantics and multiclass correctness cannot be fully evaluated using only binary gold labels; binary correctness target is declared',
        output_sha256={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(f'SELECTIVE_COMPLETE prediction_artifacts={len(rows)//2} aggregator_metrics={len(calibration)}')


if __name__=='__main__':main()
