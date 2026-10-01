"""CPU-only declared fallback using the Magellan team's py_stringmatching.

Not the official py_entitymatching/Narayan experiment. Post-audit validation;
reuse existing budget IDs and training-only CV, never tune to reference/test F1.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import py_stringmatching as sm
from src.recovery.magellan_v2 import fit
from src.recovery.core import metrics

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/free_completion_20261001/magellan_author_library'
ATTRS={'wa':['title','category','brand','modelno'],'ag':['title','manufacturer'],
       'da':['title','authors','venue'],'ab':['name','description']}
MISSING={'','none','null','nan','n/a'}
QT=sm.QgramTokenizer(qval=3,return_set=True)
WT=sm.AlphanumericTokenizer(return_set=True)
J=sm.Jaccard();CO=sm.Cosine();LE=sm.Levenshtein();JW=sm.JaroWinkler();ME=sm.MongeElkan()


def norm(v):
    if v is None or str(v).strip().lower() in MISSING:return ''
    return str(v).strip().lower()


@lru_cache(maxsize=150000)
def similarities(a,b):
    if not a or not b:return [np.nan]*7+[1.]
    at,bt=WT.tokenize(a),WT.tokenize(b)
    return [J.get_raw_score(QT.tokenize(a),QT.tokenize(b)),
            J.get_raw_score(at,bt),CO.get_raw_score(at,bt),LE.get_sim_score(a,b),
            JW.get_raw_score(a,b),ME.get_raw_score(at,bt),float(a==b),0.]


def features(ds,pairs):
    rows=[]
    for p in pairs:
        r=[]
        for attr in ATTRS[ds]:
            r.extend(similarities(norm(p['record_a'].get(attr)),norm(p['record_b'].get(attr))))
        attr='year' if ds=='da' else 'price'
        try:
            a=float(p['record_a'].get(attr));b=float(p['record_b'].get(attr))
            if not np.isfinite([a,b]).all() or a<=0 or b<=0:raise ValueError('missing')
            r.extend([abs(a-b),abs(a-b)/max(a,b),0.])
        except (ValueError,TypeError):r.extend([np.nan,np.nan,1.])
        rows.append(r)
    names=[f'{a}_{s}' for a in ATTRS[ds] for s in ['qgram_jaccard','token_jaccard','cosine','levenshtein','jaro_winkler','monge_elkan_A_to_B','exact','missing']]
    names += [f'{attr}_{s}' for s in ['absolute_difference','relative_difference','missing']]
    return pd.DataFrame(rows,columns=names)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ['features','predictions','models','oof']:(OUT/name).mkdir()
    inputs={};results=[]
    for ds in ATTRS:
        pairs={};X={}
        for split,path in [('pool',BASE/f'data/pools/{ds}_pool2000.jsonl'),
                           ('test',BASE/f'data/canonical/{ds}/test.jsonl'),
                           ('full',BASE/f'data/canonical/{ds}/train.jsonl')]:
            inputs[str(path)]=sha(path)
            pairs[split]=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
            frame=features(ds,pairs[split]);X[split]=frame.to_numpy()
            frame.assign(pair_id=[p['pair_id'] for p in pairs[split]]).to_csv(OUT/f'features/{ds}_{split}.csv',index=False)
        print('AUTHOR_LIBRARY_FEATURES',ds,flush=True)
        for budget in [50,200,1000,'full']:
            for seed in range(3 if budget=='full' else 5 if budget==1000 else 10):
                split='full' if budget=='full' else 'pool';ids=[p['pair_id'] for p in pairs[split]]
                if budget=='full':idx=np.arange(len(ids))
                else:
                    path=BASE/f'data/budgets/{ds}/b{budget}_seed{seed}.json';inputs[str(path)]=sha(path)
                    wanted=json.loads(path.read_text())['pair_ids'];idx=np.flatnonzero(np.isin(ids,wanted))
                    assert len(idx)==budget
                y=np.array([p['label'] for p in pairs[split]])[idx];yt=np.array([p['label'] for p in pairs['test']])
                for best in [False,True]:
                    system='M3-author+best' if best else 'M3-author+LR'
                    model,t,oof,params=fit(X[split][idx],y,len(idx),seed,best)
                    probabilities=model.predict_proba(X['test'])[:,1]
                    slug=f'{ds}_{system}_b{budget}_s{seed}'
                    pd.DataFrame(dict(pair_id=[p['pair_id'] for p in pairs['test']],label=yt,
                        probability=probabilities,prediction=(probabilities>=t).astype(int),threshold=t)).to_csv(OUT/f'predictions/{slug}.csv',index=False)
                    joblib.dump(model,OUT/f'models/{slug}.joblib')
                    (OUT/f'models/{slug}.json').write_text(json.dumps(dict(threshold=t,**params),indent=2),encoding='utf8')
                    pd.DataFrame(dict(pair_id=np.array(ids)[idx],label=y,oof_probability=oof)).to_csv(OUT/f'oof/{slug}.csv',index=False)
                    results.append(dict(dataset=ds,system=system,budget=budget,seed=seed,
                        threshold=t,classifier=params['classifier'],**metrics(yt,probabilities,t),
                        f1_at_05=metrics(yt,probabilities,.5)['f1']))
            pd.DataFrame(results).to_csv(OUT/'metrics_by_seed.csv',index=False)
            print('AUTHOR_LIBRARY_FIT',ds,budget,flush=True)
        similarities.cache_clear()
    summary=pd.DataFrame(results).groupby(['dataset','system','budget']).agg(
        f1_mean=('f1','mean'),f1_std=('f1','std'),n_seeds=('seed','count'),f1_at_05_mean=('f1_at_05','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    ref={'wa':71.9,'ag':49.1,'da':98.4}
    check=summary[(summary.budget=='full') & (summary.dataset.isin(ref))].copy()
    check['reference']=check.dataset.map(ref);check['difference']=check.f1_mean-check.reference
    check['reference_gate']=np.where(check.difference.abs()<=5,'WITHIN_5_DESCRIPTIVE_ONLY','OUTSIDE_5_NOT_REPLICATION')
    check.to_csv(OUT/'reference_deviation.csv',index=False)
    for p in [Path(__file__),BASE/'src/recovery/magellan_v2.py',BASE/'src/recovery/core.py']:inputs[str(p)]=sha(p)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(status='CPU_FALLBACK_COMPLETE_NOT_OFFICIAL_MAGELLAN',n_groups=len(results),paid_api_calls=0,gpu_calls=0,
        package_versions={m:__import__(m).__version__ for m in ['numpy','pandas','sklearn']},inputs=inputs,
        feature_definition='py_stringmatching 0.4.7: padded3gram Jaccard, alphanumeric-token Jaccard/Cosine, normalized Levenshtein, JaroWinkler, directed MongeElkan A-to-B, exact, missing; numeric differences',
        protocol='New post-audit baseline; use existing frozen budget IDs and training-only CV/OOF; no reference-matching parameter search',
        limitations=['Official py_entitymatching install failed: Microsoft Visual C++ build tools absent.',
                     'Authors library feature family does not prove exact Narayan feature/selection protocol reproduced.',
                     'All reference gaps retained, not forced within five points.',
                     'Keep new baseline separate from previously accepted main results.'])
    manifest['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('AUTHOR_LIBRARY_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
