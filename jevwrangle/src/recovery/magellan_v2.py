"""Complete the declared Magellan-style fallback and four-matcher selection.

Separate from legacy feature results: this is a post-audit implementation repair.
Missing similarities are imputed within each training fold, never on the test set.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold,cross_val_score
from threadpoolctl import threadpool_limits
from src.experiments.magellan_feats import compute_magellan_features_for_pair
from .core import best_oof_threshold,metrics
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/magellan_v2_20260930'


def tokens(value):return re.findall(r'\w+',str(value or '').lower())


def cosine(a,b):
    a,b=Counter(a),Counter(b)
    denominator=np.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
    return sum(v*b[k] for k,v in a.items())/denominator if denominator else np.nan


def monge_elkan(a,b):
    if not a or not b:return np.nan
    matrix=process.cdist(a,b,scorer=JaroWinkler.normalized_similarity)
    return float((matrix.max(axis=1).mean()+matrix.max(axis=0).mean())/2)


def features(ds,pairs):
    attrs={'wa':['title','category','brand','modelno'],'ag':['title','manufacturer'],
           'da':['title','authors','venue'],'ab':['name','description']}[ds]
    rows=[]
    for p in pairs:
        a,b=p['record_a'],p['record_b'];r=compute_magellan_features_for_pair(a,b,ds)
        for attr in attrs:
            ta,tb=tokens(a.get(attr)),tokens(b.get(attr))
            r[f'm_{attr}_cosine']=cosine(ta,tb)
            r[f'm_{attr}_monge_elkan_symmetric']=monge_elkan(ta,tb)
            if not ta or not tb:
                for col in list(r):
                    if col.startswith('m_'+attr+'_') and not col.endswith('_missing'):r[col]=np.nan
        for attr in (['year'] if ds=='da' else ['price']):
            if r[f'm_{attr}_missing']:
                r[f'm_{attr}_abs_diff']=np.nan;r[f'm_{attr}_rel_diff']=np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def make_model(kind,seed,C=1.):
    choices={'LR':LogisticRegression(C=C,max_iter=2000,random_state=seed),
             'RF':RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1),
             'DT':DecisionTreeClassifier(random_state=seed),
             'SVM':SVC(C=1.,kernel='rbf',probability=True,random_state=seed)}
    return make_pipeline(SimpleImputer(strategy='mean',keep_empty_features=True),StandardScaler(),choices[kind])


@threadpool_limits.wrap(limits=1)
def fit(X,y,budget,seed,best):
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int);k=min(5,int(np.bincount(y,minlength=2).min()))
    if k<2:raise ValueError('Too few minority labels for budget CV')
    cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed);C=1.;cvC={}
    if budget>=1000:
        for c in [.01,.1,1.,10.]:
            cvC[c]=float(cross_val_score(make_model('LR',seed,c),X,y,cv=cv,scoring='neg_log_loss').mean())
        C=max(cvC,key=cvC.get)
    kind='LR';scores={}
    if best:
        for candidate in ['LR','RF','DT','SVM']:
            scores[candidate]=float(cross_val_score(make_model(candidate,seed,C),X,y,cv=cv,scoring='f1').mean())
        kind=max(scores,key=scores.get)
    oof=np.zeros(len(y));count=np.zeros(len(y))
    for tr,va in RepeatedStratifiedKFold(n_splits=k,n_repeats=3,random_state=seed).split(X,y):
        m=make_model(kind,seed,C).fit(X[tr],y[tr]);oof[va]+=m.predict_proba(X[va])[:,1];count[va]+=1
    oof/=count;threshold=best_oof_threshold(y,oof);m=make_model(kind,seed,C).fit(X,y)
    return m,threshold,oof,dict(classifier=kind,C=C,cv_F1=scores,cv_neg_log_loss=cvC,folds=k)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ['features','predictions','models','oof']:(OUT/name).mkdir()
    rows=[];inputs={}
    for ds in ['wa','ag','da','ab']:
        pairs={};X={}
        for split,path in [('pool',BASE/f'data/pools/{ds}_pool2000.jsonl'),('test',BASE/f'data/canonical/{ds}/test.jsonl'),('full',BASE/f'data/canonical/{ds}/train.jsonl')]:
            inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            pairs[split]=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
            f=features(ds,pairs[split]);X[split]=f.values
            f.assign(pair_id=[p['pair_id'] for p in pairs[split]]).to_csv(OUT/f'features/{ds}_{split}.csv',index=False)
        print('MAGELLAN_V2 features',ds,flush=True)
        for budget in [50,200,1000,'full']:
            for seed in range(3 if budget=='full' else 5 if budget==1000 else 10):
                split='full' if budget=='full' else 'pool';prs=pairs[split];ids=[p['pair_id'] for p in prs]
                if budget=='full':idx=np.arange(len(prs))
                else:
                    path=BASE/f'data/budgets/{ds}/b{budget}_seed{seed}.json';inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
                    wanted=json.loads(path.read_text())['pair_ids'];idx=np.flatnonzero(np.isin(ids,wanted));assert len(idx)==budget
                y=np.array([p['label'] for p in prs])[idx];yt=np.array([p['label'] for p in pairs['test']])
                for best in [False,True]:
                    system='M2+best' if best else 'M2+LR';m,t,oof,params=fit(X[split][idx],y,len(idx),seed,best)
                    p=m.predict_proba(X['test'])[:,1];slug=f'{ds}_{system}_b{budget}_s{seed}'
                    pd.DataFrame(dict(pair_id=[r['pair_id'] for r in pairs['test']],label=yt,probability=p,prediction=(p>=t).astype(int),threshold=t)).to_csv(OUT/f'predictions/{slug}.csv',index=False)
                    joblib.dump(m,OUT/f'models/{slug}.joblib')
                    (OUT/f'models/{slug}.json').write_text(json.dumps(dict(threshold=t,**params),indent=2),encoding='utf8')
                    pd.DataFrame(dict(pair_id=np.array(ids)[idx],label=y,oof_probability=oof)).to_csv(OUT/f'oof/{slug}.csv',index=False)
                    rows.append(dict(dataset=ds,system=system,budget=budget,seed=seed,threshold=t,classifier=params['classifier'],
                                     **metrics(yt,p,t),f1_at_05=metrics(yt,p,.5)['f1']))
            print('MAGELLAN_V2',ds,'budget',budget,'completed',flush=True)
            pd.DataFrame(rows).to_csv(OUT/'metrics_by_seed.csv',index=False)
        pd.DataFrame(rows).groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),n_seeds=('seed','count'),f1_at_05_mean=('f1_at_05','mean')).reset_index().to_csv(OUT/'summary.csv',index=False)
    manifest=dict(status='POST_AUDIT_REPAIRED_BASELINE_NOT_FROZEN',inputs=inputs,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Magellan-style fallback, not py_entitymatching reproduction; v2 adds cosine/Monge-Elkan and foldwise missing-value imputation',
                  selection='Budget-only CV F1@0.5 chooses LR/RF/DT/SVM; repeated OOF selects final threshold; LR C uses budget-only neg-log-loss')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('MAGELLAN_V2_DONE',OUT,flush=True)


if __name__=='__main__':main()
