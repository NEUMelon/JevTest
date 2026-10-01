"""Auditable Flights row budgets; row-grouped OOF prevents cell leakage.

Sampling is frozen before Qwen ED outputs exist. Hospital/Adult supervised
claims are deliberately not imported into this disjoint-row experiment.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,yaml
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from .core import logit,metrics,best_oof_threshold
BASE=Path(__file__).resolve().parents[2]
SAMPLING=BASE/'runs/flights_frozen_row_budgets_20260930'

def freeze():
    if SAMPLING.exists():return
    SAMPLING.mkdir();source=BASE/'data/canonical/fl/train.jsonl'
    rows=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    groups=sorted(set(r['row_id'] for r in rows));cols=set(r['col'] for r in rows)
    for row in groups:
        if {r['col'] for r in rows if r['row_id']==row}!=cols:raise ValueError('Incomplete annotated row')
    jobs=[]
    for budget in [5,20,50,'full']:
        for seed in range(5):
            ids=groups if budget=='full' else sorted(np.random.default_rng(20260929+seed).choice(groups,size=budget,replace=False).tolist())
            jobs.append(dict(budget=budget,seed=seed,row_ids=ids))
    (SAMPLING/'sampling.json').write_text(json.dumps(dict(jobs=jobs,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selection='Uniform rows; RNG 20260929 + seed; all six cell labels for each selected row',
        protocol='Post-audit recovery; row-grouped train-only OOF and threshold; no test tuning',
        created_before_Qwen_ED_results=True),indent=2),encoding='utf8')

@threadpool_limits.wrap(limits=1)
def fit_grouped(X,y,groups,seed):
    X,y,groups=np.asarray(X),np.asarray(y),np.asarray(groups)
    if len(np.unique(y))!=2:raise ValueError('Selected rows lack a class; do not substitute another system')
    k=min(5,len(np.unique(groups)))
    cv=[]
    for repeat in range(3):
        folds=list(StratifiedGroupKFold(k,shuffle=True,random_state=seed+repeat*100).split(X,y,groups))
        for tr,va in folds:
            if set(groups[tr])&set(groups[va]):raise ValueError('Row crossed OOF boundary')
            if len(np.unique(y[tr]))<2:raise ValueError('Grouped OOF training lacks a class')
        cv.extend(folds)
    def model(C):return make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=seed))
    losses={};C=1.
    if len(y)>=1000:
        for c in [.01,.1,1.,10.]:
            losses[c]=float(np.mean([log_loss(y[va],model(c).fit(X[tr],y[tr]).predict_proba(X[va])[:,1],labels=[0,1]) for tr,va in cv[:k]]))
        C=min(losses,key=losses.get)
    oof=np.zeros(len(y));visits=np.zeros(len(y),int)
    for tr,va in cv:oof[va]+=model(C).fit(X[tr],y[tr]).predict_proba(X[va])[:,1];visits[va]+=1
    if not np.all(visits==3):raise ValueError('Missing grouped OOF predictions')
    oof/=visits;threshold=best_oof_threshold(y,oof)
    return model(C).fit(X,y),threshold,oof,dict(C=C,folds=k,repeats=3,cv_log_loss=losses)

def main(out):
    freeze();out.mkdir(exist_ok=False)
    for d in ['predictions','models','oof','features']:(out/d).mkdir()
    inputs={};rows=[];gates=[]
    def track(p):inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    sampling=json.loads(track(SAMPLING/'sampling.json').read_text(encoding='utf8'))
    canonical={};groups={};labels={}
    for split in ['train','test']:
        p=track(BASE/f'data/canonical/fl/{split}.jsonl');canonical[split]=[json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
        frame=pd.DataFrame(canonical[split]).set_index('cell_id',verify_integrity=True);groups[split]=frame.row_id;labels[split]=frame.label_error
    if set(groups['train'])&set(groups['test']):raise ValueError('Flights row leakage')
    if sampling['source_sha256']!=inputs['data/canonical/fl/train.jsonl']:raise ValueError('Frozen row pool changed')
    qpath=track(BASE/'runs/gpu_backup_20260930/qwen_recovery_20260930/diagnostics.jsonl')
    q=[r for s in qpath.read_text(encoding='utf8').splitlines() if (r:=json.loads(s))['experiment']=='E7' and r['dataset']=='fl']
    qdefs=yaml.safe_load(track(BASE/'questions/fl/decomposed.yaml').read_text(encoding='utf8'));qcols=[x for x in qdefs if x!='H_err']
    features={}
    for split in ['train','test']:
        g=pd.DataFrame([r for r in q if r['split']==split]);f=g.pivot(index='pair_id',columns='qid',values='p_yes').loc[labels[split].index]
        if f.isna().any().any() or not np.array_equal(g.groupby('pair_id').label.first().loc[f.index],labels[split]):raise ValueError('Qwen ED coverage/labels invalid')
        features['Q',split]=f
        jf=pd.read_csv(track(BASE/f'runs/ed_recovery_20260930/features/fl_{split}.csv')).set_index('cell_id',verify_integrity=True).loc[f.index]
        if (jf.status!='verified_cache').any() or not np.array_equal(jf.label,labels[split]):raise ValueError('Jev ED missing raw evidence')
        features['J',split]=jf
        f.assign(label=labels[split],row_id=groups[split]).to_csv(out/f'features/fl_Q_{split}.csv')
    for short,cols,hcol in [('Q',qcols,'H_err'),('J',['p_d_typo','p_d_format','p_d_cons','p_d_place','p_d_imp'],'p_h_err')]:
        pool,test=features[short,'train'],features[short,'test'];truth=labels['test'].values
        for system,p in [(short+'-H_err',test[hcol].values),(short+'-D0',test[cols].values.mean(1))]:
            rows.append(dict(dataset='fl',system=system,budget=0,seed=0,n_rows=0,n_cells=0,threshold=.5,**metrics(truth,p)))
        for j in sampling['jobs']:
            idx=np.flatnonzero(groups['train'].isin(j['row_ids']).values);y=labels['train'].values[idx];grp=groups['train'].values[idx]
            for kind,selected in [('H+LR',[hcol]),('D+LR',cols)]:
                system=short+'-'+kind;slug=f'fl_{system}_r{j["budget"]}_s{j["seed"]}'
                try:model,t,oof,params=fit_grouped(logit(pool[selected].values[idx]),y,grp,j['seed'])
                except ValueError as error:
                    gates.append(dict(system=system,budget=j['budget'],seed=j['seed'],status='BLOCKED_INVALID_GROUPED_OOF',reason=str(error)));continue
                p=model.predict_proba(logit(test[selected].values))[:,1]
                pd.DataFrame(dict(cell_id=test.index,label=truth,probability=p,prediction=(p>=t).astype(int),threshold=t)).to_csv(out/f'predictions/{slug}.csv',index=False)
                joblib.dump(model,out/f'models/{slug}.joblib')
                (out/f'models/{slug}.json').write_text(json.dumps(dict(threshold=t,columns=selected,**params),indent=2))
                pd.DataFrame(dict(cell_id=pool.index[idx],row_id=grp,label=y,oof_probability=oof)).to_csv(out/f'oof/{slug}.csv',index=False)
                gates.append(dict(system=system,budget=j['budget'],seed=j['seed'],status='COMPLETE'))
                rows.append(dict(dataset='fl',system=system,budget=j['budget'],seed=j['seed'],n_rows=len(j['row_ids']),n_cells=len(idx),threshold=t,**metrics(truth,p,t)))
    pd.DataFrame(rows).to_csv(out/'metrics_by_seed.csv',index=False)
    pd.DataFrame(rows).groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),n_seeds=('seed','count')).reset_index().to_csv(out/'summary.csv',index=False)
    pd.DataFrame(gates).to_csv(out/'completeness_gates.csv',index=False)
    track(Path(__file__));track(BASE/'src/recovery/core.py')
    (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_GROUPS_ONLY_NOT_FROZEN',inputs=inputs,
        caveats=['Jev legacy state omitted column profiles; Qwen frozen state includes profiles, so this is not an identical-prompt causal comparison',
            'Flights labels mark dirty/clean string differences, which can include time-format normalization',
            'Hospital sparse annotation and Adult repeated splits excluded from supervised row-budget claims',
            'Canonical Flights split has six columns; literature Raha protocol differs; no SOTA claim'],
        outputs={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
    if a.freeze_only:freeze()
    else:main(a.output.resolve())
