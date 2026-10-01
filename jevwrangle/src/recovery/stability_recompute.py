"""Reconstruct historical perturbation responses; new P2/P5 are real requests."""
import hashlib,json,sqlite3
from pathlib import Path
import joblib,numpy as np,pandas as pd,yaml
from sklearn.metrics import cohen_kappa_score
from src.experiments.e5_stability import make_perturbed_requests
from .core import noul,choice,logit,metrics
from .offline import POLARITIES
BASE=Path(__file__).resolve().parents[2]

def main(out):
    out.mkdir(exist_ok=False);(out/'predictions').mkdir();(out/'features').mkdir()
    hashes={}
    def track(path):
        hashes[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest();return path
    track(BASE/'cache/calls.sqlite')
    db=sqlite3.connect((BASE/'cache/calls.sqlite').as_uri()+'?mode=ro',uri=True)
    new={}
    journal=BASE/'runs/overnight_api_20260930/parsed.jsonl'
    if journal.exists():
        for line in track(journal).read_text(encoding='utf8').splitlines():
            r=json.loads(line)
            if r['experiment']=='E5' and r['status']=='ok':new[(r['dataset'],r['pair_id'],r['kind'])]=r
    repairs=BASE/'runs/stability_api_repairs_20260930/parsed.jsonl';extra={}
    if repairs.exists():
        for line in track(repairs).read_text(encoding='utf8').splitlines():
            r=json.loads(line)
            if r['status']=='ok':extra[(r['dataset'],r['pair_id'],r['kind'])]=r
    results=[];gates=[];missing=[]
    for ds in ['wa','ag','da','ab']:
        test=[json.loads(s) for s in track(BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()]
        qh=yaml.safe_load(track(BASE/f'questions/{ds}/holistic.yaml').read_text(encoding='utf8'));qd=yaml.safe_load(track(BASE/f'questions/{ds}/decomposed.yaml').read_text(encoding='utf8'))
        paraphrase=yaml.safe_load(track(BASE/f'questions/{ds}/paraphrase.yaml').read_text(encoding='utf8'))
        clean_h=pd.read_csv(track(BASE/f'runs/recovery_repaired_20260930/features/{ds}_test_H.csv')).set_index('pair_id')
        clean_d=pd.read_csv(track(BASE/f'runs/recovery_repaired_20260930/features/{ds}_test_D.csv')).set_index('pair_id')
        for variant in ['P1','P2','P3','P4','P5','P6','P7']:
            features=[]
            for pair in test:
                key=None;alias=None
                fixed=extra.get((ds,pair['pair_id'],variant))
                if fixed:
                    features.append(dict(pair_id=pair['pair_id'],label=pair['label'],cache_key=fixed['meta']['key'],request_alias=fixed['model'],**fixed['values']))
                    continue
                if variant in ['P2','P5']:
                    item=new.get((ds,pair['pair_id'],variant))
                    if not item:
                        missing.append(dict(dataset=ds,pair_id=pair['pair_id'],kind=variant));continue
                    values=item['values'];key=item['meta']['key'];alias=item['model']
                else:
                    state,q=make_perturbed_requests(pair,qh,qd,paraphrase,ds)[variant]
                    response=None
                    for alias in ['jev-1.13','jev-latest']:
                        payload={'model':alias,'state':state,'questions':q}
                        key=hashlib.sha256(json.dumps(['aihubmix',payload],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        row=db.execute('SELECT response FROM calls WHERE key=?',(key,)).fetchone()
                        if row:response=json.loads(row[0]);break
                    if response is None:
                        missing.append(dict(dataset=ds,pair_id=pair['pair_id'],kind=variant));continue
                    try:values={qid:noul(response['answers'][qid]) for qid in ['H1_noul',*qd]}
                    except (KeyError,ValueError,TypeError):
                        missing.append(dict(dataset=ds,pair_id=pair['pair_id'],kind=variant));continue
                features.append(dict(pair_id=pair['pair_id'],label=pair['label'],cache_key=key,request_alias=alias,**values))
            frame=pd.DataFrame(features);gates.append(dict(dataset=ds,variant=variant,n_expected=len(test),n_available=len(frame),
                status='COMPLETE' if len(frame)==len(test) else 'BLOCKED_MISSING'))
            frame.to_csv(out/f'features/{ds}_{variant}.csv',index=False)
            if len(frame)!=len(test):continue
            frame=frame.set_index('pair_id').loc[[p['pair_id'] for p in test]];y=frame.label.values
            systems=[('J-H1c',0,0)] if variant=='P5' else [('J-H1n',0,0),('J-D+LR',200,0),('J-D+LR','full',0)]
            for system,budget,seed in systems:
                if system=='J-H1c':dirty=frame.H1_choice.values;clean=clean_h.loc[frame.index,'p_h1c'].values;threshold=.5
                elif system=='J-H1n':dirty=frame.H1_noul.values;clean=clean_h.loc[frame.index,'p_h1n'].values;threshold=.5
                else:
                    slug=f'{ds}_{system}_b{budget}_s{seed}'
                    model=joblib.load(track(BASE/f'runs/recovery_repaired_20260930/models/{slug}.joblib'))
                    params=json.loads(track(BASE/f'runs/recovery_repaired_20260930/models/{slug}.json').read_text(encoding='utf8'))
                    columns=list(POLARITIES[ds])
                    dirty=model.predict_proba(logit(frame[columns].values))[:,1]
                    clean=model.predict_proba(logit(clean_d.loc[frame.index,['jev_'+c for c in columns]].values))[:,1];threshold=params['threshold']
                pred=dirty>=threshold;base_pred=clean>=threshold
                pd.DataFrame(dict(pair_id=frame.index,label=y,clean_probability=clean,probability=dirty,
                    prediction=pred.astype(int),threshold=threshold)).to_csv(out/f'predictions/{ds}_{system}_b{budget}_{variant}.csv',index=False)
                results.append(dict(dataset=ds,variant=variant,system=system,budget=budget,seed=seed,n=len(y),
                    flip_rate=float(np.mean(pred!=base_pred)),mean_abs_delta_probability=float(np.mean(abs(dirty-clean))),
                    delta_f1=metrics(y,dirty,threshold)['f1']-metrics(y,clean,threshold)['f1'],
                    kappa=float(cohen_kappa_score(base_pred,pred)),f1=metrics(y,dirty,threshold)['f1']))
    pd.DataFrame(gates).to_csv(out/'completeness_gates.csv',index=False)
    (out/'missing.json').write_text(json.dumps(missing,indent=2))
    pd.DataFrame(results).to_csv(out/'stability.csv',index=False)
    db.close()
    for name,path in [('executed_source.py',Path(__file__)),('core_source.py',BASE/'src/recovery/core.py'),
                      ('perturbation_source.py',BASE/'src/experiments/e5_stability.py'),('feature_polarities.py',BASE/'src/experiments/e2_attribution.py')]:
        track(path);(out/name).write_bytes(path.read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(status='UNFROZEN_COMPLETE_GROUPS_ONLY',gates=gates,
        input_sha256=hashes,encoding='UTF-8 explicitly, independent of Windows locale',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Historical P3/P4 holistic paraphrases generated by legacy code, not frozen YAML; DA templates mention irrelevant product variants',
                    'Old sorted-key cache cannot validate attribute/option order, so P2/P5 use new actual order-sensitive calls',
                    'No determinism claim from cache repeats; repeated raw requests still require evidence'],
        outputs={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}),indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output.resolve())
