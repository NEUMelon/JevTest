"""CPU-only post-collection diagnostics with frozen models and paired IDs.

Never refit an aggregator on perturbations/attacks or fit temperatures on test.
Outputs remain descriptive; this does not declare the preregistration frozen.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,yaml
from sklearn.metrics import cohen_kappa_score
from threadpoolctl import threadpool_limits
from .core import logit,metrics
from .calibration_ci import metric_batch,temperature_fit,transformed
from .selective import threshold_intervals
from .statistics import paired_bootstrap
BASE=Path(__file__).resolve().parents[2]

def binary_rate_ci(event,eligible=None,replicates=2000):
    event=np.asarray(event,dtype=bool)
    eligible=np.ones(len(event),bool) if eligible is None else np.asarray(eligible,dtype=bool)
    counts=np.array([np.sum(event&eligible),np.sum(~event&eligible),np.sum(~eligible)])
    if not eligible.any():return dict(estimate=None,ci_low=None,ci_high=None,n_eligible=0,n_events=0)
    draws=np.random.default_rng(20260930).multinomial(len(event),counts/len(event),size=replicates)
    denom=draws[:,:2].sum(1);values=draws[denom>0,0]/denom[denom>0]
    return dict(estimate=float(event[eligible].mean()),ci_low=float(np.percentile(values,2.5)),
        ci_high=float(np.percentile(values,97.5)),n_eligible=int(eligible.sum()),n_events=int(event[eligible].sum()))

@threadpool_limits.wrap(limits=1)
def main(out):
    out.mkdir(exist_ok=False);(out/'predictions').mkdir();(out/'curves').mkdir();inputs={}
    def track(path):
        inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest();return path
    root=BASE/'runs/gpu_backup_20260930/qwen_recovery_20260930'
    evaluation=BASE/'runs/overnight_synthesis_20260930/qwen_evaluation'
    manifest=json.loads(track(root/'manifest.json').read_text(encoding='utf8'))
    for name in ['diagnostics.jsonl','parsed.jsonl']:
        if hashlib.sha256(track(root/name).read_bytes()).hexdigest()!=manifest['outputs'][name]:raise ValueError('Qwen backup mismatch')
    records=[json.loads(s) for s in (root/'diagnostics.jsonl').read_text(encoding='utf8').splitlines()]
    diagnostic=pd.DataFrame(records)
    if diagnostic.duplicated(['experiment','dataset','split','pair_id','qid','perturbation']).any():raise ValueError('Duplicate diagnostics')
    calibration=[];temps=[];selective=[];stability=[];probes=[];ed=[];attacks=[];paired_attacks=[];mass=[]
    # All model calibration comparisons carry their exact sample protocol.
    for ds in ['wa','ag','da','ab']:
        for short,folder,field in [('J',BASE/'runs/recovery_repaired_20260930','p_h1n'),
                                  ('Q',evaluation,'p_yes'),
                                  ('L1',BASE/'runs/overnight_finalization_20260930/llm_evaluation','p_yes'),
                                  ('L2',BASE/'runs/overnight_finalization_20260930/llm_evaluation','p_yes')]:
            paths=[folder/f'features/{ds}_{split}_H.csv' if short=='J' else folder/f'features/{ds}_{short}_{split}_H.csv' for split in ['pool','test']]
            if not all(p.exists() for p in paths):continue
            pool,test=[pd.read_csv(track(p)) for p in paths]
            values=pool[[field]].values;y=pool.label.values
            T,ok,reason=temperature_fit(values,y,0)
            if not ok:raise RuntimeError('Temperature fit failed: '+reason)
            unique,counts=np.unique(np.column_stack([values,y]),axis=0,return_counts=True)
            rng=np.random.default_rng(20260930);sampled=[];failures=[]
            for k in range(2000):
                t,success,message=temperature_fit(unique[:,:-1],unique[:,-1],0,rng.multinomial(len(y),counts/counts.sum()))
                if success:sampled.append(t)
                else:failures.append(dict(replicate=k,T=t,message=message))
            (out/f'{ds}_{short}_temperature_failures.json').write_text(json.dumps(failures,indent=2))
            temps.append(dict(dataset=ds,system=short+'-H1',T=T,T_ci_low=float(np.percentile(sampled,2.5)) if sampled else None,
                T_ci_high=float(np.percentile(sampled,97.5)) if sampled else None,n_success=len(sampled),n_failed=len(failures),n_pool=len(pool),n_test=len(test),
                protocol='Fixed 500 stratified test pairs; union of b50/b200 three-seed training IDs' if short=='L1' else 'Full canonical test and pool2000'))
            for mode,p in [('raw',test[field].values),('temperature',transformed(test[[field]].values,0,T))]:
                point=metric_batch(test.label.values,p);samples={k:[] for k in point};rng=np.random.default_rng(20260930)
                for start in range(0,2000,100):
                    idx=rng.integers(len(test),size=(100,len(test)));m=metric_batch(test.label.values[idx],p[idx])
                    for k,v in m.items():samples[k].extend(v)
                for k,v in point.items():calibration.append(dict(dataset=ds,system=short+'-H1',mode=mode,metric=k,
                    estimate=float(v[0]),ci_low=float(np.percentile(samples[k],2.5)),ci_high=float(np.percentile(samples[k],97.5)),
                    n_test=len(test),T=T,interval='Test bootstrap conditional on train-fitted T'))
            candidates=[(short+'-H1',0,test.label.values,test[field].values,test[field].values>=.5)]
            for system in [short+'-H1+LR',short+'-D+LR']:
                p=folder/f'predictions/{ds}_{system}_b200_s0.csv'
                if p.exists():
                    f=pd.read_csv(track(p));candidates.append((system,200,f.label.values,f.probability.values,f.prediction.values))
            for system,budget,truth,p,pred in candidates:
                confidence=np.where(pred==1,p,1-p);correct=pred==truth;slug=f'{ds}_{system}_b{budget}_s0'
                for threshold in [.95,.99]:selective.append(dict(dataset=ds,system=system,budget=budget,seed=0,threshold=threshold,
                    n_test=len(p),**threshold_intervals(correct,confidence,threshold)))
                order=np.argsort(-confidence,kind='stable');ends=np.flatnonzero(np.r_[confidence[order][:-1]!=confidence[order][1:],True]);n=ends+1
                pd.DataFrame(dict(confidence=confidence[order][ends],n_selected=n,coverage=n/len(p),
                    risk=np.cumsum(~correct[order])[ends]/n)).to_csv(out/f'curves/{slug}.csv',index=False)
                if budget:
                    for k,v in metric_batch(truth,p).items():calibration.append(dict(dataset=ds,system=system,mode='LR_b200_seed0',metric=k,estimate=float(v[0]),n_test=len(p)))
        # Freeze clean b200/seed0 aggregation for all perturbations/attacks.
        testH=pd.read_csv(track(evaluation/f'features/{ds}_Q_test_H.csv')).set_index('pair_id',verify_integrity=True)
        testD=pd.read_csv(track(evaluation/f'features/{ds}_Q_test_D.csv')).set_index('pair_id',verify_integrity=True)
        qdefs=yaml.safe_load(track(BASE/f'questions/{ds}/decomposed.yaml').read_text(encoding='utf8'));cols=list(qdefs)
        slug=f'{ds}_Q-D+LR_b200_s0';model=joblib.load(track(evaluation/f'models/{slug}.joblib'))
        threshold=json.loads(track(evaluation/f'models/{slug}.json').read_text(encoding='utf8'))['threshold']
        def decision(frame,system):
            p=frame['p_yes'].values if system=='Q-H1' else model.predict_proba(logit(frame[cols].values))[:,1]
            t=.5 if system=='Q-H1' else threshold
            return p,(p>=t).astype(int),t
        def clean_for(ids):
            h=testH.loc[ids,['label','p_yes']];d=testD.loc[ids,cols]
            if not np.array_equal(testD.loc[ids,'label'],h.label):raise ValueError('Clean labels mismatch')
            return h.join(d)
        for variant,g in diagnostic[(diagnostic.dataset==ds)&(diagnostic.experiment=='E5')].groupby('perturbation'):
            frame=g.pivot(index='pair_id',columns='qid',values='p_yes');frame['label']=g.groupby('pair_id').label.first()
            ids=json.loads(track(root/f'{ds}_stability_ids.json').read_text(encoding='utf8'));frame=frame.loc[ids];clean=clean_for(ids)
            if frame[cols+['p_yes']].isna().any().any() or not np.array_equal(frame.label,clean.label):raise ValueError('E5 incomplete')
            for system in ['Q-H1','Q-D+LR']:
                p,pred,t=decision(frame,system);basep,basepred,_=decision(clean,system);truth=clean.label.values
                delta=paired_bootstrap([(truth,basepred,pred)])
                stability.append(dict(dataset=ds,system=system,variant=variant,n_test=len(ids),flip_rate=float((basepred!=pred).mean()),
                    flip_ci=binary_rate_ci(basepred!=pred),delta_f1=delta['delta_B_minus_A'],ci_low=delta['delta_ci_low'],ci_high=delta['delta_ci_high'],
                    kappa=float(cohen_kappa_score(basepred,pred)),mean_abs_delta_probability=float(np.abs(p-basep).mean()),
                    note='Holistic prompt unchanged for P3/P4; P5 not applicable to Noul'))
                pd.DataFrame(dict(pair_id=ids,label=truth,probability=p,prediction=pred,clean_prediction=basepred,threshold=t)).to_csv(out/f'predictions/{ds}_{system}_{variant}.csv',index=False)
        for (variant,experiment),g in diagnostic[(diagnostic.dataset==ds)&diagnostic.experiment.isin(['E10','E10_defense'])].groupby(['perturbation','experiment']):
            frame=g.pivot(index='pair_id',columns='qid',values='p_yes');ids=frame.index;clean=clean_for(ids);truth=clean.label.values
            if not np.array_equal(g.groupby('pair_id').label.first().loc[ids],truth):raise ValueError('Attack labels mismatch')
            systems=['Q-H1'] if experiment=='E10_defense' else ['Q-H1','Q-D+LR'];results={}
            for system in systems:
                if frame[['p_yes'] if system=='Q-H1' else cols].isna().any().any():raise ValueError('Attack missing')
                p,pred,t=decision(frame,system);basep,basepred,_=decision(clean,system)
                eligible=basepred==truth;event=pred==1-truth;results[system]=(eligible,event)
                attacks.append(dict(dataset=ds,system=system,variant=variant,experiment=experiment,n_test=len(ids),
                    **binary_rate_ci(event,eligible),denominator='Clean-correct cases for this system'))
                pd.DataFrame(dict(pair_id=ids,label=truth,probability=p,prediction=pred,clean_prediction=basepred,threshold=t)).to_csv(out/f'predictions/{ds}_{system}_{experiment}_{variant}.csv',index=False)
            if len(results)==2:
                common=results['Q-H1'][0]&results['Q-D+LR'][0];eh=results['Q-H1'][1];edirty=results['Q-D+LR'][1]
                for system,event in [('Q-H1',eh),('Q-D+LR',edirty)]:attacks.append(dict(dataset=ds,system=system,variant=variant,experiment=experiment,n_test=len(ids),
                    **binary_rate_ci(event,common),denominator='Common clean-correct H AND D cases'))
                if common.any():
                    states=eh[common].astype(int)*2+edirty[common].astype(int);counts=np.bincount(states,minlength=4)
                    draws=np.random.default_rng(20260930).multinomial(len(states),counts/len(states),size=2000)
                    diff=(draws[:,1]-draws[:,2])/len(states)
                    paired_attacks.append(dict(dataset=ds,variant=variant,system_A='Q-H1',system_B='Q-D+LR',
                        delta_ASR_B_minus_A=float(edirty[common].mean()-eh[common].mean()),ci_low=float(np.percentile(diff,2.5)),
                        ci_high=float(np.percentile(diff,97.5)),n_common_correct=len(states),condition='Both systems correct on clean input'))
    for (ds,qid),g in diagnostic[diagnostic.experiment=='E9'].groupby(['dataset','qid']):
        if set(g.label)!={0,1} or g.label.value_counts().nunique()!=1:raise ValueError('Probe classes not balanced')
        probes.append(dict(dataset=ds,probe=qid,n=len(g),accuracy=float(((g.p_yes>=.5)==g.label).mean()),
            **metrics(g.label,g.p_yes),**{k:float(v[0]) for k,v in metric_batch(g.label.values,g.p_yes.values).items()},
            gold='Frozen programmatic numeric/code comparison; not human semantic matching labels'))
    for (ds,split,qid),g in diagnostic[diagnostic.experiment=='E7'].groupby(['dataset','split','qid']):
        canonical=[json.loads(s) for s in track(BASE/f'data/canonical/{ds}/{split}.jsonl').read_text(encoding='utf8').splitlines()]
        expected={r['cell_id']:r['label_error'] for r in canonical};actual=dict(zip(g.pair_id,g.label))
        if actual!=expected:raise ValueError('E7 canonical coverage/labels mismatch')
        if split=='test' and qid in ['H_err','H0_hospital']:ed.append(dict(dataset=ds,split=split,system='Q-'+qid,n=len(g),
            **metrics(g.label,g.p_yes),**{k:float(v[0]) for k,v in metric_batch(g.label.values,g.p_yes.values).items()},
            protocol='Frozen cell annotations; no fully annotated-row supervised claim, no matched literature SOTA claim'))
    for (experiment,ds,qid),g in diagnostic.groupby(['experiment','dataset','qid']):mass.append(dict(experiment=experiment,dataset=ds,qid=qid,n=len(g),
        fraction_mass_below_half=float((g.yes_no_mass<.5).mean()),mean_mass=float(g.yes_no_mass.mean())))
    ditto_path=root/'ditto_injection_predictions.csv'
    if ditto_path.exists():
        f=pd.read_csv(track(ditto_path))
        for (ds,variant),g in f[f.variant!='clean'].groupby(['dataset','variant']):
            clean=f[(f.dataset==ds)&(f.variant=='clean')].set_index('pair_id');g=g.set_index('pair_id').loc[clean.index]
            if not np.array_equal(g.label,clean.label):raise ValueError('Ditto injected labels mismatch')
            attacks.append(dict(dataset=ds,system='Ditto-full',variant=variant,experiment='E10',n_test=len(g),
                **binary_rate_ci(g.prediction.values==1-g.label.values,clean.prediction.values==clean.label.values),
                denominator='Clean-correct cases for this system',fraction_truncated=float(g.truncated.mean()),
                limitation='Appended attacks may be truncated at 256 tokens'))
    from .code_injection import collect as code_attacks
    attacks.extend(code_attacks(out/'code_injection',binary_rate_ci))
    track(out/'code_injection/manifest.json')
    for name,rows in [('calibration',calibration),('temperature',temps),('selective_thresholds',selective),('stability',stability),
                      ('numeric_probes',probes),('error_detection',ed),('attacks',attacks),('paired_attack_differences',paired_attacks),('yes_no_mass',mass)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    track(Path(__file__))
    for name in ['core.py','calibration_ci.py','statistics.py','selective.py']:track(BASE/'src/recovery'/name)
    (out/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTICS_COMPLETE_NOT_FROZEN',input_sha256=inputs,
        selection='Frozen b200 seed0 clean aggregators; no refitting on attacks or perturbations; temperatures fitted using pool labels only',
        limitations=['Token P(Yes | Yes or No) is conditional; low unconditional mass is reported',
            'No Adult supervised reuse','No invented human E8/E12 verification','H5 semantic human gold absent',
            'All intervals descriptive; two-way seed bootstrap used for multi-seed comparisons elsewhere'],
        outputs={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output.resolve())
