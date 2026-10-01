"""Independent ID/confusion checks and answer/probability separation."""
import argparse
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .core import metrics

BASE=Path(__file__).resolve().parents[2]


def finalize(run):
    manifest_path=run/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('postprocessed'):raise ValueError('Already finalized; do not append duplicate rows')
    db=sqlite3.connect((BASE/'cache/calls.sqlite').as_uri()+'?mode=ro',uri=True)
    diagnostics=[];newmetrics=[];checks=[]
    for ds in ['wa','ag','da','ab']:
        canonical=pd.DataFrame([json.loads(s) for s in (BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()])
        llm=pd.read_csv(run/f'features/{ds}_test_LLM.csv')
        for col,system in [('p_luna','L1-H1'),('p_deepseek','L2-H1')]:
            answers=[];conflicts=[]
            for _,row in llm.iterrows():
                answer=np.nan;conflict=False
                key=row.get(col+'_cache_key')
                if isinstance(key,str):
                    response=json.loads(db.execute('SELECT response FROM calls WHERE key=?',(key,)).fetchone()[0])
                    try:
                        value=json.loads(response['choices'][0]['message']['content'])
                        text=str(value.get('answer') or value.get('prediction') or '').lower().strip()
                        if text in ['yes','match','true','same','1']:answer=1
                        elif text in ['no','non_match','false','different','0']:answer=0
                        if np.isfinite(answer):conflict=(bool(answer) != (row[col]>=.5))
                    except (ValueError,KeyError,TypeError):pass
                answers.append(answer);conflicts.append(conflict)
            answers=np.array(answers);finite=np.isfinite(answers)
            m=metrics(llm.label.values[finite],answers[finite])
            diagnostics.append(dict(dataset=ds,system=system,n=len(llm),n_answer=int(finite.sum()),
                n_answer_probability_conflicts=int(sum(conflicts)),conflict_rate=float(np.mean(conflicts)),
                answer_f1=m['f1'],raw_probability_f1=metrics(llm.label.values[np.isfinite(llm[col])],llm[col].values[np.isfinite(llm[col])])['f1'],
                interpretation='Reported match_probability kept unmodified; explicit answer is a separate secondary decision source'))
            pd.DataFrame(dict(pair_id=llm.pair_id,label=llm.label,explicit_answer=answers,
                              reported_match_probability=llm[col],answer_probability_conflict=conflicts)).to_csv(run/f'predictions/{ds}_{system}_answer_diagnostics.csv',index=False)
            newmetrics.append(dict(dataset=ds,system=system+'-explicit-answer',budget=0,seed=0,
                n_expected=len(llm),n_available=int(finite.sum()),coverage=float(finite.mean()),
                status='secondary_explicit_answer',threshold=.5,f1_at_05=m['f1'],**m,
                note='Secondary decision metric, not a probability/calibration result; differs from preregistered probability threshold source'))
        # Verify every main prediction artifact against canonical IDs and labels.
        for p in (run/'predictions').glob(f'{ds}_*_b*_s*.csv'):
            df=pd.read_csv(p)
            assert df.pair_id.tolist()==canonical.pair_id.tolist(),f'ID order failure: {p.name}'
            assert df.label.tolist()==canonical.label.tolist(),f'Label failure: {p.name}'
            finite=np.isfinite(df.probability)
            assert df.loc[finite,'prediction'].tolist()==(df.loc[finite,'probability']>=df.loc[finite,'threshold']).astype(int).tolist(),p.name
            checks.append(dict(artifact=p.name,n=len(df),n_missing=int((~finite).sum()),status='passed'))
    db.close()
    pd.DataFrame(diagnostics).to_csv(run/'tables/llm_answer_probability_diagnostics.csv',index=False)
    df=pd.read_csv(run/'tables/metrics_by_seed.csv')
    for row in diagnostics:
        mask=(df.dataset==row['dataset'])&(df.system==row['system'])
        df.loc[mask,'note']=df.loc[mask,'note'].fillna('')+f"; answer/probability conflict rate={row['conflict_rate']:.6f}; semantic review required"
    df=pd.concat([df,pd.DataFrame(newmetrics)],ignore_index=True)
    df.to_csv(run/'tables/metrics_by_seed.csv',index=False)
    df.groupby(['dataset','system','budget','status'],dropna=False).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),
        n_seeds=('seed','count'),coverage=('coverage','min'),f1_at_05_mean=('f1_at_05','mean')).reset_index().to_csv(run/'tables/em_summary.csv',index=False)
    slices=pd.read_csv(run/'tables/error_slices_by_seed.csv')
    slices.loc[slices.tp+slices.fn==0,'auprc']=np.nan
    slices.to_csv(run/'tables/error_slices_by_seed.csv',index=False)
    for (ds,system,budget,seed),group in slices.groupby(['dataset','system','budget','seed']):
        row=df[(df.dataset==ds)&(df.system==system)&(df.budget.astype(str)==str(budget))&(df.seed==seed)]
        assert len(row)==1
        for col in ['tp','fp','fn','tn']:assert int(group[col].sum())==int(row.iloc[0][col])
    pd.DataFrame(checks).to_csv(run/'tables/prediction_integrity_checks.csv',index=False)
    # The initial process loaded these modules before concurrent client edits.
    # Preserve hashes of the executed versions rather than claiming later edits ran.
    archive=BASE/'archive/pre_codex_repair_20260930/legacy_code_reports_progress.zip'
    if run.name=='recovery_20260930':
        with zipfile.ZipFile(archive) as z:
            for rel in ['src/recovery/core.py','src/experiments/e2_attribution.py']:
                manifest['inputs'][rel]=hashlib.sha256(z.read('jevwrangle/'+rel)).hexdigest()
    manifest['postprocessed']=True
    manifest['postprocess_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['verification']=dict(main_prediction_files=len(checks),all_ids_labels_thresholds_passed=True,
        slices_reconcile=True,probability_and_answer_separated=True,result_freeze=False)
    manifest['output_sha256']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in run.rglob('*') if p.is_file() and p!=manifest_path}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(verified_prediction_files=len(checks),diagnostics=diagnostics),ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();finalize(args.run.resolve())
