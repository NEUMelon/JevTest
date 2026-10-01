"""Recover ED evidence and build a clearly separate Adult group-split appendix."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from src.experiments.e7_error_detection import load_ed_dataset,build_questions_for_row
from .core import noul,metrics,logit,fit_aggregator
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/ed_recovery_20260930'


def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'features').mkdir();(OUT/'predictions').mkdir()
    db=sqlite3.connect((BASE/'cache/calls.sqlite').as_uri()+'?mode=ro',uri=True)
    summaries=[];issues=[];strict={};inputs={}
    def track(p):inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    mapping={'H_err':'p_h_err','D_typo':'p_d_typo','D_format':'p_d_format','D_cons':'p_d_cons','D_place':'p_d_place','D_imp':'p_d_imp'}
    for ds in ['ho','ad','fl']:
        for split in ['train','test']:
            canonical=[json.loads(s) for s in track(BASE/f'data/canonical/{ds}/{split}.jsonl').read_text(encoding='utf8').splitlines()]
            expected={c['cell_id']:c['label_error'] for c in canonical}
            old=pd.read_csv(track(BASE/f'runs/e7_jev_{ds}_{split}.csv')).set_index('cell_id',verify_integrity=True)
            rows=[]
            for group in load_ed_dataset(ds,split):
                q=build_questions_for_row(group['row'],[c['col'] for c in group['cells']],ds);candidates=[]
                for alias in ['jev-latest','jev-1.13']:
                    payload={'model':alias,'state':{'row':group['row']},'questions':q}
                    key=hashlib.sha256(json.dumps(['aihubmix',payload],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                    hit=db.execute('SELECT response FROM calls WHERE key=?',(key,)).fetchone()
                    if hit:
                        response=json.loads(hit[0])
                        try:
                            decoded={c['cell_id']:{column:noul(response['answers'][qid+'_'+c['col']]) for qid,column in mapping.items()}
                                     for c in group['cells']}
                            match=all(abs(decoded[c['cell_id']]['p_h_err']-float(old.loc[c['cell_id'],'p_h_err']))<1e-9 for c in group['cells'])
                            candidates.append((match,decoded,key,alias,response.get('model')))
                        except (ValueError,KeyError,TypeError):pass
                chosen=next((r for r in candidates if r[0]),None)
                group_id=hashlib.sha256(json.dumps(group['row'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                for c in group['cells']:
                    assert expected[c['cell_id']]==int(old.loc[c['cell_id'],'label_error'])
                    r=dict(cell_id=c['cell_id'],table_row_id=group['table_row_id'],content_group=group_id,col=c['col'],label=c['label_error'])
                    if chosen:r.update(chosen[1][c['cell_id']],cache_key=chosen[2],request_alias=chosen[3],response_model=chosen[4],status='verified_cache')
                    else:r['status']='untraceable_legacy';issues.append(dict(dataset=ds,split=split,cell_id=c['cell_id'],kind='cache_conflict_or_missing'))
                    rows.append(r)
            df=pd.DataFrame(rows);strict[ds,split]=df
            df.to_csv(OUT/f'features/{ds}_{split}.csv',index=False)
        # Full Jev zero-shot metrics. Never compare these with a smaller LLM subset.
        test=strict[ds,'test'];finite=np.isfinite(test.p_h_err)
        summaries.append(dict(dataset=ds,system='J-Herr',protocol='legacy zero-shot full test',n_expected=len(test),
                              n_available=int(finite.sum()),status='complete' if finite.all() else 'partial_BLOCKED',
                              **metrics(test.label.values[finite],test.p_h_err.values[finite])))
        for model in ['gpt-6-luna','deepseek-v4-flash']:
            # This file has explicit error probabilities (not the EM answer-confidence bug).
            llm=pd.read_csv(track(BASE/f'runs/e7_llm_{ds}_{model}.csv')).set_index('cell_id',verify_integrity=True)
            common=test[test.cell_id.isin(llm.index)].copy();aligned=llm.loc[common.cell_id]
            assert aligned.label_error.tolist()==common.label.tolist()
            good=np.isfinite(common.p_h_err)&np.isfinite(aligned.p_error.values)
            for name,p in [('J-Herr',common.p_h_err.values),(model,aligned.p_error.values)]:
                summaries.append(dict(dataset=ds,system=name,protocol='legacy first 200 rows common subset',n_expected=len(common),
                    n_available=int(good.sum()),status='legacy_subset_descriptive' if good.all() else 'partial_BLOCKED',
                    **metrics(common.label.values[good],p[good]),note='LLM raw-response reconstruction remains pending; not an independently sampled full-test comparison'))
    # New appendix protocol: identical dirty rows remain in one group. Retain one
    # deterministic representative per group/column only when labels agree.
    adult=strict['ad','test'];dedup=[];ambiguous=[]
    for (group,col),cells in adult.groupby(['content_group','col'],sort=True):
        if cells.label.nunique()!=1:ambiguous.append(dict(content_group=group,col=col));continue
        verified=cells[cells.status=='verified_cache'].sort_values('cell_id')
        if len(verified):dedup.append(verified.iloc[0].to_dict())
        else:issues.append(dict(dataset='ad',content_group=group,col=col,kind='dedup_group_has_no_verified_features'))
    df=pd.DataFrame(dedup);groups=np.array(sorted(df.content_group.unique()));rng=np.random.default_rng(20260930);rng.shuffle(groups)
    n=len(groups);cut1=int(.64*n);cut2=int(.80*n)
    partitions={g:('train' if i<cut1 else 'valid' if i<cut2 else 'test') for i,g in enumerate(groups)}
    df['new_split']=df.content_group.map(partitions)
    df.to_csv(OUT/'features/ad_group_split_appendix.csv',index=False)
    for a in ['train','valid','test']:
        for b in ['train','valid','test']:
            if a!=b:assert not set(df[df.new_split==a].content_group)&set(df[df.new_split==b].content_group)
    train=df[df.new_split=='train'];test=df[df.new_split=='test'];cols=list(mapping.values())
    model,thr,oof,params=fit_aggregator(logit(train[cols].values),train.label.values,len(train),0)
    probability=model.predict_proba(logit(test[cols].values))[:,1]
    test.assign(probability=probability,prediction=(probability>=thr).astype(int)).to_csv(OUT/'predictions/ad_group_appendix_full.csv',index=False)
    summaries.append(dict(dataset='ad',system='J-H+D+LR',protocol='NEW post-audit grouped deduplicated appendix; 64/16/20 content groups',
                          n_expected=len(test),n_available=len(test),status='EXPLORATORY_NEW_PROTOCOL_NOT_COMPARABLE_TO_LITERATURE',
                          threshold=thr,**metrics(test.label,probability,thr)))
    import joblib
    joblib.dump(model,OUT/'ad_appendix_model.joblib')
    train[['cell_id','content_group','label']].assign(oof_probability=oof).to_csv(OUT/'ad_appendix_oof.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'ed_descriptive_metrics.csv',index=False)
    references=[dict(dataset=ds,system=name,f1=f1,source='https://www.vldb.org/pvldb/vol16/p738-narayan.pdf',table='2',protocol='Literature protocol; not matched to this run')
        for ds,values in [('ho',[('HoloClean',51.4),('HoloDetect',94.4),('GPT3-175B k10',97.8)]),('ad',[('HoloClean',54.5),('HoloDetect',99.1),('GPT3-175B k10',99.1)])] for name,f1 in values]
    references.append(dict(dataset='fl',system='Raha',f1=81,source='https://nantang.github.io/research/pubs/raha.pdf',table='5',protocol='Literature full-table 20 labeled tuples; not comparable to this split'))
    pd.DataFrame(references).to_csv(OUT/'literature_context_ONLY.csv',index=False)
    db.close();track(BASE/'cache/calls.sqlite');track(BASE/'src/experiments/e7_error_detection.py')
    manifest=dict(status='DESCRIPTIVE_AND_APPENDIX_NOT_FROZEN',inputs=inputs,ad_groups=n,ad_cells=len(df),ambiguous_cells=ambiguous,
                  ad_partitions={s:int((df.new_split==s).sum()) for s in ['train','valid','test']},
                  notes=['Original Adult supervised results withdrawn','Hospital sparse cell annotations do not satisfy a fully annotated-row budget',
                         'Legacy questions omit frozen column profiles; append labels are post-audit exploratory',
                         'No SOTA claim; literature results only provide separate context'],issues=issues,threshold=thr,aggregator=params,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(pd.DataFrame(summaries)[['dataset','system','protocol','n_available','f1','status']].to_string(index=False))


if __name__=='__main__':main()
