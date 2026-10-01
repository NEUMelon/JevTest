"""Diagnostic EM pair-content deduplication; official splits stay unchanged."""
import argparse,hashlib,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from .core import metrics
BASE=Path(__file__).resolve().parents[2]
FROZEN=BASE/'runs/em_content_sensitivity_frozen_20260930'

def content(row):
    return hashlib.sha256(json.dumps({k:row[k] for k in ['record_a','record_b']},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest()

def freeze():
    if FROZEN.exists():
        if not (FROZEN/'manifest.json').exists():raise ValueError('Partial sensitivity freeze, preserve before repair')
        return
    FROZEN.mkdir();datasets={};inputs={}
    for ds in ['wa','ag','da','ab']:
        splits={}
        for split in ['train','valid','test']:
            p=BASE/f'data/canonical/{ds}/{split}.jsonl'
            inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
            splits[split]=[json.loads(line) for line in p.read_text(encoding='utf8').splitlines()]
        seen={content(r) for split in ['train','valid'] for r in splits[split]}
        group_labels=defaultdict(set)
        for r in splits['test']:group_labels[content(r)].add(r['label'])
        retained=set();keep=[];excluded=[];cross_seen=0;test_duplicate=0;test_conflict=0
        for r in splits['test']:
            key=content(r)
            if key in seen:reason='Exact full pair also in train or valid';cross_seen+=1
            elif len(group_labels[key])>1:reason='Conflicting test labels for identical full input';test_conflict+=1
            elif key in retained:reason='Duplicate full pair within test';test_duplicate+=1
            else:
                retained.add(key);keep.append(r['pair_id']);continue
            excluded.append(dict(pair_id=r['pair_id'],reason=reason))
        datasets[ds]=dict(keep_ids=keep,excluded=excluded,n_original=len(splits['test']),n_retained=len(keep),
            n_cross_train_valid=cross_seen,n_test_duplicate=test_duplicate,n_test_conflicting=test_conflict,
            n_positive_original=sum(r['label'] for r in splits['test']),n_positive_retained=sum(r['label'] for r in splits['test'] if r['pair_id'] in set(keep)))
    (FROZEN/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (FROZEN/'manifest.json').write_text(json.dumps(dict(status='FROZEN_DIAGNOSTIC_MASKS_NOT_PRIMARY_SPLITS',datasets=datasets,input_sha256=inputs,
        protocol='Exact JSON pair content; excludes full-pair train/valid repeats, collapses within-test duplicates, excludes conflicting duplicate-test gold. Sharing one entity alone is allowed.',
        timing='Defined after Jev/LLM results were examined and before Qwen analysis; diagnostic only, not a retrospective preregistration.',
        interpretation='Conservative common mask excludes all train/valid repeats even if a low-budget seed did not label them. No retraining, no benchmark label correction, no unseen-entity generalization claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf8')

def main(out):
    freeze();out.mkdir(exist_ok=False)
    mask=json.loads((FROZEN/'manifest.json').read_text(encoding='utf8'))
    inputs={str((FROZEN/'manifest.json').relative_to(BASE)):hashlib.sha256((FROZEN/'manifest.json').read_bytes()).hexdigest()}
    for rel,expected in mask['input_sha256'].items():
        assert hashlib.sha256((BASE/rel).read_bytes()).hexdigest()==expected
    rows=[]
    folders=[BASE/'runs/recovery_repaired_20260930/predictions',BASE/'runs/magellan_v2_20260930/predictions',
        BASE/'runs/overnight_finalization_20260930/llm_evaluation/predictions',
        BASE/'runs/overnight_synthesis_20260930/qwen_evaluation/predictions',BASE/'runs/overnight_synthesis_20260930/nli_evaluation/predictions']
    jobs=[]
    for folder in folders:
        if not folder.exists():raise ValueError('Completed prediction source missing: '+str(folder))
        for p in folder.glob('*.csv'):
            match=re.fullmatch(r'(wa|ag|da|ab)_(.+)_b(\d+|full)_s(\d+)\.csv',p.name)
            if match:
                # The recovery folder also retains historical LLM adapters and
                # the old string baseline. Their protocol is not the new one.
                if folder==folders[0] and not (match.group(2).startswith('J-') or match.group(2)=='C+LR'):continue
                jobs.append((p,*match.groups(),None))
    for p in (BASE/'runs/gpu_backup_20260930/ditto').glob('*/manifest.json'):
        meta=json.loads(p.read_text(encoding='utf8'));j=meta['job']
        if meta['status']!='TRAINED_NOT_FROZEN':raise ValueError('Incomplete Ditto sensitivity source')
        jobs.append((p.parent/'test_predictions.csv',j['dataset'],'Ditto',str(j['budget']),str(j['seed']),meta['threshold']))
    for p,ds,system,budget,seed,t in jobs:
        f=pd.read_csv(p).set_index('pair_id',verify_integrity=True)
        canonical={r['pair_id']:r['label'] for r in [json.loads(s) for s in (BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()]}
        if not set(f.index)<=set(canonical) or any(canonical[k]!=v for k,v in zip(f.index,f.label)):raise ValueError('Sensitivity ID/gold mismatch')
        if len(f)!=len(canonical) and not (system.startswith('L1') and len(f)==500):raise ValueError('Unexpected partial prediction source')
        threshold=t if t is not None else float(f.threshold.iloc[0])
        if not np.array_equal(f.prediction,(f.probability>=threshold).astype(int)):raise ValueError('Sensitivity frozen decision mismatch')
        keep=set(mask['datasets'][ds]['keep_ids']);g=f.loc[[k for k in f.index if k in keep]]
        if not len(g):raise ValueError('Empty sensitivity sample')
        inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        full=metrics(f.label,f.probability,threshold);reduced=metrics(g.label,g.probability,threshold)
        rows.append(dict(dataset=ds,system=system,budget=budget,seed=int(seed),n_original=len(f),n_retained=len(g),n_excluded=len(f)-len(g),
            f1_original=full['f1'],f1_retained=reduced['f1'],delta_f1=reduced['f1']-full['f1'],
            tp=reduced['tp'],fp=reduced['fp'],fn=reduced['fn'],tn=reduced['tn'],
            protocol='Fixed Luna 500 intersect diagnostic mask' if system.startswith('L1') else 'Full canonical test intersect diagnostic mask'))
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics_by_seed.csv',index=False)
    frame.groupby(['dataset','system','budget','protocol']).agg(f1_original_mean=('f1_original','mean'),f1_retained_mean=('f1_retained','mean'),delta_f1_mean=('delta_f1','mean'),
        n_seeds=('seed','count'),n_original=('n_original','first'),n_retained=('n_retained','first'),n_excluded=('n_excluded','first')).reset_index().to_csv(out/'summary.csv',index=False)
    pd.DataFrame([dict(dataset=ds,**{k:v for k,v in data.items() if k not in ['keep_ids','excluded']}) for ds,data in mask['datasets'].items()]).to_csv(out/'mask_counts.csv',index=False)
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_SENSITIVITY_COMPLETE_NOT_PRIMARY',inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation=mask['interpretation'],n_prediction_groups=len(rows),outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.freeze_only:freeze()
    else:main(args.output.resolve())
