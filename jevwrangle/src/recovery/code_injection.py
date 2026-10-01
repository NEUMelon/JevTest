"""Fixed C+LR clean/attack inference; no API calls or retraining."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from src.experiments.e2_attribution import compute_code_features
BASE=Path(__file__).resolve().parents[2]

def collect(out,rate):
    out.mkdir(exist_ok=False);inputs={};attack_records={};predictions=[];summary=[]
    prompts=BASE/'runs/gpu_backup_20260930/qwen_recovery_20260930/prompts.jsonl'
    inputs[prompts.relative_to(BASE).as_posix()]=hashlib.sha256(prompts.read_bytes()).hexdigest()
    # Stream the large corpus; only the fixed holistic attack states are needed.
    with prompts.open(encoding='utf8') as file:
        for line in file:
            p=json.loads(line)
            if p['experiment']!='E10' or p['qid']!='p_yes':continue
            state=json.loads(p['messages'][1]['content'].split('Record data (JSON):\n',1)[1].split('\n\nQuestion: ',1)[0])
            identity=(p['dataset'],p['pair_id'],p['perturbation'])
            if identity in attack_records:raise ValueError('Duplicate C+LR attack state')
            attack_records[identity]=dict(pair_id=p['pair_id'],label=p['label'],variant=p['perturbation'],**state)
    for ds in ['wa','ab']:
        source=BASE/f'data/canonical/{ds}/test.jsonl';inputs[source.relative_to(BASE).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
        canonical={r['pair_id']:r for s in source.read_text(encoding='utf8').splitlines() if (r:=json.loads(s))}
        dirty=[v for key,v in attack_records.items() if key[0]==ds];ids=sorted(set(r['pair_id'] for r in dirty))
        clean=[dict(**{k:canonical[pair_id][k] for k in ['pair_id','label','record_a','record_b']},variant='clean') for pair_id in ids]
        if len(ids)!=400 or len(dirty)!=1200:raise ValueError('C+LR attack corpus coverage differs from Qwen')
        for row in dirty:
            if row['label']!=canonical[row['pair_id']]['label']:raise ValueError('Attack gold label differs')
        rows=clean+dirty;features=compute_code_features(ds,rows)
        slug=f'{ds}_C+LR_b200_s0';folder=BASE/'runs/recovery_repaired_20260930/models'
        for ext in ['joblib','json']:
            p=folder/(slug+'.'+ext);inputs[p.relative_to(BASE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        model=joblib.load(folder/(slug+'.joblib'));threshold=json.loads((folder/(slug+'.json')).read_text(encoding='utf8'))['threshold']
        p=model.predict_proba(features.values)[:,1]
        frame=pd.DataFrame(dict(dataset=ds,pair_id=[r['pair_id'] for r in rows],label=[r['label'] for r in rows],
            variant=[r['variant'] for r in rows],probability=p,prediction=(p>=threshold).astype(int),threshold=threshold))
        predictions.append(frame);baseline=frame[frame.variant=='clean'].set_index('pair_id').loc[ids]
        for variant in ['T1','T2','T3']:
            g=frame[frame.variant==variant].set_index('pair_id').loc[ids]
            summary.append(dict(dataset=ds,system='C+LR_b200_seed0',variant=variant,experiment='E10',n_test=len(ids),
                **rate(g.prediction.values==1-g.label.values,baseline.prediction.values==baseline.label.values),
                denominator='Clean-correct cases for this system',clean_accuracy=float((baseline.prediction==baseline.label).mean())))
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False);pd.DataFrame(summary).to_csv(out/'attack_summary.csv',index=False)
    for source in [Path(__file__),BASE/'src/experiments/e2_attribution.py']:
        inputs[source.relative_to(BASE).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest();(out/source.name).write_bytes(source.read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_FIXED_CLASSIFIER_NO_NEW_INFERENCE_COST',inputs=inputs,
        note='Use the exact Qwen attack records and frozen C+LR b200 seed0; low ASR alone does not imply high clean matching quality',
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file()}),indent=2))
    return summary
