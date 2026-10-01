"""GPU evidence verification and immutable aggregate artifacts after backup."""
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from .llm_aggregate import aggregate
from .core import metrics
from .compare_recovered import main as compare
BASE=Path(__file__).resolve().parents[2]

def main():
    backup=BASE/'runs/gpu_backup_20260930';out=BASE/'runs/overnight_synthesis_20260930'
    while True:
        status=json.loads((backup/'backup_status.json').read_text(encoding='utf8')) if (backup/'backup_status.json').exists() else {}
        if status.get('status','').startswith('SHUTDOWN_SENT'):break
        time.sleep(30)
    out.mkdir(exist_ok=False)
    checks=[];ditto=[];refs={'wa':86.76,'ag':75.58,'da':98.99}
    for manifest in (backup/'ditto').glob('*/manifest.json'):
        m=json.loads(manifest.read_text(encoding='utf8'));j=m['job'];folder=manifest.parent
        for rel,expected in m['output_sha256'].items():
            if hashlib.sha256((folder/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Backup changed '+rel)
        frame=pd.read_csv(folder/'test_predictions.csv');ds=j['dataset']
        canonical=[json.loads(s) for s in (BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()]
        if frame.pair_id.tolist()!=[r['pair_id'] for r in canonical] or frame.label.tolist()!=[r['label'] for r in canonical]:raise ValueError('Ditto ID/label mismatch')
        measured=metrics(frame.label,frame.probability,m['threshold'])
        if not np.array_equal(frame.prediction,(frame.probability>=m['threshold']).astype(int)):raise ValueError('Ditto threshold mismatch')
        if abs(measured['f1']-m['test_metrics']['f1'])>1e-8:raise ValueError('Ditto metric mismatch')
        ditto.append(dict(dataset=ds,system='Ditto',budget=j['budget'],seed=j['seed'],elapsed_seconds=m['elapsed_seconds'],**measured))
        if j['budget']=='full' and ds in refs:
            gap=measured['f1']-refs[ds];checks.append(dict(dataset=ds,f1=measured['f1'],reference=refs[ds],gap=gap,
                status='WITHIN_5_POINTS_PORT_CHECK' if abs(gap)<=5 else 'STOP_FREEZE_REFERENCE_GAP_REQUIRES_EXPLANATION'))
    if len(ditto)!=40:raise ValueError('Not all 40 Ditto jobs backed up')
    pd.DataFrame(ditto).to_csv(out/'ditto_metrics_by_seed.csv',index=False)
    pd.DataFrame(ditto).groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),n_seeds=('seed','count')).reset_index().to_csv(out/'ditto_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'ditto_port_checks.csv',index=False)
    for name,model,short in [('qwen_recovery_20260930','Qwen3-8B','Q'),('nli_recovery_20260930','bart-large-mnli','NLI')]:
        collection=backup/name;manifest=json.loads((collection/'manifest.json').read_text(encoding='utf8'))
        for rel,expected in manifest['outputs'].items():
            if hashlib.sha256((collection/rel).read_bytes()).hexdigest()!=expected:raise ValueError('GPU probability evidence changed: '+rel)
        aggregate(collection,out/('qwen_evaluation' if short=='Q' else 'nli_evaluation'),models=[(model,short)])
    while not (BASE/'runs/overnight_finalization_20260930/status.json').exists():time.sleep(30)
    compare(out/'paired_comparisons')
    (out/'status.json').write_text(json.dumps(dict(status='AGGREGATED_NOT_FROZEN',gpu_backup=status,
        n_ditto=40,port_checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remaining='Diagnostic Qwen calibration/E5/E7/E9/E10, plot QA, story report and full evidence audit still required'),indent=2))

if __name__=='__main__':main()
