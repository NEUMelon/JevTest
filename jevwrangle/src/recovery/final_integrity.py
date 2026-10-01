"""Independent final artifact checks; no model calls, training or test tuning."""
import hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/final_integrity_20261001_reviewed'

def main():
    OUT.mkdir(exist_ok=False);digests={};checks=[]
    def sha(p):
        key=str(p.resolve())
        if key not in digests:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            digests[key]=h.hexdigest()
        return digests[key]
    def require_hash(p,expected):
        if sha(p)!=expected:raise ValueError('SHA mismatch '+str(p))
    folders=['reports/overnight_recovery_20261001_reviewed','runs/qwen_diagnostics_20260930','runs/flights_row_recovery_20260930','runs/em_content_sensitivity_20261001_reviewed','runs/error_evidence_missing_normalized_20260930','runs/gpu_backup_20260930/qwen_recovery_20260930','runs/gpu_backup_20260930/nli_recovery_20260930']
    for rel in folders:
        folder=BASE/rel;m=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        for field in ['inputs','input_sha256']:
            for path,expected in m.get(field,{}).items():require_hash(BASE/path,expected)
        for path,expected in m.get('outputs',{}).items():require_hash(folder/path,expected)
        checks.append(dict(folder=rel,status=m['status'],n_outputs=len(m.get('outputs',{}))))
    backup=json.loads((BASE/'runs/gpu_backup_20260930/backup_status.json').read_text(encoding='utf8'))
    if backup['status']!='SHUTDOWN_SENT_SSH_UNREACHABLE':raise ValueError('Shutdown evidence requires review')
    for path,expected in backup['downloaded'].items():require_hash(BASE/'runs/gpu_backup_20260930'/path,expected)
    canonical={ds:{r['pair_id']:r['label'] for r in map(json.loads,(BASE/f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines())} for ds in ['wa','ag','da','ab']}
    frames={};measured=[]
    sensitivity=json.loads((BASE/'runs/em_content_sensitivity_20261001_reviewed/manifest.json').read_text(encoding='utf8'))
    jobs=[]
    for rel in sensitivity['inputs']:
        if '/predictions/' not in rel and not rel.endswith('/test_predictions.csv'):continue
        p=BASE/rel
        if p.name=='test_predictions.csv':
            m=json.loads((p.parent/'manifest.json').read_text(encoding='utf8'));j=m['job'];ds=j['dataset'];system='Ditto';budget=str(j['budget']);seed=j['seed'];t=m['threshold']
        else:
            match=re.fullmatch(r'(wa|ag|da|ab)_(.+)_b(\d+|full)_s(\d+)\.csv',p.name)
            if not match:raise ValueError('Prediction filename schema mismatch')
            ds,system,budget,seed=match.groups();seed=int(seed);t=None
        f=pd.read_csv(p).set_index('pair_id',verify_integrity=True);frames[(ds,system,budget,seed)]=(f,t)
        truth=canonical[ds]
        if set(f.index)-set(truth) or any(truth[k]!=v for k,v in zip(f.index,f.label)):raise ValueError('Canonical prediction ID/label mismatch')
        if len(f)!=len(truth) and not (system.startswith('L1') and len(f)==500):raise ValueError('Unexpected partial group')
        threshold=float(f.threshold.iloc[0]) if t is None else t
        if not np.isfinite(f.probability).all() or not f.probability.between(0,1).all():raise ValueError('Nonfinite or out-of-range probability')
        if not np.array_equal((f.probability>=threshold).astype(int),f.prediction):raise ValueError('Frozen threshold mismatch')
        y=f.label.to_numpy();pred=f.prediction.to_numpy();tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum());tn=int(((y==0)&(pred==0)).sum());den=2*tp+fp+fn
        measured.append(dict(dataset=ds,system=system,budget=budget,seed=seed,n_test=len(f),f1=100*2*tp/den if den else 0.,tp=tp,fp=fp,fn=fn,tn=tn))
    if len(measured)!=1296:raise ValueError('Expected 1296 primary canonical prediction groups')
    full=pd.DataFrame(measured);full.to_csv(OUT/'independent_confusion_by_seed.csv',index=False)
    means=full.groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),n_seeds=('seed','count')).reset_index()
    master=pd.read_csv(BASE/'reports/overnight_recovery_20261001_reviewed/tables/full_system_summary.csv');master.budget=master.budget.astype(str)
    aligned=master.merge(means,on=['dataset','system','budget'],how='outer',validate='one_to_one',suffixes=('_reported','_independent'),indicator=True)
    if not (aligned['_merge']=='both').all() or not np.allclose(aligned.f1_mean_reported,aligned.f1_mean_independent,atol=1e-9,rtol=0) or not np.array_equal(aligned.n_seeds_reported,aligned.n_seeds_independent):raise ValueError('Main table does not reproduce from canonical predictions')
    aligned.to_csv(OUT/'master_roundtrip.csv',index=False)
    selection=json.loads((BASE/'runs/overnight_api_20260930/manifest.json').read_text(encoding='utf8'))['protocol']
    matched=pd.read_csv(BASE/'reports/overnight_recovery_20261001_reviewed/tables/matched_500_metrics_by_seed.csv')
    if len(matched)!=304 or matched.duplicated(['dataset','system','budget','seed']).any():raise ValueError('Matched group completeness failure')
    for row in matched.itertuples(index=False):
        f,t=frames[(row.dataset,row.system,str(row.budget),row.seed)];g=f.loc[selection[row.dataset]['Luna_test_ids']]
        y=g.label.to_numpy();pred=g.prediction.to_numpy();tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum());tn=int(((y==0)&(pred==0)).sum());den=2*tp+fp+fn
        if [tp,fp,fn,tn]!=[row.tp,row.fp,row.fn,row.tn] or abs((100*2*tp/den if den else 0)-row.f1)>1e-9:raise ValueError('Matched table confusion mismatch')
    gates=pd.read_csv(BASE/'runs/flights_row_recovery_20260930/completeness_gates.csv')
    if len(gates)!=80 or not (gates.status=='COMPLETE').all():raise ValueError('Flights partial groups must be disclosed')
    packet=[json.loads(line) for line in (BASE/'reports/overnight_recovery_20261001_reviewed/owner_30_review_packet.jsonl').read_text(encoding='utf8').splitlines()]
    if len(packet)!=30:raise ValueError('Owner packet completeness mismatch')
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'status.json').write_text(json.dumps(dict(status='FINAL_NUMERIC_AND_EVIDENCE_INTEGRITY_VERIFIED_VISUAL_REVIEW_SEPARATE',n_hashed_files=len(digests),gpu_backup_files=len(backup['downloaded']),n_prediction_groups=1296,n_master_groups=len(master),n_matched_groups=304,n_flights_groups=80,owner_review='30 prepared cases, human review still pending',collections=checks,outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}),indent=2),encoding='utf8')
    print(json.dumps({'n_hashed_files':len(digests),'master_groups':len(master),'matched_groups':304,'canonical_prediction_groups':1296,'flights_groups':80,'status':'VERIFIED'}))

if __name__=='__main__':main()
