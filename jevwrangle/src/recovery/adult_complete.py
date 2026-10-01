"""Repair missing Adult zero-shot evidence only; never reuse repeated splits for training."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from src.experiments.e7_error_detection import load_ed_dataset,build_questions_for_row
from src.clients.ledger import BudgetExceededError
from .core import noul,metrics
BASE=Path(__file__).resolve().parents[2]

def complete(client,out):
    out.mkdir(exist_ok=False)
    source=BASE/'runs/ed_recovery_20260930/features/ad_test.csv';frame=pd.read_csv(source).set_index('cell_id',verify_integrity=True)
    original=frame.copy();observations=[];failed=[]
    mapping={'H_err':'p_h_err','D_typo':'p_d_typo','D_format':'p_d_format','D_cons':'p_d_cons','D_place':'p_d_place','D_imp':'p_d_imp'}
    for group in load_ed_dataset('ad','test'):
        ids=[c['cell_id'] for c in group['cells']]
        missing=frame.loc[ids].status!='verified_cache'
        if not missing.any():continue
        q=build_questions_for_row(group['row'],[c['col'] for c in group['cells']],'ad')
        try:
            response,meta=client.ask({'row':group['row']},q,context={'experiment':'Adult_missing_zero_shot_evidence','cell_ids':ids})
            decoded={c['cell_id']:{name:noul(response['answers'][qid+'_'+c['col']]) for qid,name in mapping.items()} for c in group['cells']}
            for identity in frame.loc[ids].index[missing]:
                for name,value in decoded[identity].items():frame.loc[identity,name]=value
                frame.loc[identity,['cache_key','request_alias','response_model','status']]=[meta['key'],'jev-1.13',response.get('model'),'new_repair']
            observations.append(dict(cell_ids=ids,response=response,meta=meta,questions=q))
        except BudgetExceededError:break
        except Exception as error:failed.append(dict(cell_ids=ids,error_type=type(error).__name__,error=str(error)[:200]))
    untouched=original.status=='verified_cache'
    for name in mapping.values():
        if not np.array_equal(frame.loc[untouched,name].values,original.loc[untouched,name].values):raise ValueError('Adult successful prediction changed')
    canonical=[json.loads(s) for s in (BASE/'data/canonical/ad/test.jsonl').read_text(encoding='utf8').splitlines()]
    expected={c['cell_id']:c['label_error'] for c in canonical}
    if frame.label.to_dict()!=expected:raise ValueError('Adult coverage/labels changed')
    finite=np.isfinite(frame.p_h_err);result=dict(dataset='ad',system='J-Herr',protocol='Legacy zero-shot test; only absent evidence repaired with original question context',
        n_expected=len(frame),n_available=int(finite.sum()),status='COMPLETE_ZERO_SHOT_NOT_SUPERVISED' if finite.all() else 'BLOCKED_MISSING',
        **metrics(frame.label.values[finite],frame.p_h_err.values[finite]))
    frame.reset_index().to_csv(out/'ad_test_features.csv',index=False)
    pd.DataFrame([result]).to_csv(out/'metrics.csv',index=False)
    (out/'repair_observations.json').write_text(json.dumps(observations,indent=2),encoding='utf8')
    (out/'manifest.json').write_text(json.dumps(dict(status=result['status'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failures=failed,
        limitation='Original train/valid/test repeated; never claim supervised performance from original splits',
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file()}),indent=2),encoding='utf8')
    return result
