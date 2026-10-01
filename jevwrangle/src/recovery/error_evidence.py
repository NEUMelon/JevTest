"""Partitioned error slices and a real, unfilled human-review packet."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from .error_rules import classify_pair_taxonomy
from .core import metrics
BASE=Path(__file__).resolve().parents[2]

def main(out):
    out.mkdir(exist_ok=False);inputs={};slices=[];sampled=[];integrity=[];gains=[]
    for ds in ['wa','ag','da','ab']:
        path=BASE/f'data/canonical/{ds}/test.jsonl';inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        pairs=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()];ids=[r['pair_id'] for r in pairs];truth=np.array([r['label'] for r in pairs])
        categories=np.array([classify_pair_taxonomy(ds,r['record_a'],r['record_b'],r['label']) for r in pairs]);predictions={}
        for system,budget,folder in [('J-H1n',0,'recovery_repaired_20260930'),('J-D+LR',200,'recovery_repaired_20260930'),
                                     ('J-D+C+LR',200,'recovery_repaired_20260930'),('L2-D+LR',200,'overnight_finalization_20260930/llm_evaluation')]:
            path=BASE/f'runs/{folder}/predictions/{ds}_{system}_b{budget}_s0.csv'
            if not path.exists():continue
            inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            f=pd.read_csv(path).set_index('pair_id',verify_integrity=True).loc[ids]
            if not np.array_equal(f.label,truth):raise ValueError('Error taxonomy ID/label mismatch')
            p=f.probability.values;pred=f.prediction.values;t=float(f.threshold.iloc[0]);predictions[system]=pred
            if not np.array_equal(pred,(p>=t).astype(int)):raise ValueError('Error taxonomy decision mismatch')
            local=[]
            for category in sorted(set(categories)):
                mask=categories==category;measurement=metrics(truth[mask],p[mask],t)
                row=dict(dataset=ds,system=system,budget=budget,seed=0,category=category,n=int(mask.sum()),
                    n_errors=int((pred[mask]!=truth[mask]).sum()),**measurement,
                    label_noise_status='Heuristic candidate only, never a corrected benchmark label')
                local.append(row);slices.append(row)
            overall=metrics(truth,p,t)
            for key in ['tp','fp','fn','tn']:
                if sum(r[key] for r in local)!=overall[key]:raise ValueError('Partitioned confusion counts do not reconstruct main table')
            integrity.append(dict(dataset=ds,system=system,status='ALL_SLICES_RECONSTRUCT_MAIN_COUNTS',**overall))
            error_idx=np.flatnonzero(pred!=truth);selected=np.random.default_rng(20260930).choice(error_idx,size=min(100,len(error_idx)),replace=False)
            for i in selected:sampled.append(dict(dataset=ds,system=system,pair_id=ids[i],label=int(truth[i]),prediction=int(pred[i]),
                probability=float(p[i]),rule_category=categories[i],record_a=pairs[i]['record_a'],record_b=pairs[i]['record_b'],
                agent_semantic_review=None,reviewer_category=None,reviewer_agrees=None,note='Machine rule; manual correction and owner agreement not yet observed'))
        if all(k in predictions for k in ['J-D+LR','J-D+C+LR']):
            a,b=predictions['J-D+LR'],predictions['J-D+C+LR']
            for category in sorted(set(categories)):
                mask=categories==category;wrong=(a!=truth)&mask;newwrong=(b!=truth)&mask
                gains.append(dict(dataset=ds,category=category,n=int(mask.sum()),base_errors=int(wrong.sum()),
                    errors_repaired=int((wrong&~newwrong).sum()),errors_introduced=int((~wrong&newwrong).sum()),
                    repaired_fraction=float((wrong&~newwrong).sum()/wrong.sum()) if wrong.any() else None))
    pd.DataFrame(slices).to_csv(out/'partitioned_slices.csv',index=False)
    pd.DataFrame(integrity).to_csv(out/'confusion_reconciliation.csv',index=False)
    pd.DataFrame(gains).to_csv(out/'hybrid_slice_repairs.csv',index=False)
    (out/'sampled_errors_for_semantic_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sampled),encoding='utf8')
    selected=np.random.default_rng(20260930).choice(len(sampled),size=min(30,len(sampled)),replace=False)
    packet=[dict(review_id=i+1,**sampled[index]) for i,index in enumerate(selected)]
    (out/'owner_30_review_packet.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in packet),encoding='utf8')
    instructions='''# 错误类别人工复核：30 个待审样本

每例包含原记录、冻结标签、预测与机器规则类别。请阅读记录，填写 reviewer_category、reviewer_agrees。Label Noise / Ambiguous 仅表示候选，不能据此修改测试标签。当前 agent_semantic_review、负责人一致率均未观察；不把机器分类称成人工复核。

完整自动分类及主表混淆矩阵的逐项对账见 partitioned_slices.csv、confusion_reconciliation.csv。每个样本只属一个规则类别，所有 TP/FP/FN/TN 按类别求和须还原主表。
'''
    (out/'复核说明.md').write_text(instructions,encoding='utf8')
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    for filename,source in [('error_rules.py',BASE/'src/recovery/error_rules.py'),('legacy_error_taxonomy.py',BASE/'src/experiments/e8_error_taxonomy.py')]:
        (out/filename).write_bytes(source.read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(status='AUTOMATIC_SLICES_VERIFIED_HUMAN_REVIEW_PENDING',inputs=inputs,
        rule_source_sha256=hashlib.sha256((BASE/'src/experiments/e8_error_taxonomy.py').read_bytes()).hexdigest(),
        normalized_rule_sha256=hashlib.sha256((BASE/'src/recovery/error_rules.py').read_bytes()).hexdigest(),
        rule_note='Normalize null/None/nan/n/a missing markers before applying the same ordered, exclusive heuristic; predictions and benchmark gold labels unchanged. Rule category is not a verified causal explanation.',
        n_sampled=len(sampled),owner_packet=len(packet),agreement=None,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*') if p.is_file()}),indent=2))
