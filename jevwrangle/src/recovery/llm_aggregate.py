"""Evaluate complete repaired LLM features; never replace failed calls by 0.5.

Dataset/model groups are accepted only when BOTH pool and test H/D features are
complete. This can be run while other groups are still being collected. Output
directories are immutable, so each evaluation uses a fresh destination.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from .core import probability, logit, metrics, fit_aggregator

BASE = Path(__file__).resolve().parents[2]


def read_group(parsed, dataset, model, split, kind, pairs, qids):
    """Align by ID and require exact labels and all declared probabilities."""
    selected = {r['pair_id']: r for r in parsed.values()
                if (r['dataset'], r['model'], r['split'], r['kind']) == (dataset, model, split, kind)}
    expected = {p['pair_id'] for p in pairs}
    if set(selected) - expected:
        raise ValueError('Unexpected pair IDs in collection')
    rows, missing = [], []
    for pair in pairs:
        r = selected.get(pair['pair_id'])
        if r is None or r['status'] != 'ok':
            missing.append(dict(pair_id=pair['pair_id'], status=r['status'] if r else 'not_collected'))
            continue
        if r['label'] != pair['label']:
            raise ValueError('Collection label differs from canonical label')
        values = [probability(r['values'][q]) for q in qids]
        rows.append(dict(pair_id=pair['pair_id'], label=pair['label'], **dict(zip(qids, values))))
    return pd.DataFrame(rows), missing


def merge_repairs(parsed, raw):
    for line in raw.decode('utf8').splitlines():
        r=json.loads(line);identity=tuple(r[k] for k in ['dataset','model','split','kind','pair_id'])
        if identity not in parsed or parsed[identity]['status']=='ok':
            raise ValueError('Repair must target an existing failed job, never a successful prediction')
        if r['label']!=parsed[identity]['label']:raise ValueError('Repair label changed')
        if r['status']=='ok':parsed[identity]=r


def aggregate(collection, out, repairs=None, models=None, protocol=None):
    out.mkdir(parents=True, exist_ok=False)
    for d in ['features', 'predictions', 'models', 'oof', 'source_snapshot']:
        (out / d).mkdir()
    hashes = {}
    def track(p):
        hashes[p.relative_to(BASE).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    for rel in ['src/recovery/llm_aggregate.py', 'src/recovery/core.py']:
        p = track(BASE / rel); dest = out / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    journal = track(collection / 'parsed.jsonl')
    # A snapshot, not a streaming mutable file: hash exactly the bytes evaluated.
    raw = journal.read_bytes(); hashes[journal.relative_to(BASE).as_posix()] = hashlib.sha256(raw).hexdigest()
    (out / 'parsed_snapshot.jsonl').write_bytes(raw)
    parsed = {}
    for line in raw.decode('utf8').splitlines():
        r = json.loads(line); identity = tuple(r[k] for k in ['dataset', 'model', 'split', 'kind', 'pair_id'])
        if identity in parsed:
            raise ValueError('Duplicate collection records require explicit attempt reconciliation')
        parsed[identity] = r
    if repairs is not None:
        repair_manifest=json.loads(track(repairs/'manifest.json').read_text(encoding='utf8'))
        collection_manifest=json.loads(track(collection/'manifest.json').read_text(encoding='utf8'))
        if repair_manifest['collection_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Repair refers to a different collection snapshot')
        if repair_manifest['prompt_source_sha256']!=collection_manifest['prompt_source_sha256']:raise ValueError('Repair prompt differs from collection')
        raw_repairs=track(repairs/'parsed.jsonl').read_bytes()
        hashes[(repairs/'parsed.jsonl').relative_to(BASE).as_posix()]=hashlib.sha256(raw_repairs).hexdigest()
        (out/'repairs_snapshot.jsonl').write_bytes(raw_repairs)
        merge_repairs(parsed,raw_repairs)
    rows, gates = [], []
    for ds in ['wa', 'ag', 'da', 'ab']:
        q = yaml.safe_load(track(BASE / f'questions/{ds}/decomposed.yaml').read_text(encoding='utf8'))
        records = {}
        for split, rel in [('pool', f'data/pools/{ds}_pool2000.jsonl'), ('test', f'data/canonical/{ds}/test.jsonl')]:
            records[split] = [json.loads(s) for s in track(BASE / rel).read_text(encoding='utf8').splitlines()]
        for model, short in (models or [('gpt-6-luna', 'L1'), ('deepseek-v4-flash', 'L2')]):
            model_records=records
            budgets=[50,200,1000]
            seed_counts={50:10,200:10,1000:5}
            if protocol and model=='gpt-6-luna':
                selection=protocol[ds]
                model_records={split:[r for r in values if r['pair_id'] in set(selection['Luna_'+split+'_ids'])]
                               for split,values in records.items()}
                budgets=[50,200];seed_counts={50:3,200:3}
            groups, allmissing = {}, []
            for split in model_records:
                for kind, cols in ([('H',['p_yes'])] if model=='bart-large-mnli' else [('H', ['p_yes']), ('D', list(q))]):
                    frame, missing = read_group(parsed, ds, model, split, kind, model_records[split], cols)
                    groups[split, kind] = frame
                    allmissing.extend(dict(split=split, kind=kind, **r) for r in missing)
            gate = dict(dataset=ds, model=model, n_missing=len(allmissing), status='BLOCKED_MISSING' if allmissing else 'COMPLETE_EVALUATED')
            gates.append(gate)
            if allmissing:
                (out / f'{ds}_{short}_missing.json').write_text(json.dumps(allmissing, indent=2), encoding='utf8')
                continue
            for (split, kind), f in groups.items():
                f.to_csv(out / f'features/{ds}_{short}_{split}_{kind}.csv', index=False)
            yt = np.array([p['label'] for p in model_records['test']])
            def evaluate(system, budget, seed, p, threshold):
                slug = f'{ds}_{system}_b{budget}_s{seed}'
                pd.DataFrame(dict(pair_id=[r['pair_id'] for r in model_records['test']], label=yt,
                    probability=p, prediction=(p >= threshold).astype(int), threshold=threshold)).to_csv(out / f'predictions/{slug}.csv', index=False)
                rows.append(dict(dataset=ds, system=system, budget=budget, seed=seed, threshold=threshold,
                    n_test=len(yt), **metrics(yt, p, threshold)))
                return slug
            evaluate(short + '-H1', 0, 0, groups['test', 'H'].p_yes.values, .5)
            pool_ids = [p['pair_id'] for p in model_records['pool']]
            ypool = np.array([p['label'] for p in model_records['pool']])
            for budget in budgets:
                for seed in range(seed_counts[budget]):
                    ids = json.loads(track(BASE / f'data/budgets/{ds}/b{budget}_seed{seed}.json').read_text())['pair_ids']
                    idx = np.flatnonzero(np.isin(pool_ids, ids))
                    if len(idx) != budget or len(set(ids)) != budget:
                        raise ValueError('Frozen budget not covered exactly by feature pool')
                    for kind, system, cols in ([('H',short+'-H1+LR',['p_yes'])] if model=='bart-large-mnli' else [('H', short + '-H1+LR', ['p_yes']), ('D', short + '-D+LR', list(q))]):
                        Xpool = logit(groups['pool', kind][cols].values)
                        Xtest = logit(groups['test', kind][cols].values)
                        m, t, oof, params = fit_aggregator(Xpool[idx], ypool[idx], budget, seed)
                        slug = evaluate(system, budget, seed, m.predict_proba(Xtest)[:, 1], t)
                        joblib.dump(m, out / f'models/{slug}.joblib')
                        (out / f'models/{slug}.json').write_text(json.dumps(dict(threshold=t, feature_columns=cols, **params), indent=2), encoding='utf8')
                        pd.DataFrame(dict(pair_id=np.array(pool_ids)[idx], label=ypool[idx], oof_probability=oof)).to_csv(out / f'oof/{slug}.csv', index=False)
    pd.DataFrame(gates).to_csv(out / 'completeness_gates.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'metrics_by_seed.csv', index=False)
    if rows:
        pd.DataFrame(rows).groupby(['dataset', 'system', 'budget']).agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'), n_seeds=('seed', 'count')).reset_index().to_csv(out / 'summary.csv', index=False)
    manifest = dict(status='PARTIAL_COLLECTION_BLOCKED' if any(g['n_missing'] for g in gates) else 'COMPLETE_NOT_FROZEN',
        result_freeze=False, inputs=hashes, decision_source='unaltered P(YES), explicit textual answers not substituted',
        gates=gates, no_full_budget='LLM features collected only on pool2000 plus test, per plan')
    manifest['output_sha256'] = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps(gates, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--collection', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repairs',type=Path)
    a = ap.parse_args(); aggregate(a.collection.resolve(), a.output.resolve(),a.repairs.resolve() if a.repairs else None)
