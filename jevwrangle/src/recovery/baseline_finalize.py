"""Verify repaired baseline artifacts and compare paired budget runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import metrics
from .statistics import paired_bootstrap

BASE = Path(__file__).resolve().parents[2]


def finalize():
    run = BASE / 'runs/magellan_v2_20260930'
    reference = BASE / 'runs/recovery_repaired_20260930'
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('verified'):
        raise ValueError('Already verified')
    for rel, digest in manifest['inputs'].items():
        assert hashlib.sha256((BASE / rel).read_bytes()).hexdigest() == digest, rel
    source = BASE / 'src/recovery/magellan_v2.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    snapshot = run / 'source_snapshot'
    for rel in ['src/recovery/magellan_v2.py', 'src/recovery/core.py',
                'src/experiments/magellan_feats.py', 'src/recovery/statistics.py',
                'src/recovery/baseline_finalize.py']:
        p = snapshot / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((BASE / rel).read_bytes())
    # These dependencies remained unchanged during the baseline process.
    manifest['source_snapshot_note'] = 'Copied after execution; primary script hash verified against execution manifest; dependency hashes retained for reproducibility'
    table = pd.read_csv(run / 'metrics_by_seed.csv')
    checks, comparisons, compared_hashes = [], [], {}
    for ds in ['wa', 'ag', 'da', 'ab']:
        canonical = pd.DataFrame([json.loads(s) for s in (BASE / f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()])
        for p in sorted((run / 'predictions').glob(f'{ds}_*.csv')):
            f = pd.read_csv(p)
            assert f.pair_id.tolist() == canonical.pair_id.tolist(), p.name
            assert f.label.tolist() == canonical.label.tolist(), p.name
            assert np.isfinite(f.probability).all(), p.name
            assert f.prediction.tolist() == (f.probability >= f.threshold).astype(int).tolist(), p.name
            slug = p.stem
            row = table[(table.dataset == ds) & table.apply(lambda r: f'{r.dataset}_{r.system}_b{r.budget}_s{r.seed}' == slug, axis=1)]
            assert len(row) == 1, p.name
            measured = metrics(f.label.values, f.probability.values, float(f.threshold.iloc[0]))
            for col in ['tp', 'fp', 'fn', 'tn', 'f1', 'precision', 'recall', 'auprc']:
                assert np.isclose(measured[col], row.iloc[0][col], atol=1e-8), (p.name, col)
            system, budget, seed = row.iloc[0][['system', 'budget', 'seed']]
            train = pd.read_csv(run / f'oof/{slug}.csv')
            origin = 'train' if budget == 'full' else None
            if origin:
                expected = [json.loads(s)['pair_id'] for s in (BASE / f'data/canonical/{ds}/train.jsonl').read_text(encoding='utf8').splitlines()]
            else:
                expected = json.loads((BASE / f'data/budgets/{ds}/b{budget}_seed{seed}.json').read_text())['pair_ids']
            assert len(train) == len(expected) and set(train.pair_id) == set(expected), p.name
            assert not set(train.pair_id) & set(f.pair_id), p.name
            checks.append(dict(artifact=p.name, n_test=len(f), n_labeled=len(train), status='passed'))
        for baseline in ['M2+LR', 'M2+best']:
            for budget in [50, 200]:
                aligned = []
                for seed in range(10):
                    a_path = run / f'predictions/{ds}_{baseline}_b{budget}_s{seed}.csv'
                    b_path = reference / f'predictions/{ds}_J-D+LR_b{budget}_s{seed}.csv'
                    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
                    assert a.pair_id.tolist() == b.pair_id.tolist() and a.label.tolist() == b.label.tolist()
                    aligned.append((a.label.values, a.prediction.values, b.prediction.values))
                    for file in [a_path, b_path]:
                        compared_hashes[file.relative_to(BASE).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
                comparisons.append(dict(dataset=ds, budget=budget, system_A=baseline, system_B='J-D+LR',
                    **paired_bootstrap(aligned), status='POST_AUDIT_DESCRIPTIVE_NOT_FROZEN'))
    assert len(checks) == 224
    pd.DataFrame(checks).to_csv(run / 'prediction_integrity_checks.csv', index=False)
    pd.DataFrame(comparisons).to_csv(run / 'paired_comparisons.csv', index=False)
    manifest.update(verified=True, n_prediction_files=len(checks), result_freeze=False,
        comparison_input_sha256=compared_hashes,
        statistics_note='2000 paired hierarchical bootstrap replicates, seed 20260930; no selected-subset Holm correction; neither bootstrap tails nor CI exclusion are formal preregistered p-values')
    manifest['output_sha256'] = {p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in run.rglob('*') if p.is_file() and p != manifest_path}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(pd.DataFrame(comparisons)[['dataset', 'budget', 'system_A', 'delta_B_minus_A', 'delta_ci_low', 'delta_ci_high']].to_string(index=False))


if __name__ == '__main__':
    finalize()
