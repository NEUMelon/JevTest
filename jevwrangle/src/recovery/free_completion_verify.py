"""Independent offline verification; no API or remote client is imported."""
import csv
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / 'runs/free_completion_20261001'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    with p.open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def write(p, rows):
    with p.open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cm(y, pred):
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, pred))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                accuracy=100 * (tp + tn) / len(y))


def verify_manifest(folder, files):
    m = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
    for name, expected in m['inputs'].items():
        p = Path(name)
        assert sha(p) == expected, str(p)
        files[str(p)] = expected
    for name, expected in m['outputs'].items():
        p = folder / name
        assert sha(p) == expected, str(p)
        files[str(p)] = expected
    files[str(folder / 'manifest.json')] = sha(folder / 'manifest.json')


def legacy():
    out = ROOT / 'legacy_independent_readouts'
    out.mkdir(exist_ok=False)
    src = ROOT / 'legacy_diagnostics'
    files = {}
    verify_manifest(src, files)
    db = sqlite3.connect((BASE / 'cache/calls.sqlite').as_uri() + '?mode=ro', uri=True)
    summaries, groups = [], {}
    for meta in read(src / 'coverage_and_metrics.csv'):
        exp, ds, model, variant = (meta[k] for k in ['experiment', 'dataset', 'model', 'variant'])
        if exp == 'E5':
            slug = f'e5_llm_{ds}_{model}_{variant}'
        elif exp == 'E9':
            slug = f'e9_{variant}_{model}'
        else:
            slug = f'e10_{ds}_{variant}_{model}'
        rows = read(src / f'predictions/{slug}.csv')
        assert len(rows) == int(meta['n_expected'])
        assert len(set(r['id'] for r in rows)) == len(rows)
        labels, raw, answer = [], [], []
        n_ambiguous_probability_field = 0
        for r in rows:
            hit = db.execute('SELECT response FROM calls WHERE key=?', (r['cache_key'],)).fetchone()
            assert hit is not None
            response = json.loads(hit[0])
            choice = response['choices'][0]
            assert choice.get('finish_reason') != 'length'
            text = choice['message']['content']
            match = re.search(r'\{.*\}', text, re.S)
            obj = json.loads(match.group(0) if match else text)
            key = next(k for k in ['match_probability', 'p_yes', 'probability', 'prob'] if obj.get(k) is not None)
            p = float(obj[key])
            assert math.isfinite(p) and 0 <= p <= 1
            assert abs(p - float(r['probability'])) < 1e-12
            ans = str(obj.get('answer', '')).strip().lower()
            assert ans == r['answer']
            n_ambiguous_probability_field += key in ['probability', 'prob']
            labels.append(int(r['label']))
            raw.append(int(p >= .5))
            answer.append({'yes': 1, 'no': 0}.get(ans))
        counts = cm(labels, raw)
        for k in ['tp', 'fp', 'fn', 'tn', 'f1', 'accuracy']:
            assert abs(counts[k] - float(meta[k])) < 1e-8, (slug, k)
        unknown = sum(p is None for p in answer)
        row = dict(experiment=exp, dataset=ds, model=model, variant=variant,
                   n=len(rows), n_unknown_answers=unknown,
                   raw_field_semantics='UNSPECIFIED_YES_PROBABILITY' if n_ambiguous_probability_field else 'NAMED_MATCH_PROBABILITY_WITH_POSSIBLE_CONFLICT',
                   n_ambiguous_probability_field=n_ambiguous_probability_field,
                   n_answer_probability_conflicts=sum(a is not None and a != b for a, b in zip(answer, raw)),
                   raw_field_at_05_f1=counts['f1'],
                   answer_f1=cm(labels, answer)['f1'] if unknown == 0 else None,
                   answer_accuracy=cm(labels, answer)['accuracy'] if unknown == 0 else None,
                   interpretation='POST_HOC_SEPARATE_READOUTS_APPENDIX_ONLY_NOT_CALIBRATED_MAIN_COMPARISON')
        summaries.append(row)
        groups[(exp, ds, model, variant)] = (rows, labels, answer)
    db.close()
    from scipy.stats import beta
    attacks = []
    for ds in ['wa', 'ab']:
        for model in ['gpt-6-luna', 'deepseek-v4-flash']:
            clean, truth, pred = groups[('E10', ds, model, 'clean')]
            ids = [r['id'] for r in clean]
            assert None not in pred
            eligible = [a == b for a, b in zip(truth, pred)]
            for variant in ['T1_escape', 'T2_authority', 'T3_paraphrase']:
                dirty, yt, pt = groups[('E10', ds, model, variant)]
                assert [r['id'] for r in dirty] == ids and yt == truth and None not in pt
                n = sum(eligible)
                k = sum(e and a != b for e, a, b in zip(eligible, yt, pt))
                attacks.append(dict(dataset=ds, model=model, variant=variant, readout='ANSWER_POST_HOC',
                    n_cohort=len(ids), n_eligible=n, n_events=k, asr=100*k/n if n else None,
                    exact95_low=100*float(beta.ppf(.025, k, n-k+1)) if k else 0.,
                    exact95_high=100*float(beta.ppf(.975, k+1, n-k)) if k<n else 100.))
    write(out / 'separate_readouts.csv', summaries)
    write(out / 'answer_injection_exact_binomial.csv', attacks)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    manifest = dict(status='INDEPENDENT_RAW_CACHE_AND_CONFUSION_CHECK_PASSED', n_groups=len(summaries),
                    n_rows=sum(r['n'] for r in summaries), paid_api_calls=0, gpu_calls=0,
                    n_unknown_answers=sum(r['n_unknown_answers'] for r in summaries),
                    inputs=files,
                    limitations=['Probability and answer are distinct post-hoc readouts, not candidates selected by test F1.',
                                 'E9 generic probability may represent confidence in the chosen answer; no P(Yes) claim.',
                                 'Historical diagnostic cohorts do not replace the accepted new main protocol.'])
    manifest['outputs'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps({k: manifest[k] for k in ['status', 'n_groups', 'n_rows', 'n_unknown_answers']}))


def baseline():
    import numpy as np
    from .statistics import paired_bootstrap
    src = ROOT / 'magellan_author_library'
    assert (src / 'manifest.json').exists(), 'CPU baseline still running'
    out = ROOT / 'baseline_independent_verification'
    out.mkdir(exist_ok=False)
    files = {}
    verify_manifest(src, files)
    measurements = read(src / 'metrics_by_seed.csv')
    assert len(measurements) == 224
    seen, checks, comparisons = set(), [], []
    for r in measurements:
        ds, system, budget, seed = (r[k] for k in ['dataset', 'system', 'budget', 'seed'])
        key = (ds, system, budget, seed)
        assert key not in seen
        seen.add(key)
        slug = f'{ds}_{system}_b{budget}_s{seed}'
        rows = read(src / f'predictions/{slug}.csv')
        canonical = [json.loads(s) for s in (BASE / f'data/canonical/{ds}/test.jsonl').read_text(encoding='utf8').splitlines()]
        assert [r['pair_id'] for r in rows] == [p['pair_id'] for p in canonical]
        assert [int(r['label']) for r in rows] == [int(p['label']) for p in canonical]
        probs = [float(p['probability']) for p in rows]
        t = float(r['threshold'])
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
        pred = [int(p >= t) for p in probs]
        assert pred == [int(p['prediction']) for p in rows]
        assert all(float(p['threshold']) == t for p in rows)
        counts = cm([p['label'] for p in canonical], pred)
        for k in ['tp', 'fp', 'fn', 'tn', 'f1']:
            assert abs(counts[k] - float(r[k])) < 1e-8
        oof = read(src / f'oof/{slug}.csv')
        if budget == 'full':
            pool = [json.loads(s) for s in (BASE / f'data/canonical/{ds}/train.jsonl').read_text(encoding='utf8').splitlines()]
            wanted = {p['pair_id'] for p in pool}
        else:
            pool = [json.loads(s) for s in (BASE / f'data/pools/{ds}_pool2000.jsonl').read_text(encoding='utf8').splitlines()]
            wanted = set(json.loads((BASE / f'data/budgets/{ds}/b{budget}_seed{seed}.json').read_text(encoding='utf8'))['pair_ids'])
            assert len(wanted) == int(budget)
        assert len(oof) == len(wanted) and {p['pair_id'] for p in oof} == wanted
        lookup = {p['pair_id']: int(p['label']) for p in pool}
        assert all(int(p['label']) == lookup[p['pair_id']] for p in oof)
        oy = [int(p['label']) for p in oof]
        op = [float(p['oof_probability']) for p in oof]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in op)
        # Independent brute force verifies the training-only OOF threshold and tie rule.
        # Vectorized cumulative count for large full datasets avoids quadratic work.
        order = np.argsort(op, kind='stable')
        vals, first = np.unique(np.asarray(op)[order], return_index=True)
        yy = np.asarray(oy)[order]
        prefix = np.r_[0, np.cumsum(yy)]
        score = 2*(sum(oy)-prefix[first])/(len(oy)-first+sum(oy))
        assert abs(float(vals[np.argmax(score)]) - t) < 1e-12
        checks.append(dict(dataset=ds, system=system, budget=budget, seed=seed, n_test=len(rows),
                           n_labeled=len(oof), status='ID_LABEL_PROBABILITY_THRESHOLD_CM_AND_OOF_PASSED'))
    for ds in ['wa', 'ag', 'da', 'ab']:
        for budget in [50, 200]:
            for system in ['M3-author+LR', 'M3-author+best']:
                pairs = []
                for seed in range(10):
                    a_path = src / f'predictions/{ds}_{system}_b{budget}_s{seed}.csv'
                    b_path = BASE / f'runs/recovery_repaired_20260930/predictions/{ds}_J-D+LR_b{budget}_s{seed}.csv'
                    a, b = read(a_path), read(b_path)
                    assert [p['pair_id'] for p in a] == [p['pair_id'] for p in b]
                    assert [p['label'] for p in a] == [p['label'] for p in b]
                    files[str(b_path)] = sha(b_path)
                    pairs.append(([int(p['label']) for p in a], [int(p['prediction']) for p in a], [int(p['prediction']) for p in b]))
                comparisons.append(dict(dataset=ds, budget=budget, system_A=system, system_B='J-D+LR',
                    **paired_bootstrap(pairs), inference='POST_AUDIT_DESCRIPTIVE_CI_NOT_COMPLETE_HOLM_FAMILY'))
    write(out / 'group_checks.csv', checks)
    write(out / 'paired_bootstrap.csv', comparisons)
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    files[str(BASE / 'src/recovery/statistics.py')] = sha(BASE / 'src/recovery/statistics.py')
    manifest = dict(status='224_BASELINE_GROUPS_AND_16_PAIRED_COMPARISONS_VERIFIED', n_groups=224,
                    n_comparisons=16, paid_api_calls=0, gpu_calls=0, inputs=files)
    manifest['outputs'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps({k: manifest[k] for k in ['status', 'n_groups', 'n_comparisons']}))


if __name__ == '__main__':
    import sys
    {'legacy': legacy, 'baseline': baseline}[sys.argv[1]]()
