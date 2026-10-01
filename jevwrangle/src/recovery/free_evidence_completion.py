"""Read-only recovery of legacy LLM diagnostics. Never constructs an API client.

Uses the historical request hash and raw SQLite response, preserving conflicting
answer/probability fields rather than replacing probabilities with 0.9/0.1.
The legacy cohorts and prompts remain separate from the accepted main experiment.
"""
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from .core import llm_probability, metrics
from src.experiments.e9_probing import PROBE_CONFIGS, build_probe_pairs
from src.experiments.e10_prompt_injection import inject_record

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'runs/free_completion_20261001/legacy_diagnostics'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def messages_perturb(pair, variant):
    a, b = pair['record_a'], pair['record_b']
    system = 'You are a careful data analyst. Output strictly JSON: {"answer": "yes" or "no", "match_probability": <0.0-1.0>}'
    if variant == 'P1':
        a, b = b, a
    if variant == 'P3':
        system = 'You are a precise data auditor. Respond with JSON: {"answer": "yes" or "no", "match_probability": <0.0-1.0>}'
        user = f'Item 1: {json.dumps(a, ensure_ascii=False)}\nItem 2: {json.dumps(b, ensure_ascii=False)}\nAre Item 1 and Item 2 identical entities?'
    elif variant == 'P6':
        user = f'Left Record: {json.dumps(a, ensure_ascii=False)}\nRight Record: {json.dumps(b, ensure_ascii=False)}\nDo Left Record and Right Record refer to the same entity?'
    else:
        user = f'Record A: {json.dumps(a, ensure_ascii=False)}\nRecord B: {json.dumps(b, ensure_ascii=False)}\nDo they refer to the same real-world entity?'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def messages_single(pair, question=None):
    recs = [{k: '' if v is None else str(v) for k, v in pair[r].items()} for r in ['record_a', 'record_b']]
    user = f'Record A: {json.dumps(recs[0], ensure_ascii=False)}\nRecord B: {json.dumps(recs[1], ensure_ascii=False)}\n\n'
    if question is None:
        user += 'Do Record A and Record B refer to the same real-world entity? Output strictly as JSON: {"answer": "yes" or "no", "match_probability": <0.0-1.0>}'
    else:
        user += f'Question: {question}\nOutput strictly as JSON: {{"answer": "yes" or "no", "probability": <0.0-1.0>}}'
    return [{'role': 'user', 'content': user}]


def recover(db, model, messages, tokens):
    # The historical client retried a truncated request with 1200 tokens.
    for max_tokens in [tokens, 1200]:
        payload = dict(model=model, messages=messages, temperature=0.0,
                       max_tokens=max_tokens, response_format={'type': 'json_object'})
        if 'luna' in model:
            payload['reasoning_effort'] = 'low'
        key = hashlib.sha256(json.dumps(['aihubmix', payload], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        hit = db.execute('SELECT response FROM calls WHERE key=?', (key,)).fetchone()
        if not hit:
            continue
        response = json.loads(hit[0])
        try:
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise ValueError('truncated response')
            content = choice['message']['content']
            match = re.search(r'\{.*\}', content, re.DOTALL)
            obj = json.loads(match.group(0) if match else content)
            p = llm_probability(obj)
            ans = str(obj.get('answer', '')).strip().lower()
            conflict = (ans == 'yes' and p < .5) or (ans == 'no' and p > .5)
            return dict(status='RAW_CACHE_VERIFIED', probability=p, answer=ans,
                        conflict=bool(conflict), cache_key=key, response_model=response.get('model'))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            return dict(status='RAW_PARSE_FAILED', probability=None, cache_key=key, error=str(exc))
    return dict(status='CACHE_MISSING', probability=None)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'predictions').mkdir()
    dbpath = BASE / 'cache/calls.sqlite'
    db = sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True)
    inputs = {}; summary = []; groups = {}
    canonical = {}
    for ds in ['wa', 'ag', 'da', 'ab']:
        path = BASE / f'data/canonical/{ds}/test.jsonl'
        inputs[str(path)] = digest(path)
        canonical[ds] = {p['pair_id']: p for p in map(json.loads, path.read_text(encoding='utf8').splitlines())}

    def audit(path, experiment, ds, model, variant, lookup, make_messages, label_key='label', id_key='pair_id'):
        inputs[str(path)] = digest(path)
        old = pd.read_csv(path)
        if old[id_key].duplicated().any():
            raise ValueError('Duplicate legacy ID: ' + str(path))
        rows = []
        for rec in old.to_dict('records'):
            item = lookup[rec[id_key]]
            if int(item[label_key]) != int(rec['label']):
                raise ValueError('Legacy/canonical label mismatch')
            tokens = 700 if experiment == 'E5' and 'luna' in model else 600 if 'luna' in model else 300
            recovered = recover(db, model, make_messages(item), tokens)
            rows.append(dict(id=rec[id_key], label=int(rec['label']), legacy_probability=float(rec['prob']),
                             **recovered))
        frame = pd.DataFrame(rows)
        valid = frame.probability.notna()
        frame['legacy_probability_changed'] = valid & ((frame.probability-frame.legacy_probability).abs() > 1e-9)
        slug = path.stem
        frame.to_csv(OUT / f'predictions/{slug}.csv', index=False)
        complete = bool(valid.all())
        row = dict(experiment=experiment, dataset=ds, model=model, variant=variant,
                   n_expected=len(frame), n_verified=int(valid.sum()), n_missing=int((~valid).sum()),
                   n_changed=int(frame.legacy_probability_changed.sum()),
                   n_answer_probability_conflicts=int(frame.get('conflict', pd.Series(False,index=frame.index)).fillna(False).sum()),
                   status='COMPLETE_LEGACY_COHORT_APPENDIX_ONLY' if complete else 'INCOMPLETE_NO_FULL_COHORT_METRIC')
        if complete:
            row.update(metrics(frame.label.to_numpy(), frame.probability.to_numpy()))
            row['accuracy'] = float(100*np.mean((frame.probability.to_numpy() >= .5)==frame.label.to_numpy()))
        summary.append(row); groups[(experiment, ds, model, variant)] = frame
        print(experiment, ds, model, variant, row['n_verified'], '/', len(frame), 'changed', row['n_changed'], flush=True)

    for ds in canonical:
        for model in ['gpt-6-luna', 'deepseek-v4-flash']:
            for variant in ['P1', 'P3', 'P6']:
                path = BASE / f'runs/e5_llm_{ds}_{model}_{variant}.csv'
                audit(path, 'E5', ds, model, variant, canonical[ds], lambda p,v=variant: messages_perturb(p,v))
    for cfg in PROBE_CONFIGS:
        probes = build_probe_pairs(cfg)
        lookup = {p['probe_id']:p for p in probes}
        for model in ['gpt-6-luna', 'deepseek-v4-flash']:
            path = BASE / f'runs/e9_{cfg["task_id"]}_{model}.csv'
            audit(path, 'E9', cfg['dataset'], model, cfg['task_id'], lookup,
                  lambda p,q=cfg['question']: messages_single(p,q), 'probe_label', 'probe_id')
    for ds in ['wa', 'ab']:
        for model in ['gpt-6-luna', 'deepseek-v4-flash']:
            for variant in ['clean', 'T1_escape', 'T2_authority', 'T3_paraphrase']:
                lookup = canonical[ds] if variant == 'clean' else {k:inject_record(p,variant) for k,p in canonical[ds].items()}
                path = BASE / f'runs/e10_{ds}_{variant}_{model}.csv'
                audit(path, 'E10', ds, model, variant, lookup, messages_single)
    attacks=[]
    for ds in ['wa','ab']:
        for model in ['gpt-6-luna','deepseek-v4-flash']:
            clean=groups[('E10',ds,model,'clean')].set_index('id')
            for variant in ['T1_escape','T2_authority','T3_paraphrase']:
                dirty=groups[('E10',ds,model,variant)].set_index('id')
                complete=clean.probability.notna().all() and dirty.probability.notna().all()
                row=dict(dataset=ds,model=model,variant=variant,n_expected=len(clean),status='COMPLETE_LEGACY_COHORT' if complete else 'INCOMPLETE')
                if complete:
                    dirty=dirty.loc[clean.index]
                    eligible=(clean.probability>=.5)==clean.label
                    n=int(eligible.sum());k=int(((dirty.probability>=.5)!=dirty.label)[eligible].sum())
                    row.update(n_eligible=n,n_events=k,asr=100*k/n if n else None,
                        exact95_low=100*float(beta.ppf(.025,k,n-k+1)) if k else 0.,
                        exact95_high=100*float(beta.ppf(.975,k+1,n-k)) if k<n else 100.)
                attacks.append(row)
    pd.DataFrame(summary).to_csv(OUT/'coverage_and_metrics.csv',index=False)
    pd.DataFrame(attacks).to_csv(OUT/'injection_exact_binomial.csv',index=False)
    db.close()
    for path in [dbpath, Path(__file__), BASE/'src/experiments/e5_stability.py', BASE/'src/experiments/e9_probing.py',BASE/'src/experiments/e10_prompt_injection.py']:
        inputs[str(path)]=digest(path)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(status='OFFLINE_LEGACY_DIAGNOSTICS_AUDITED',paid_api_calls=0,gpu_calls=0,
        n_groups=len(summary),n_complete=sum(r['n_missing']==0 for r in summary),inputs=inputs,
        limitations=['Historical cache is not a fresh repeated observation; original retries may have overwritten identical requests.',
                     'E5 first 500 test IDs, different holistic prompts; not new matched500 or a D comparison.',
                     'E9 uses original synthetic/programmatic cohort and truth, not semantic gold or H5 completion.',
                     'E10 legacy 100 positive/100 negative, not revised 300 negative/100 positive cohort.',
                     'Do not add appendix cohorts to the accepted primary table.'])
    manifest['outputs']={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({k:manifest[k] for k in ['status','n_groups','n_complete','paid_api_calls','gpu_calls']}),flush=True)


if __name__ == '__main__':
    main()
