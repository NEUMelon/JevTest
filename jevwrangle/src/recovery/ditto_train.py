"""Auditable Ditto port: official augmentation/DK, native AMP, validation only.

This is a documented port, not an assertion of bit-for-bit Apex reproduction.
Model downloads/installations are explicit preparation tasks. Training requires
a local RoBERTa directory, spaCy en_core_web_lg, and an actual CUDA device.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .core import metrics

BASE = Path(__file__).resolve().parents[2]
UPSTREAM = BASE / 'third_party/ditto_upstream'
COMMIT = '52985564a93fb11308439516d3e17a033d43ec8f'

def drop_col_preserving_separator(tokens, labels):
    starts=[i for i,t in enumerate(tokens) if t=='COL']
    spans=[]
    for i,start in enumerate(starts):
        end=starts[i+1]-1 if i+1<len(starts) else len(tokens)-1
        # DK can insert an ORG marker between SEP and the following COL.
        # The upstream check of only tokens[end] then deletes SEP itself.
        separators=[j for j in range(start,end+1) if tokens[j]=='[SEP]']
        if separators:end=separators[0]-1
        if 0<end-start+1<=8:spans.append((start,end))
    if not spans:return tokens,labels
    start,end=random.choice(spans)
    return tokens[:start]+tokens[end+1:],labels[:start]+labels[end+1:]


def validation_threshold(y, p):
    """Official grid and strict comparison, with deterministic first-best ties."""
    best_score, best_threshold = -1., .5
    for threshold in np.arange(0., 1., .05):
        score = f1_score(y, np.asarray(p) > threshold, zero_division=0)
        if score > best_score:
            best_score, best_threshold = float(score), float(threshold)
    return best_score, best_threshold


def preflight(input_root, slug):
    manifest = json.loads((input_root / 'manifest.json').read_text(encoding='utf8'))
    job = next(j for j in manifest['jobs'] if f"{j['dataset']}_b{j['budget']}_s{j['seed']}" == slug)
    folder = input_root / job['input_dir'].replace('\\', '/')
    files, pairs, ids = {}, {}, {}
    for split in ['train', 'valid', 'test']:
        file = folder / f'{split}.txt'
        rel = file.relative_to(input_root).as_posix()
        expected = manifest['outputs'].get(rel) or manifest['outputs'][rel.replace('/', '\\')]
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise ValueError('Prepared input hash changed')
        rows = [s.split('\t') for s in file.read_text(encoding='utf8').splitlines()]
        if any(len(r) != 3 for r in rows) or len(rows) != expected['n']:
            raise ValueError('Malformed prepared Ditto input')
        pairs[split] = [(a, b, int(y)) for a, b, y in rows]
        ids[split] = json.loads((folder / f'{split}_ids.json').read_text())
        if len(ids[split]) != len(rows) or len(set(ids[split])) != len(rows):
            raise ValueError('Invalid pair IDs')
        canonical_split = 'test' if split == 'test' else 'valid' if job['budget'] == 'full' and split == 'valid' else 'train'
        canonical = {r['pair_id']: r for r in [json.loads(s) for s in (BASE / f"data/canonical/{job['dataset']}/{canonical_split}.jsonl").read_text(encoding='utf8').splitlines()]}
        if any(i not in canonical or canonical[i]['label'] != row[2] for i, row in zip(ids[split], pairs[split])):
            raise ValueError('Prepared labels differ from canonical')
        files[rel] = digest
    if any(set(ids[a]) & set(ids[b]) for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]):
        raise ValueError('Pair ID split overlap')
    if job['budget'] != 'full':
        expected_ids = json.loads((BASE / f"data/budgets/{job['dataset']}/b{job['budget']}_seed{job['seed']}.json").read_text())['pair_ids']
        if len(ids['train']) + len(ids['valid']) != job['budget'] or set(ids['train'] + ids['valid']) != set(expected_ids):
            raise ValueError('Train plus validation exceed or differ from frozen budget')
    return job, files, pairs, ids


def train(input_root, slug, output, model_dir):
    job, files, pairs, ids = preflight(input_root, slug)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; inputs prepared, training NOT executed')
    if not model_dir.is_dir():
        raise ValueError('Provide a downloaded local RoBERTa model directory')
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    import transformers, spacy
    sys.path.insert(0, str(UPSTREAM))
    from ditto_light.augment import Augmenter as UpstreamAugmenter
    from ditto_light.knowledge import GeneralDKInjector
    output.mkdir(parents=True, exist_ok=False)
    seed = job['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    injector = GeneralDKInjector({}, 'general')
    class Augmenter(UpstreamAugmenter):
        def augment(self,tokens,labels,op='del'):
            return drop_col_preserving_separator(tokens,labels) if op=='drop_col' else super().augment(tokens,labels,op)
    augmenter = Augmenter()
    transformed = {}
    dk_db=sqlite3.connect(BASE/'runs/ditto_dk_cache.sqlite')
    dk_db.execute('CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,value TEXT)')
    def transform(value):
        key=hashlib.sha256((COMMIT+'|spacy='+spacy.__version__+'|'+value).encode()).hexdigest()
        cached=dk_db.execute('SELECT value FROM records WHERE key=?',(key,)).fetchone()
        if cached:return cached[0]
        result=injector.transform(value)
        dk_db.execute('INSERT INTO records VALUES (?,?)',(key,result))
        return result
    for split, values in pairs.items():
        # Preserve row order; no labels are used in domain-knowledge preprocessing.
        transformed[split] = [(' '.join(transform(a).split()), ' '.join(transform(b).split()), y) for a, b, y in values]
        (output / f'{split}_dk.txt').write_text('\n'.join(f'{a}\t{b}\t{y}' for a, b, y in transformed[split]) + '\n', encoding='utf8')
        dk_db.commit()
        print('DITTO_DK',slug,split,len(values),flush=True)
    dk_db.close()

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, split): self.split = split
        def __len__(self): return len(transformed[self.split])
        def __getitem__(self, idx):
            a, b, y = transformed[self.split][idx]
            if self.split == 'train':
                aug = augmenter.augment_sent(a + ' [SEP] ' + b, job['augmentation'])
                if aug.count('[SEP]')!=1:raise ValueError('Augmentation changed pair separators')
                # A dropped terminal attribute can leave SEP at a boundary;
                # upstream's space-delimited split then raises ValueError.
                left, right = [s.strip() for s in aug.split('[SEP]',1)]
                return a, b, y, left, right
            return a, b, y

    def collate(batch):
        a, b, y = zip(*[row[:3] for row in batch])
        def encode(left, right):
            return tokenizer(list(left), list(right), truncation=True, max_length=job['max_len'], padding=True, return_tensors='pt')
        clean = encode(a, b); augmented = encode(*zip(*[row[3:] for row in batch])) if len(batch[0]) == 5 else None
        return clean, augmented, torch.tensor(y, dtype=torch.long)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
            self.fc = torch.nn.Linear(self.encoder.config.hidden_size, 2)
        def forward(self, clean, augmented=None):
            encode = lambda x: self.encoder(**{k: v.cuda() for k, v in x.items()}).last_hidden_state[:, 0, :]
            if augmented is None: representation = encode(clean)
            else:
                # One joint encoder pass preserves upstream MixDA dropout batching.
                length = max(clean['input_ids'].shape[1], augmented['input_ids'].shape[1]); joint = {}
                for key in clean:
                    fill = tokenizer.pad_token_id if key == 'input_ids' else 0
                    left = torch.nn.functional.pad(clean[key], (0, length - clean[key].shape[1]), value=fill)
                    right = torch.nn.functional.pad(augmented[key], (0, length - augmented[key].shape[1]), value=fill)
                    joint[key] = torch.cat([left, right])
                both = encode(joint); n = len(clean['input_ids']); lam = np.random.beta(.8, .8)
                representation = both[:n] * lam + both[n:] * (1 - lam)
            return self.fc(representation)

    loaders = {split: torch.utils.data.DataLoader(Dataset(split), batch_size=job['batch_size'],
        shuffle=split == 'train', num_workers=0, collate_fn=collate) for split in pairs}
    model = Model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=job['lr'], eps=1e-6, weight_decay=0.)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(loaders['train']) * job['n_epochs'])
    scaler = torch.amp.GradScaler('cuda'); best, best_epoch, best_threshold = -1., None, None
    meta = dict(status='TRAINING_NOT_FROZEN', job=job, input_sha256=files, upstream_repository='https://github.com/megagonlabs/ditto',
        upstream_commit=COMMIT, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_files_sha256={p.relative_to(model_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.rglob('*') if p.is_file()},
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, spacy=spacy.__version__),
        device=torch.cuda.get_device_name(0), port_changes=['native CUDA AMP replaces Apex O2', 'correct tokenizer pad ID and attention masks',
        'scheduler uses actual ceiling batch count', 'preserve SEP at boundary after drop_col; permit empty augmented side',
        'content-addressed label-free DK cache', 'normalize DK whitespace',
        'fix upstream drop_col separator deletion when DK inserts ORG between SEP and COL; cap drop span before SEP',
        'test evaluated only after best validation checkpoint is fixed'],
        reproduction_status='PORT_NOT_YET_VALIDATED_AGAINST_FULL_BUDGET_REFERENCES')
    (output / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    (output / 'executed_source.py').write_bytes(Path(__file__).read_bytes())
    for rel in ['ditto_light/augment.py', 'ditto_light/knowledge.py', 'LICENSE']:
        target = output / 'source_snapshot' / rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((UPSTREAM / rel).read_bytes())

    def predict(split):
        model.eval(); result = []
        with torch.no_grad():
            for clean, _, labels in loaders[split]:
                with torch.autocast('cuda', dtype=torch.float16): logits = model(clean)
                result.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
        return np.array(result)
    started = time.perf_counter(); log = []
    for epoch in range(1, job['n_epochs'] + 1):
        model.train(); losses = []
        for clean, augmented, labels in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(clean, augmented), labels.cuda())
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step(); losses.append(float(loss.detach()))
        p = predict('valid'); y = np.array([r[2] for r in pairs['valid']]); f1, threshold = validation_threshold(y, p)
        pd.DataFrame(dict(pair_id=ids['valid'], label=y, probability=p)).to_csv(output / f'valid_epoch_{epoch}.csv', index=False)
        log.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), valid_f1=100*f1, threshold=threshold))
        if f1 > best:
            best, best_epoch, best_threshold = f1, epoch, threshold
            torch.save(dict(model=model.state_dict(), epoch=epoch, threshold=threshold, valid_f1=f1), output / 'best_model.pt')
        pd.DataFrame(log).to_csv(output / 'training_log.csv', index=False)
        print(f'DITTO {slug} epoch={epoch} valid_f1={100*f1:.3f}', flush=True)
    checkpoint = torch.load(output / 'best_model.pt', map_location='cuda', weights_only=True); model.load_state_dict(checkpoint['model'])
    p = predict('test'); y = np.array([r[2] for r in pairs['test']]); strict = np.nextafter(best_threshold, np.inf)
    pd.DataFrame(dict(pair_id=ids['test'], label=y, probability=p, prediction=(p > best_threshold).astype(int),
        threshold=strict)).to_csv(output / 'test_predictions.csv', index=False)
    meta.update(status='TRAINED_NOT_FROZEN', result_freeze=False, best_epoch=best_epoch, threshold=strict,
        elapsed_seconds=time.perf_counter()-started, test_metrics=metrics(y, p, strict))
    meta['output_sha256'] = {f.relative_to(output).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in output.rglob('*') if f.is_file() and f.name != 'manifest.json'}
    (output / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', type=Path, default=BASE/'data/gpu_recovery_20260930')
    ap.add_argument('--job', required=True); ap.add_argument('--output', type=Path); ap.add_argument('--model-dir', type=Path); ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    if a.preflight:
        j, h, p, i = preflight(a.inputs.resolve(), a.job); print(json.dumps(dict(job=j, split_sizes={k:len(v) for k,v in p.items()}, status='INPUTS_VERIFIED_NOT_TRAINED'), indent=2))
    else:
        if not a.output or not a.model_dir: ap.error('--output and --model-dir are required for training')
        train(a.inputs.resolve(), a.job, a.output.resolve(), a.model_dir.resolve())
