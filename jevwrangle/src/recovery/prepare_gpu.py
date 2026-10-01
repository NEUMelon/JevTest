"""Generate the agreed 40 Ditto jobs from canonical data and frozen budgets."""
import argparse
import hashlib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
BASE=Path(__file__).resolve().parents[2]


def serialize(p):
    def record(r):
        return ' '.join(f"COL {k} VAL {'' if v is None else str(v)}" for k,v in r.items()).replace('\t',' ').replace('\n',' ').replace('\r',' ')
    return record(p['record_a'])+'\t'+record(p['record_b'])+'\t'+str(p['label'])


def prepare(dest):
    dest.mkdir(parents=True,exist_ok=False);jobs=[];inputs={};outputs={}
    def read(path):
        inputs[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
    def write(folder,name,pairs):
        path=folder/(name+'.txt');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(serialize(p) for p in pairs)+'\n',encoding='utf8')
        (folder/(name+'_ids.json')).write_text(json.dumps([p['pair_id'] for p in pairs]),encoding='utf8')
        outputs[str(path.relative_to(dest))]=dict(n=len(pairs),positive=sum(p['label'] for p in pairs),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for ds in ['wa','ag','da','ab']:
        tr=read(BASE/f'data/canonical/{ds}/train.jsonl');va=read(BASE/f'data/canonical/{ds}/valid.jsonl');te=read(BASE/f'data/canonical/{ds}/test.jsonl')
        lookup={p['pair_id']:p for p in tr}
        assert len(lookup)==len(tr)
        for b in [50,200,1000,'full']:
            for seed in ([0] if b=='full' else range(3)):
                folder=dest/f'{ds}/b{b}_s{seed}'
                if b=='full':train,valid=tr,va
                else:
                    path=BASE/f'data/budgets/{ds}/b{b}_seed{seed}.json';inputs[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    ids=json.loads(path.read_text())['pair_ids'];pairs=[lookup[i] for i in ids]
                    assert len(pairs)==b and len(set(ids))==b
                    train,valid=train_test_split(pairs,test_size=.2,random_state=seed,stratify=[p['label'] for p in pairs])
                    assert len(train)+len(valid)==b
                assert not {p['pair_id'] for p in train}&{p['pair_id'] for p in valid}
                assert not {p['pair_id'] for p in train+valid}&{p['pair_id'] for p in te}
                for name,pairs in [('train',train),('valid',valid),('test',te)]:write(folder,name,pairs)
                jobs.append(dict(dataset=ds,budget=b,seed=seed,input_dir=str(folder.relative_to(dest)),lm='roberta',
                                 batch_size=32,lr=3e-5,n_epochs=15 if b=='full' else 20 if b==1000 else 40,
                                 max_len=256,dk='general',augmentation='drop_col' if ds=='wa' else 'swap',
                                 status='PREPARED_NOT_TRAINED',full_budget_role='port_validation_only' if b=='full' else None))
    assert len(jobs)==40
    manifest=dict(status='PREPARED_NOT_TRAINED',jobs=jobs,inputs=inputs,outputs=outputs,
                  note='Canonical official EM splits retained; benchmark content duplicates disclosed separately; no GPU invocation')
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'GPU_INPUTS_PREPARED {dest} jobs={len(jobs)}')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();prepare(a.output.resolve())
