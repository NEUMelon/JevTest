"""Export correct full Abt-Buy catalogs with stable derived IDs.

These IDs are content hashes, not claimed to be upstream table IDs. Every split
is reconstructed from canonical records and checked, without overwriting legacy
tables or changing the official benchmark's pairs/labels.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parents[2]


def main():
    out=BASE/'data/raw_tables_recovered/ab';out.mkdir(parents=True,exist_ok=False)
    catalogs={'A':{},'B':{}};splits={};inputs={}
    for split in ['train','valid','test']:
        path=BASE/f'data/canonical/ab/{split}.jsonl';inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        pairs=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()];rows=[]
        for p in pairs:
            ids=[]
            for side,field in [('A','record_a'),('B','record_b')]:
                record=p[field];key=side+'-'+hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                assert key not in catalogs[side] or catalogs[side][key]==record
                catalogs[side][key]=record;ids.append(key)
            rows.append(dict(pair_id=p['pair_id'],ltable_id=ids[0],rtable_id=ids[1],label=p['label']))
            assert catalogs['A'][ids[0]]==p['record_a'] and catalogs['B'][ids[1]]==p['record_b']
        splits[split]=rows;pd.DataFrame(rows).to_csv(out/f'{split}.csv',index=False)
    for side,records in catalogs.items():
        pd.DataFrame([dict(id=key,**record) for key,record in records.items()]).to_csv(out/f'table{side}.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(status='JOIN_VALIDATED',id_scheme='SHA256 content-derived IDs; not upstream original table IDs',
        inputs=inputs,n_tableA=len(catalogs['A']),n_tableB=len(catalogs['B']),n_pairs={s:len(r) for s,r in splits.items()},
        note='Legacy broken raw_tables/ab retained for audit; canonical pair IDs, splits and labels unchanged'),indent=2),encoding='utf8')
    print('ABT_TABLES_RECOVERED',out,{s:len(r) for s,r in splits.items()})


if __name__=='__main__':main()
