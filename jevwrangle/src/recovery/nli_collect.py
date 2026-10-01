"""Frozen BART MNLI baseline; BF16 local forward and exact entail/contrad ratio."""
import hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from .qwen_collect import read
BASE=Path(__file__).resolve().parents[2]

def main():
    folder=Path('/root/autodl-tmp/models/bart-large-mnli')
    out=BASE/'runs/nli_recovery_20260930';out.mkdir(exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True,torch_dtype=torch.bfloat16).cuda().eval()
    classes={str(name).lower():int(idx) for name,idx in model.config.label2id.items()}
    contradiction=next(v for k,v in classes.items() if 'contrad' in k);entailment=next(v for k,v in classes.items() if 'entail' in k)
    started=time.perf_counter();observations=[];hashes={}
    with torch.inference_mode(),(out/'parsed.jsonl').open('w') as file:
        for ds in ['wa','ag','da','ab']:
            entity='publication' if ds=='da' else 'product'
            for split,path in [('pool',BASE/f'data/pools/{ds}_pool2000.jsonl'),('test',BASE/f'data/canonical/{ds}/test.jsonl')]:
                hashes[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest();rows=read(path)
                for start in range(0,len(rows),16):
                    batch=rows[start:start+16]
                    premises=['Record A: '+json.dumps(p['record_a'],ensure_ascii=False)+'. Record B: '+json.dumps(p['record_b'],ensure_ascii=False)+'.' for p in batch]
                    hypotheses=[f'Record A and Record B describe the same {entity}.']*len(batch)
                    inputs=tokenizer(premises,hypotheses,padding=True,truncation='only_first',max_length=1024,return_tensors='pt').to('cuda')
                    logits=model(**inputs).logits.float();p=logits[:,[contradiction,entailment]].softmax(1)[:,1].cpu().tolist()
                    for row,value in zip(batch,p):
                        file.write(json.dumps(dict(dataset=ds,split=split,pair_id=row['pair_id'],label=row['label'],model='bart-large-mnli',kind='H',status='ok',values={'p_yes':value}))+'\n')
                    file.flush()
                print('NLI',ds,split,len(rows),flush=True)
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    meta=dict(status='COMPLETE_NOT_FROZEN',elapsed_seconds=time.perf_counter()-started,input_sha256=hashes,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model='facebook/bart-large-mnli',
              model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*') if p.is_file()},
              truncation='only_first max_length=1024 includes hypothesis',dtype='bfloat16',device=torch.cuda.get_device_name(0),
              outputs={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['parsed.jsonl','executed_source.py']})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
