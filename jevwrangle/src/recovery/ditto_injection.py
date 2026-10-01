"""Inference-only attacks on validation-selected full Ditto checkpoints."""
import hashlib,json,sys,time
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoModel,AutoTokenizer
from .qwen_collect import read
from .prepare_gpu import serialize
BASE=Path(__file__).resolve().parents[2]

def main():
    out=BASE/'runs/ditto_injection_20260930';out.mkdir(exist_ok=True)
    model_dir=Path('/root/autodl-tmp/models/roberta-base')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    sys.path.insert(0,str(BASE/'third_party/ditto_upstream'))
    from ditto_light.knowledge import GeneralDKInjector
    injector=GeneralDKInjector({},'general')
    queue=json.loads((BASE/'runs/gpu_overnight_20260930/status.json').read_text())
    prompts_=read(BASE/'runs/qwen_recovery_20260930/prompts.jsonl')
    observations=[];inputs={};started=time.perf_counter()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=AutoModel.from_pretrained(model_dir,local_files_only=True)
            self.fc=torch.nn.Linear(self.encoder.config.hidden_size,2)
        def forward(self,inputs):return self.fc(self.encoder(**inputs).last_hidden_state[:,0,:])
    for ds in ['wa','ab']:
        info=queue['jobs'][ds+'_bfull_s0'];folder=Path(info.get('output') or Path(info['manifest']).parent)
        manifest=json.loads((folder/'manifest.json').read_text());checkpoint=folder/'best_model.pt'
        expected=manifest['output_sha256']['best_model.pt']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected:raise ValueError('Ditto checkpoint changed')
        inputs[str(checkpoint)]=expected
        model=Model().cuda().eval();state=torch.load(checkpoint,map_location='cuda',weights_only=True);model.load_state_dict(state['model'])
        threshold=state['threshold'];test={p['pair_id']:p for p in read(BASE/f'data/canonical/{ds}/test.jsonl')}
        attacks=[p for p in prompts_ if p['experiment']=='E10' and p['dataset']==ds and p['qid']=='p_yes']
        variants=[];seen=set()
        for p in attacks:
            pair=test[p['pair_id']]
            if p['pair_id'] not in seen:
                variants.append(dict(pair_id=p['pair_id'],label=pair['label'],variant='clean',state={k:pair[k] for k in ['record_a','record_b']}));seen.add(p['pair_id'])
            data=p['messages'][1]['content'].split('Record data (JSON):\n',1)[1].split('\n\nQuestion: ',1)[0]
            variants.append(dict(pair_id=p['pair_id'],label=pair['label'],variant=p['perturbation'],state=json.loads(data)))
        transformed=[]
        for row in variants:
            pair=dict(**row['state'],label=row['label']);a,b,_=serialize(pair).split('\t')
            a,b=' '.join(injector.transform(a).split()),' '.join(injector.transform(b).split())
            length=len(tokenizer.encode(a,b,truncation=False));transformed.append((row,a,b,length))
        with torch.inference_mode():
            for start in range(0,len(transformed),32):
                batch=transformed[start:start+32];encoded=tokenizer([a for _,a,_,_ in batch],[b for _,_,b,_ in batch],
                    truncation=True,max_length=256,padding=True,return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.float16):p=model(encoded).float().softmax(1)[:,1].cpu().tolist()
                for (row,a,b,length),value in zip(batch,p):
                    observations.append(dict(dataset=ds,pair_id=row['pair_id'],label=row['label'],variant=row['variant'],
                        probability=value,prediction=int(value>threshold),threshold=threshold,untruncated_tokens=length,
                        truncated=length>256,serialized_dk_sha256=hashlib.sha256((a+'\t'+b).encode()).hexdigest()))
        del model;torch.cuda.empty_cache()
    frame=pd.DataFrame(observations);frame.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for ds in ['wa','ab']:
        group=frame[frame.dataset==ds];clean=group[group.variant=='clean'].set_index('pair_id')
        for variant in ['T1','T2','T3']:
            dirty=group[group.variant==variant].set_index('pair_id').loc[clean.index]
            eligible=clean.prediction.values==clean.label.values;success=dirty.prediction.values==1-clean.label.values
            n=int(eligible.sum());rows.append(dict(dataset=ds,variant=variant,n=len(clean),eligible=n,
                attack_success_rate=float(success[eligible].mean()) if n else None,
                n_success=int(success[eligible].sum()),fraction_truncated=float(dirty.truncated.mean())))
    pd.DataFrame(rows).to_csv(out/'attack_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_NOT_FROZEN',elapsed_seconds=time.perf_counter()-started,
        checkpoint_sha256=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_selection='Full checkpoint already fixed by clean validation; no refitting on injected data',
        limitations='max_length256 can truncate appended attack; rates must be read with truncation fraction',
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}),indent=2))
    (out/'executed_source.py').write_bytes(Path(__file__).read_bytes())

if __name__=='__main__':main()
