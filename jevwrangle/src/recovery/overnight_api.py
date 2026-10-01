"""Budget-constrained supplement fixed before new outcomes, cache-first."""
import concurrent.futures as cf
import copy, hashlib, json, os, random, threading
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from src.clients.llm import LLMClient
from src.clients.systemone import SystemOneClient
from src.clients.ledger import Ledger, BudgetExceededError
from .api_pilot import messages
from .core import probability, llm_probability, noul, score, choice

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/overnight_api_20260930'

def read(path):return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
def key(r):return tuple(r.get(k) for k in ['experiment','dataset','split','pair_id','model','kind'])

def main():
    OUT.mkdir(exist_ok=True)
    # All paid supplements use this single ledger. Original paid work is earlier.
    (OUT/'budget.yaml').write_text('total_usd_hard_limit: 2.95\n',encoding='utf8')
    ledger=Ledger(str(OUT/'ledger.csv'),str(OUT/'budget.yaml'));os.environ['JEV_RUN_ID']=OUT.name
    llms={m:LLMClient(model=m,cache_path=str(BASE/'cache/calls_v2.sqlite'),ledger=ledger) for m in ['gpt-6-luna','deepseek-v4-flash']}
    jev=SystemOneClient(ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite'))
    dm=SystemOneClient(model='decision-model-preview',price_in_per_mtok=0.,official_in_per_mtok=0.,ledger=ledger,cache_path=str(BASE/'cache/calls_v2.sqlite'))
    completed={};dest=OUT/'parsed.jsonl'
    if dest.exists():
        for r in read(dest):completed[key(r)]=r
    legacy={}
    for r in read(BASE/'runs/api_llm_recovery_20260930/parsed.jsonl'):
        if r['status']=='ok':legacy[(r['dataset'],r['split'],r['pair_id'],r['model'],r['kind'])]=r
    jobs=[];protocol={};hashes={}
    def add(identity,pair,questions,state=None):
        if key(identity) in completed:return
        jobs.append((identity,pair,questions,state))
    for ds in ['wa','ag','da','ab']:
        h=yaml.safe_load((BASE/f'questions/{ds}/holistic.yaml').read_text(encoding='utf8'))
        d=yaml.safe_load((BASE/f'questions/{ds}/decomposed.yaml').read_text(encoding='utf8'))
        test=read(BASE/f'data/canonical/{ds}/test.jsonl');pool=read(BASE/f'data/pools/{ds}_pool2000.jsonl')
        subset,_=train_test_split(test,train_size=500,stratify=[p['label'] for p in test],random_state=0)
        selected_ids=set()
        for b in [50,200]:
            for seed in range(3):selected_ids.update(json.loads((BASE/f'data/budgets/{ds}/b{b}_seed{seed}.json').read_text())['pair_ids'])
        protocol[ds]={'Luna_test_ids':[p['pair_id'] for p in subset],'Luna_pool_ids':sorted(selected_ids),
                      'DeepSeek_test_ids':[p['pair_id'] for p in test],'DeepSeek_pool_ids':[p['pair_id'] for p in pool]}
        for model in llms:
            for split,pairs in [('pool',pool),('test',test)]:
                if model=='gpt-6-luna':pairs=[p for p in pairs if p['pair_id'] in (selected_ids if split=='pool' else set(protocol[ds]['Luna_test_ids']))]
                for p in pairs:
                    for kind,q in [('H',{'H1_noul':h['H1_noul']}),('D',d)]:
                        identity=dict(experiment='EM',dataset=ds,split=split,pair_id=p['pair_id'],model=model,kind=kind,label=p['label'])
                        old=legacy.get((ds,split,p['pair_id'],model,kind))
                        if old and key(identity) not in completed:
                            row=dict(**old,experiment='EM',reused_from='api_llm_recovery_20260930');completed[key(row)]=row
                            with dest.open('a',encoding='utf8') as file:file.write(json.dumps(row)+'\n')
                        else:add(identity,p,q)
        for split in ['train','valid','test']:
            for p in read(BASE/f'data/canonical/{ds}/{split}.jsonl'):
                add(dict(experiment='EM',dataset=ds,split=split,pair_id=p['pair_id'],model='decision-model-preview',kind='typed',label=p['label']),p,{**h,**d})
        for p in test:
            rng=random.Random(42);state={k:p[k] for k in ['record_a','record_b']};shuffled={}
            for k,r in state.items():keys=list(r);rng.shuffle(keys);shuffled[k]={x:r[x] for x in keys}
            add(dict(experiment='E5',dataset=ds,split='test',pair_id=p['pair_id'],model='jev-1.13',kind='P2',label=p['label']),p,{'H1_noul':h['H1_noul'],**d},shuffled)
            reverse=copy.deepcopy(h['H1_choice']);reverse['criteria']=dict(reversed(list(reverse['criteria'].items())))
            add(dict(experiment='E5',dataset=ds,split='test',pair_id=p['pair_id'],model='jev-1.13',kind='P5',label=p['label']),p,{'H1_choice':reverse})
        for rel in [f'questions/{ds}/holistic.yaml',f'questions/{ds}/decomposed.yaml',f'data/pools/{ds}_pool2000.jsonl',f'data/canonical/{ds}/test.jsonl']:
            hashes[rel]=hashlib.sha256((BASE/rel).read_bytes()).hexdigest()
    # Schedule free DM concurrently, paid jobs ordered by main scientific value.
    jobs.sort(key=lambda j:(0 if j[0]['model']=='deepseek-v4-flash' else 1 if j[0]['model']=='gpt-6-luna' else 2 if j[0]['model']=='jev-1.13' else 3,
                            j[0]['dataset'],j[0]['split'],j[0]['pair_id'],j[0]['kind']))
    manifest=dict(status='RUNNING_NOT_FROZEN',api_new_hard_cap=3.12,ledger_cap=2.95,
        reserve='0.17 for pilot, unknown-billing calls and concurrency; GPU charged separately',
        amendment='User balance constraint: Luna 500 stratified test pairs/dataset and budgets 50/200 seeds 0-2; DeepSeek original full pool/test. No Luna b1000/full claim.',
        selection='sklearn train_test_split stratified random_state=0, selected before new results',protocol=protocol,input_sha256=hashes,
        prompt_sha256=hashlib.sha256((BASE/'src/recovery/api_pilot.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())['protocol']!=protocol:raise ValueError('Protocol changed during collection')
    path.write_text(json.dumps(manifest,indent=2));(OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    def one(job):
        identity,p,q,state=job
        if identity['model'] in llms:
            r,meta=llms[identity['model']].ask(messages(p,q,identity['kind']=='D'),
                max_tokens=700 if identity['model']=='gpt-6-luna' else 400,
                reasoning_effort='none' if identity['model']=='gpt-6-luna' else None,context=identity)
            content=json.loads(r['choices'][0]['message']['content'])
            values={'p_yes':llm_probability(content)} if identity['kind']=='H' else {qid:probability(content[qid]) for qid in q}
        else:
            client=dm if identity['model']=='decision-model-preview' else jev
            r,meta=client.ask(state or {k:p[k] for k in ['record_a','record_b']},q,context=identity)
            answers=r['answers'];values={qid:(noul(answers[qid]) if definition['type']=='noul' else score(answers[qid]) if definition['type']=='score' else choice(answers[qid])) for qid,definition in q.items()}
        return dict(**identity,status='ok',values=values,meta=meta,response_model=r.get('model'))
    def save():
        status=dict(status=manifest['status'],recorded=len(completed),failed=sum(r['status']!='ok' for r in completed.values()),cost=ledger.summary())
        temp=OUT/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'status.json')
    lock=threading.Lock()
    def collect(work,workers):
        iterator=iter(work);pending={};failed=0
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            def refill():
                while len(pending)<workers:
                    try:j=next(iterator)
                    except StopIteration:return
                    pending[executor.submit(one,j)]=j
            refill();n=0
            while pending:
                ready,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    j=pending.pop(future)
                    try:r=future.result();failed=0
                    except BudgetExceededError:
                        manifest['status']='STOPPED_AT_BUDGET';save();return
                    except Exception as error:
                        r=dict(**j[0],status='failed',values={},error_type=type(error).__name__,error=str(error)[:200]);failed+=1
                    with lock:
                        completed[key(r)]=r
                        with dest.open('a',encoding='utf8') as file:file.write(json.dumps(r,ensure_ascii=False)+'\n')
                        n+=1
                        if n%100==0:save();print('OVERNIGHT_API',n,ledger.summary()['total_paid'],flush=True)
                    if failed>=6:manifest['status']='STOPPED_REPEATED_FAILURE';save();return
                refill()
    paid=[j for j in jobs if j[0]['model']!='decision-model-preview'];free=[j for j in jobs if j[0]['model']=='decision-model-preview']
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        results=[executor.submit(collect,paid,6),executor.submit(collect,free,6)]
        for r in results:r.result()
    if manifest['status']=='RUNNING_NOT_FROZEN':manifest['status']='COLLECTION_COMPLETE_NOT_FROZEN'
    manifest['cost']=ledger.summary();path.write_text(json.dumps(manifest,indent=2));save()

if __name__=='__main__':main()
