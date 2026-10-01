"""Sequential GPU queue. Durable per-job status, no invented missing results."""
import json, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'runs/gpu_overnight_20260930';out.mkdir(exist_ok=True)
    jobs=json.loads((ROOT/'data/gpu_recovery_20260930/manifest.json').read_text())['jobs']
    # Full-budget WA is a port check; other full checks are retained explicitly.
    jobs.sort(key=lambda j:(0 if j['dataset']=='wa' and j['budget']=='full' else 2 if j['budget']=='full' else 1,
                           2000 if j['budget']=='full' else j['budget'],j['dataset'],j['seed']))
    status={'started':time.time(),'jobs':{},'status':'RUNNING_NOT_FROZEN'}
    path=out/'status.json'
    if path.exists():status=json.loads(path.read_text())
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(path)
    for j in jobs:
        slug=f"{j['dataset']}_b{j['budget']}_s{j['seed']}";dest=ROOT/'runs/ditto_recovery_20260930'/slug
        previous=status['jobs'].get(slug,{})
        prior=previous.get('output') or (str(Path(previous['manifest']).parent) if previous.get('manifest') else None)
        if previous.get('status')=='COMPLETE' and prior:
            saved=Path(prior)/'manifest.json'
            if saved.exists() and json.loads(saved.read_text()).get('status')=='TRAINED_NOT_FROZEN':
                # A successful retry may live under an immutable attempt path.
                # Do not train it again just because the original path failed.
                continue
        manifest=dest/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()).get('status')=='TRAINED_NOT_FROZEN':
            status['jobs'][slug]={'status':'COMPLETE','manifest':str(manifest)};save();continue
        if dest.exists():
            # Keep incomplete attempt intact; never overwrite its observations.
            dest=dest.with_name(slug+'_attempt_'+str(int(time.time())))
        dest.parent.mkdir(exist_ok=True)
        status['current_job']=slug;status['jobs'][slug]={'status':'RUNNING','started':time.time(),'output':str(dest)};save()
        command=[sys.executable,'-m','src.recovery.ditto_train','--job',slug,'--output',str(dest),
                 '--model-dir','/root/autodl-tmp/models/roberta-base']
        with (out/(slug+'.log')).open('a') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status['jobs'][slug].update(status='COMPLETE' if result.returncode==0 else 'FAILED',
                                    returncode=result.returncode,finished=time.time());save()
        if result.returncode!=0:
            status['status']='STOPPED_JOB_FAILED';save();return
    status.update(status='DITTO_40_COMPLETE_WAITING_QWEN',finished=time.time());save()
    # Qwen command is a separate checked-in script, only launched when prepared.
    ready=out/'QWEN_READY'
    while not ready.exists():time.sleep(15)
    status['status']='QWEN_RUNNING';save()
    with (out/'qwen.log').open('a') as log:
        result=subprocess.run([sys.executable,'-m','src.recovery.qwen_collect','--output',str(ROOT/'runs/qwen_recovery_20260930')],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    status.update(status='GPU_COMPUTE_COMPLETE_REQUIRES_DOWNLOAD' if result.returncode==0 else 'QWEN_FAILED',
                  qwen_returncode=result.returncode,finished=time.time());save()

if __name__=='__main__':main()
