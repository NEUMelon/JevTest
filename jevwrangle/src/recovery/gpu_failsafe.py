"""Cost failsafe for this authorized overnight run; never release the instance."""
import json,subprocess,time,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    started=time.time();idle=0;deadline=started+10*3600
    path=ROOT/'runs/gpu_failsafe_status.json'
    while True:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        try:memory,util=[float(v.strip()) for v in result.stdout.split(',')]
        except ValueError:memory,util=-1,-1
        idle=idle+60 if 0<=memory<1024 and util==0 else 0
        status=dict(started=started,deadline=deadline,time=time.time(),idle_seconds=idle,memory_mib=memory,utilization=util,status='MONITORING')
        path.write_text(json.dumps(status,indent=2))
        if idle>=1800 or (time.time()>=deadline and idle>=600):
            status['status']='COST_FAILSAFE_SHUTDOWN';status['reason']='30min no allocated GPU work' if idle>=1800 else 'overnight window exceeded and GPU idle for 10min'
            path.write_text(json.dumps(status,indent=2))
            # Results survive shutdown on the data disk; preserve diagnostics in
            # one critical archive too. Large checkpoints are backed up by the
            # local guardian and are deliberately not re-compressed here.
            with tarfile.open(ROOT/'runs/overnight_critical_failsafe.tar.gz','w:gz') as tar:
                for p in (ROOT/'runs').rglob('*'):
                    if p.is_file() and p.name not in ['best_model.pt','overnight_critical_failsafe.tar.gz']:
                        tar.add(p,arcname=p.relative_to(ROOT))
            subprocess.run(['/usr/bin/shutdown']);return
        time.sleep(60)
if __name__=='__main__':main()
