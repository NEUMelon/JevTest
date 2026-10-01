"""Reviewed scientific figure exports from immutable numeric tables."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'reports/final_delivery_20261001/figures_reviewed'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    master_path=BASE/'reports/overnight_recovery_20261001_reviewed/tables/full_system_summary.csv'
    master=pd.read_csv(master_path);master.budget=master.budget.astype(str)
    cost=pd.read_csv(BASE/'runs/remote_cost_probe_inferera_20260930/matched_task_cost_latency.csv')
    cal=pd.read_csv(BASE/'runs/qwen_diagnostics_20260930/calibration.csv')
    names={'wa':'Walmart–Amazon','ag':'Amazon–Google','da':'DBLP–ACM','ab':'Abt–Buy'}
    colors={'J-D+LR':'#0072B2','J-D+C+LR':'#009E73','M2+best':'#777777','Ditto':'#D55E00','Q-D+LR':'#CC79A7','L2-D+LR':'#E69F00'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    def save(fig,name,note):
        fig.text(.5,-.025,note,ha='center',fontsize=8)
        for suffix in ['pdf','png']:fig.savefig(OUT/f'{name}.{suffix}',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharey=True,layout='constrained');order=['50','200','1000','full']
    for ds,ax in zip(names,axes.flat):
        for system,color in colors.items():
            f=master[(master.dataset==ds)&(master.system==system)&master.budget.isin(order)].copy();f['x']=f.budget.map({b:i for i,b in enumerate(order)});f=f.sort_values('x')
            if len(f):ax.errorbar(f.x,f.f1_mean,yerr=f.f1_std.fillna(0),marker='o',markersize=3,capsize=2,color=color,label=system,lw=1.3)
        ax.set(title=names[ds],xticks=range(4),xticklabels=order,ylim=(0,102),xlabel='Labeled pairs (full Ditto: train + validation)');ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Test F1 (%)');axes[1,0].set_ylabel('Test F1 (%)');axes[0,0].legend(fontsize=7,ncol=2,loc='lower right')
    save(fig,'learning_curves_full_test','Mean ± 1 seed SD, not a confidence interval. Full Ditto is a one-seed port check; Luna uses a separate 500-pair table.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained');loc=np.arange(3)
    for k,task in enumerate(['H','D']):
        f=cost[cost.task==task].set_index('model').loc[['jev-1.13','deepseek-v4-flash','gpt-6-luna']]
        for ax,values in [(axes[0],f.cost_per_1000_decisions),(axes[1],f.latency_p50_ms/1000)]:ax.bar(loc+(k-.5)*.35,values,width=.35,label='Holistic (H)' if task=='H' else 'Decomposed (D)')
    for ax in axes:ax.set_xticks(loc,['Jev','DeepSeek','Luna']);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Estimated USD / 1,000 decisions');axes[1].set_ylabel('Median latency (seconds)')
    fig.suptitle('Same 48 design pairs per model/task, 6 concurrent calls, same AutoDL route',fontsize=9)
    save(fig,'matched_task_cost_latency','Usage × configured unit prices; not an account invoice or an equal-accuracy comparison. Local GPU costs are excluded.')
    fig,axes=plt.subplots(1,4,figsize=(11,3.2),layout='constrained')
    for ds,ax in zip(names,axes):
        for i,(system,color) in enumerate([('J-H1','#0072B2'),('Q-H1','#CC79A7'),('L2-H1','#E69F00')]):
            f=cal[(cal.dataset==ds)&(cal.system==system)&(cal.metric=='ece_equal_mass')]
            for mode,marker,offset in [('raw','o',0),('temperature','s',.15)]:
                z=f[f['mode']==mode].iloc[0];ax.vlines(i+offset,z.ci_low,z.ci_high,color=color);ax.scatter(i+offset,z.estimate,marker=marker,color=color,s=16)
        ax.set(title=names[ds],xticks=[0,1,2],xticklabels=['Jev','Qwen','DS']);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Equal-mass ECE (15 bins)')
    fig.suptitle('Circle: raw; square: temperature fitted on 2,000 labeled pool pairs per dataset',fontsize=9)
    save(fig,'calibration_full_test','Full canonical test. Conditional test-bootstrap intervals; independent panel y-scales. Calibration labels are separate from b=200.')
    fig,axes=plt.subplots(1,4,figsize=(11,3.2),layout='constrained')
    for ds,ax in zip(names,axes):
        for system,color in [('J-D+LR','#0072B2'),('Q-D+LR','#CC79A7'),('L2-D+LR','#E69F00')]:
            f=pd.read_csv(BASE/f'runs/qwen_diagnostics_20260930/curves/{ds}_{system}_b200_s0.csv')
            if f.risk.max()>.16:raise ValueError('Risk range must not conceal observations')
            ax.step(f.coverage,f.risk,where='post',label=system,color=color,lw=1.3)
        ax.set(title=names[ds],xlim=(0,1),ylim=(0,.16),yticks=[0,.05,.10,.15],xlabel='Coverage');ax.grid(alpha=.2)
    axes[0].set_ylabel('Selective error rate');axes[-1].legend(fontsize=7)
    save(fig,'selective_risk_b200_seed0','Fixed b=200, seed 0 models; shared y-range 0–0.16. Conditional accuracy depends on class mix; this is not positive-class recall.')
    (OUT/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    inputs=[master_path,BASE/'runs/remote_cost_probe_inferera_20260930/matched_task_cost_latency.csv',BASE/'runs/qwen_diagnostics_20260930/calibration.csv']
    inputs+=list((BASE/'runs/qwen_diagnostics_20260930/curves').glob('*_b200_s0.csv'))
    (OUT/'manifest.json').write_text(json.dumps(dict(status='EXPORTED_REQUIRES_VISUAL_REVIEW',inputs={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}),indent=2),encoding='utf8')

if __name__=='__main__':main()
