"""Continue CPU analysis after automatic GPU backup/shutdown; no API spending."""
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import metrics
from .qwen_diagnostics import main as diagnostics
from .flights_row_budgets import main as flights
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parent
OUT=BASE/'reports/overnight_recovery_20260930'

def markdown(frame):
    columns=list(frame.columns)
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for values in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join('—' if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v))
            else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(rows)

def main():
    synthesis=BASE/'runs/overnight_synthesis_20260930'
    while not (synthesis/'status.json').exists():time.sleep(30)
    synthesis_status=json.loads((synthesis/'status.json').read_text(encoding='utf8'))
    if synthesis_status.get('status')!='AGGREGATED_NOT_FROZEN' or synthesis_status.get('n_ditto')!=40:
        raise ValueError('Synthesis has not passed its completion gates')
    OUT.mkdir(exist_ok=False);(OUT/'figures').mkdir();(OUT/'tables').mkdir()
    status_path=OUT/'status.json'
    status_path.write_text(json.dumps(dict(status='CPU_ANALYSIS_RUNNING_NOT_FROZEN')))
    diagnostic=BASE/'runs/qwen_diagnostics_20260930'
    if not diagnostic.exists():diagnostics(diagnostic)
    def require_manifest(folder,expected):
        p=folder/'manifest.json'
        if not p.exists() or json.loads(p.read_text(encoding='utf8')).get('status')!=expected:
            raise ValueError('Existing partial stage requires a preserved repair snapshot: '+str(folder))
    require_manifest(diagnostic,'DIAGNOSTICS_COMPLETE_NOT_FROZEN')
    ed=BASE/'runs/flights_row_recovery_20260930'
    if not ed.exists():flights(ed)
    require_manifest(ed,'COMPLETE_GROUPS_ONLY_NOT_FROZEN')
    from .error_evidence import main as error_analysis
    error_out=BASE/'runs/error_evidence_missing_normalized_20260930'
    if not error_out.exists():error_analysis(error_out)
    require_manifest(error_out,'AUTOMATIC_SLICES_VERIFIED_HUMAN_REVIEW_PENDING')
    from .content_sensitivity import main as content_analysis
    sensitivity_out=BASE/'runs/em_content_sensitivity_20261001_reviewed'
    if not sensitivity_out.exists():content_analysis(sensitivity_out)
    require_manifest(sensitivity_out,'DIAGNOSTIC_SENSITIVITY_COMPLETE_NOT_PRIMARY')
    inputs={}
    def csv(path):
        inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest();return pd.read_csv(path)
    def jread(path):
        inputs[path.relative_to(BASE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text(encoding='utf8'))
    jev=csv(BASE/'runs/recovery_repaired_20260930/tables/em_summary.csv')
    jev=jev[jev.system.str.startswith('J-') | (jev.system=='C+LR')].copy()
    strings=csv(BASE/'runs/magellan_v2_20260930/summary.csv')
    ditto=csv(synthesis/'ditto_summary.csv');qwen=csv(synthesis/'qwen_evaluation/summary.csv')
    nli=csv(synthesis/'nli_evaluation/summary.csv')
    llm=csv(BASE/'runs/overnight_finalization_20260930/llm_evaluation/summary.csv')
    frames=[]
    for f,source in [(jev,'Rebuilt Jev'),(strings,'Custom string features'),(ditto,'40 Ditto GPU jobs'),(qwen,'Qwen HF token probabilities'),(nli,'BART MNLI appendix'),(llm,'API repaired protocol')]:
        f=f.copy();f['source']=source;f['test_protocol']=np.where(f.system.str.startswith('L1'),'Fixed 500 stratified pairs','Full canonical test');frames.append(f)
    master=pd.concat(frames,ignore_index=True);master.budget=master.budget.astype(str)
    if master.duplicated(['dataset','system','budget']).any():raise ValueError('Historical/primary system identities must not be merged')
    master.to_csv(OUT/'tables/full_system_summary.csv',index=False)
    paired=csv(synthesis/'paired_comparisons/paired_system_comparisons.csv')
    h2=csv(synthesis/'paired_comparisons/h2_interaction.csv')
    h1=csv(BASE/'runs/recovery_repaired_20260930/statistics_crossed_metadata_20260930/paired_hierarchical_bootstrap.csv')
    cost=csv(BASE/'runs/remote_cost_probe_inferera_20260930/matched_task_cost_latency.csv')
    calibration=csv(diagnostic/'calibration.csv');temperature=csv(diagnostic/'temperature.csv')
    attack=csv(diagnostic/'paired_attack_differences.csv');probe=csv(diagnostic/'numeric_probes.csv')
    hybrid_comparison=paired[(paired.system_A=='J-D+LR')&(paired.system_B=='J-D+C+LR')&(paired.budget.astype(str)=='200')]
    hybrid_comparison=hybrid_comparison[['dataset','n_seeds','delta_B_minus_A','delta_ci_low','delta_ci_high']]
    repairs=csv(error_out/'hybrid_slice_repairs.csv').groupby('dataset')[['base_errors','errors_repaired','errors_introduced']].sum().reset_index()
    sensitivity_counts=csv(sensitivity_out/'mask_counts.csv')
    sensitivity=csv(sensitivity_out/'summary.csv')
    sensitivity_primary=sensitivity[(sensitivity.budget.astype(str)=='200')&sensitivity.system.isin(['J-D+LR','J-D+C+LR','M2+best','Q-D+LR','L2-D+LR','Ditto'])]
    selection=jread(BASE/'runs/overnight_api_20260930/manifest.json')['protocol']
    # Every entry in the Luna comparison is recomputed on identical 500 IDs,
    # with the same three frozen budget draws. No full-vs-subset comparison.
    matched=[]
    source_dirs=[(BASE/'runs/recovery_repaired_20260930/predictions',['J-H1n','J-H1+LR','J-D+LR','J-D+C+LR','C+LR']),
                 (BASE/'runs/magellan_v2_20260930/predictions',['M2+best']),
                 (synthesis/'qwen_evaluation/predictions',['Q-H1','Q-H1+LR','Q-D+LR']),
                 (BASE/'runs/overnight_finalization_20260930/llm_evaluation/predictions',['L1-H1','L1-H1+LR','L1-D+LR','L2-H1','L2-H1+LR','L2-D+LR'])]
    for ds in ['wa','ag','da','ab']:
        ids=selection[ds]['Luna_test_ids'];truth=None
        for folder,systems in source_dirs:
            for system in systems:
                for budget in ([0] if system in ['J-H1n','Q-H1','L1-H1','L2-H1'] else [50,200]):
                    for seed in range(1 if budget==0 else 3):
                        path=folder/f'{ds}_{system}_b{budget}_s{seed}.csv'
                        if not path.exists():raise ValueError('Required matched baseline prediction missing: '+str(path))
                        f=csv(path).set_index('pair_id',verify_integrity=True).loc[ids]
                        if truth is None:truth=f.label.values
                        if not np.array_equal(truth,f.label):raise ValueError('Matched subset labels conflict')
                        threshold=float(f.threshold.iloc[0]);measured=metrics(truth,f.probability.values,threshold)
                        if not np.array_equal(f.prediction.values,(f.probability.values>=threshold).astype(int)):raise ValueError('Matched decision threshold conflict')
                        matched.append(dict(dataset=ds,system=system,budget=budget,seed=seed,n_test=500,**measured))
        for manifest in (BASE/'runs/gpu_backup_20260930/ditto').glob('*/manifest.json'):
            m=jread(manifest);job=m['job']
            if job['dataset']!=ds or job['budget'] not in [50,200]:continue
            f=csv(manifest.parent/'test_predictions.csv').set_index('pair_id',verify_integrity=True).loc[ids]
            if not np.array_equal(truth,f.label):raise ValueError('Matched Ditto labels conflict')
            matched.append(dict(dataset=ds,system='Ditto',budget=job['budget'],seed=job['seed'],n_test=500,
                **metrics(truth,f.probability.values,m['threshold'])))
    if len(matched)!=304:raise ValueError('Expected 304 matched baseline/seed groups, got '+str(len(matched)))
    matched=pd.DataFrame(matched);matched.to_csv(OUT/'tables/matched_500_metrics_by_seed.csv',index=False)
    matched_summary=matched.groupby(['dataset','system','budget']).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),n_seeds=('seed','count')).reset_index()
    matched_summary.to_csv(OUT/'tables/matched_500_summary.csv',index=False)
    refs={'wa':86.76,'ag':75.58,'da':98.99,'ab':89.33};port=[]
    for ds,value in refs.items():
        f=ditto[(ditto.dataset==ds)&(ditto.budget.astype(str)=='full')].iloc[0];gap=float(f.f1_mean-value)
        port.append(dict(dataset=ds,observed_f1=float(f.f1_mean),reference=value,gap=gap,status='WITHIN_5_POINTS' if abs(gap)<=5 else 'BLOCK_FREEZE',
            source='https://arxiv.org/pdf/2004.00584',table='5',note='One-seed port check, not a multi-seed literature replication'))
    pd.DataFrame(port).to_csv(OUT/'tables/ditto_port_checks.csv',index=False)
    # Standard, exportable manuscript figures. Finite gaps remain gaps.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
    names={'wa':'Walmart–Amazon','ag':'Amazon–Google','da':'DBLP–ACM','ab':'Abt–Buy'}
    colors={'J-D+LR':'#0072B2','J-D+C+LR':'#009E73','M2+best':'#777777','Ditto':'#D55E00','Q-D+LR':'#CC79A7','L2-D+LR':'#E69F00'}
    def save(fig,name):
        fig.savefig(OUT/f'figures/{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'figures/{name}.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharey=True,layout='constrained')
    order=['50','200','1000','full'];xmap={x:i for i,x in enumerate(order)}
    for ds,ax in zip(['wa','ag','da','ab'],axes.flat):
        for system,color in colors.items():
            g=master[(master.dataset==ds)&(master.system==system)&master.budget.isin(order)].copy()
            g['x']=g.budget.map(xmap);g=g.sort_values('x')
            if len(g):ax.errorbar(g.x,g.f1_mean,yerr=g.f1_std.fillna(0),marker='o',markersize=3,capsize=2,color=color,label=system,lw=1.3)
        ax.set(title=names[ds],xticks=range(4),xticklabels=order,ylim=(0,102),xlabel='Labeled pairs (full Ditto: train + validation)');ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Test F1 (%)');axes[1,0].set_ylabel('Test F1 (%)')
    axes[0,0].legend(fontsize=7,ncol=2,loc='lower right');save(fig,'learning_curves_full_test')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    labels=['Jev','DeepSeek','Luna'];models=['jev-1.13','deepseek-v4-flash','gpt-6-luna'];loc=np.arange(3)
    for k,task in enumerate(['H','D']):
        g=cost[cost.task==task].set_index('model').loc[models]
        axes[0].bar(loc+(k-.5)*.35,g.cost_per_1000_decisions,width=.35,label=task)
        axes[1].bar(loc+(k-.5)*.35,g.latency_p50_ms/1000,width=.35,label=task)
    for ax in axes:ax.set_xticks(loc,labels);ax.legend(title='Task');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('USD / 1,000 decisions');axes[1].set_ylabel('Median latency (seconds)')
    fig.suptitle('Matched 48 design pairs per model/task; 6 concurrent calls, same AutoDL route',fontsize=9);save(fig,'matched_task_cost_latency')
    fig,axes=plt.subplots(1,4,figsize=(11,3),layout='constrained')
    for ds,ax in zip(['wa','ag','da','ab'],axes):
        for system,color in [('J-H1','#0072B2'),('Q-H1','#CC79A7'),('L2-H1','#E69F00')]:
            g=calibration[(calibration.dataset==ds)&(calibration.system==system)&(calibration.metric=='ece_equal_mass')]
            for mode,marker in [('raw','o'),('temperature','s')]:
                row=g[g['mode']==mode]
                if len(row):
                    r=row.iloc[0];x=['J-H1','Q-H1','L2-H1'].index(system)+(0 if mode=='raw' else .15)
                    # Percentile intervals of biased ECE need not straddle
                    # the point estimate. Draw their actual endpoints.
                    ax.vlines(x,r.ci_low,r.ci_high,color=color,lw=1.)
                    ax.scatter([x],[r.estimate],marker=marker,color=color,s=16)
        ax.set(title=names[ds],xticks=[0,1,2],xticklabels=['Jev','Qwen','DS']);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Equal-mass ECE (15 bins)');fig.suptitle('Circle: raw; square: pool-fitted temperature. Full test, conditional test-bootstrap CI.',fontsize=9)
    save(fig,'calibration_full_test')
    fig,axes=plt.subplots(1,4,figsize=(11,3),layout='constrained')
    for ds,ax in zip(['wa','ag','da','ab'],axes):
        for system,color in [('J-D+LR','#0072B2'),('Q-D+LR','#CC79A7'),('L2-D+LR','#E69F00')]:
            p=diagnostic/f'curves/{ds}_{system}_b200_s0.csv'
            if p.exists():g=csv(p);ax.plot(g.coverage,g.risk,label=system,color=color,lw=1.3)
        ax.set(title=names[ds],xlim=(0,1),ylim=(0,1),xlabel='Coverage');ax.grid(alpha=.2)
    axes[0].set_ylabel('Selective error rate');axes[-1].legend(fontsize=7);save(fig,'selective_risk_b200_seed0')
    backup=jread(BASE/'runs/gpu_backup_20260930/backup_status.json')
    final=jread(BASE/'runs/overnight_finalization_20260930/status.json')
    total_api=final['api_main_spent']+final['repair_spent']+final['route_pilot_spent']+float(csv(BASE/'runs/remote_cost_probe_inferera_20260930/ledger.csv').paid_usd.sum())
    repeats=jread(BASE/'runs/determinism_fresh_20260930/status.json')
    total_api+=repeats['paid']
    gpu_hours=(backup['queue']['finished']-backup['started'])/3600
    qmeta=jread(BASE/'runs/gpu_backup_20260930/qwen_recovery_20260930/manifest.json')
    primary=master[(master.budget=='200')&master.system.isin(['J-H1+LR','J-D+LR','J-D+C+LR','M2+best','Ditto','Q-D+LR','L2-D+LR'])][['dataset','system','f1_mean','f1_std','n_seeds','test_protocol']]
    h1rows=h1[h1.hypothesis=='H1'][['dataset','delta_B_minus_A','delta_ci_low','delta_ci_high']]
    h1positive=int((h1rows.delta_B_minus_A>0).sum())
    matched_primary=matched_summary[(matched_summary.budget==200)&matched_summary.system.isin(['J-D+LR','L1-D+LR','L2-D+LR','Q-D+LR','Ditto','M2+best'])]
    matched_zero=matched_summary[(matched_summary.budget==0)&matched_summary.system.isin(['J-H1n','L1-H1','L2-H1','Q-H1'])]
    text=f'''# 审计后实验结果与论文故事（待人工验收）

这份报告由原始概率、逐样本预测和冻结采样重新汇总。当前状态是 **CPU 汇总完成，图表视觉验收与负责人错误样本复核尚待完成**；不声称原预注册全部验收或结果已冻结。原 Gemini 报告不再作为数字来源，原文件和缓存保留。

## 可以讲的研究问题

本研究比较不同实现的概率决策：商业类型化接口、开源 Qwen 的 Yes/No token 概率、通用 LLM 的自报概率，以及按标注预算训练的 Ditto 和字符串特征分类器。论文应回答“在什么数据与预算下，冻结原子问题加轻量汇总器值得使用”，而不是预先宣称某系统全面领先。

1. **标注预算与任务难度共同决定系统选择。** 同一冻结预算比较显示，低预算训练的 Ditto 在商品匹配上可能明显不足，在 DBLP–ACM 上较强；全量标注后其质量显著改善。商品和论文数据应分别讨论，不能省略负面数据集。
2. **拆题不是普遍增益。** b200 的 Jev 拆题增益在 {h1positive}/4 数据集点估计为正，WA/AG 区间包括 0，DA 为负。H1 原文的方向性条件与“显著提升”是不同结论。H2 必须看下表中四预测交互，而不是仅比较最终 F1。
3. **概率来自哪里，需要实测校准。** Qwen 使用 BF16 完整词表的 P(Yes)/(P(Yes)+P(No))；LLM 是口头自报概率；Jev 是接口返回的类型化概率。输出类型可约束格式，不能保证事实正确或校准良好。报告温度、ECE、Brier、NLL 和选择性风险，保留 Yes/No 无条件质量标记。
4. **加入符号特征的增益需要逐数据集核验。** 混合模型在相同标注、相同测试上比较；修复错误和引入错误同时报告，不能仅列成功案例。数字探针使用独立程序真值。语义原子问题的人工真值没有补造，不能把 H5 的语义对照写成完成。
5. **部署代价要与相同任务对齐。** E6 已有同一 AutoDL 路由、相同 48 个设计对、6 并发的 H/D 日志。成本和延迟优势限于这个受控测量，不沿用旧报告“节省 78%/快 3.4 倍”或把整体问成本与拆题成本混比。

## 主体结果：完整测试集，b200

{markdown(primary)}

种子数不同源于事先记录的费用修订。Ditto 的 b 包括训练与内部验证标签；全量点使用原训练和验证。Qwen/LLM 只计算 pool2000 特征，没有全量学习点。M2+best 为自建字符串特征加交叉验证选择分类器，不能称官方 Magellan 复现。

## Luna 公平对照：完全相同的 500 对，三个预算种子

先给无需拟合汇总器的零样本整体问；这里的“零样本”是没有训练标签拟合分类器，不表示从未人工设计提示：

{markdown(matched_zero)}

以下为b200固定三种子对照；不能省略上面更便宜的零样本LLM成绩，只强调带标注模型与Ditto的差距：

{markdown(matched_primary)}

所有系统均按 Luna 的冻结 ID 子集重新计算，不能将这张表与上面完整测试表直接做差。

## 拆题增益与假设

Jev 自身的 b200 D−H（百分点）：

{markdown(h1rows)}

H2 为 (Jev D−H)−(对照 D−H)，正数才是 Jev 的拆题增益更大；交叉 bootstrap 保留同一测试行跨种子的相关性：

{markdown(h2)}

原主要假设族还缺 H5 人工语义真值等终点，未把选出的子集冒充完整 Holm 校正族。H6 的 Qwen H/D 在共同干净正确子集上比较，见下表；不能推广为 Luna、DeepSeek 的新注入结果。

{markdown(attack)}

## 符号特征与错误切片

b200的J-D+C+LR相对J-D+LR变化（百分点、十种子交叉bootstrap）：

{markdown(hybrid_comparison)}

固定seed0切片中的错误修复与新错计数：

{markdown(repairs)}

该表是同预算固定模型的比较，不是H5语义原子真值的替代。16组系统/数据集的切片TP/FP/FN/TN求和均严格还原主表。机器规则按固定优先级给互斥类别，并统一空值标记；之前AG的None字符串漏判已修复，前后主表F1完全不变。类别表示规则命中，不等于已经人工确认的错误原因。每组至多100例、共915例错误及30例负责人待复核包保留；负责人一致率尚未观察。

## 原划分内容重复的敏感性检查

官方划分的pair_id不交叉，不代表整对记录内容没有重复。以下诊断排除测试中与任意原训练/验证整对内容完全相同的案例，并将测试内相同整对内容只保留第一次；它不要求实体本身不跨划分，也不更改官方主测试、标签或模型。

{markdown(sensitivity_counts)}

相同诊断测试ID上的b200结果如下（full指原主测试，retained为过滤后；单位为F1百分点）：

{markdown(sensitivity_primary)}

这些排除规则在Jev/LLM结果已查看、Qwen分析尚未发生时写入，属于审计后敏感性分析，不能称为过去的预注册。低预算某种子未必使用全部原训练标签，此共同过滤因此是保守的内容重复检查；没有重新训练，也不能推导至未见实体的泛化。Luna500的过滤结果另在summary中，仍不与其他系统完整测试混比。

已有问题文件哈希能够核对当前快照，但现有材料未证明历史测试调用之前的签字冻结或公开注册。新修订与执行源码可以约束后续补验，不能反向建立旧实验的确认性身份。

## GPU 移植核验与调用质量

40 次 Ditto 均有最佳 checkpoint、每轮训练/验证日志、逐测试样本预测、源码快照与输出哈希。四个全量点与原论文表 5 相差均须小于 5 点；这是一种子移植检查，不是重现其多种子最佳值。

{markdown(pd.DataFrame(port)[['dataset','observed_f1','reference','gap','status']])}

参考：[Ditto 原论文，表 5](https://arxiv.org/pdf/2004.00584)。Qwen 后端为 {qmeta['backend']}；vLLM/批量/前缀复用未通过预设 gate 的记录保留。低于 0.5 的 Yes/No 无条件概率质量占总体 {qmeta['fraction_yes_no_mass_below_half']:.3%}，分实验/题目见 yes_no_mass.csv，不把条件概率当作模型愿意按要求回答的保证。

## 成本、关机与协议边界

本夜按返回 usage 与配置单价计算的新增 API **费用估计为 ${total_api:.6f}**，这不是已核实的账户扣款。用户提供的新增余额上限为 $3.12，但网关已明确返回余额不足，所有付费 API 已停止，没有充值或升级免费模型；不得按估计金额推断账户剩余余额。失败请求的实际计费及估计与钱包状态的差异尚缺账户账单，原因未知。监控窗口至计算结束约 **{gpu_hours:.3f} 小时**，不含更早模型准备时间；GPU 单价未得到回复，不能编造人民币总额。

GPU controller 状态：`{backup['status']}`。只有成功调用 `/usr/bin/shutdown` 并 SSH 不可达，才能称已发出关机且断连；本报告没有 AutoDL 控制台电源状态读数。

以下限制保留在最终稿中：decision-model-preview 免费配额仅返回少量有效样本，无法形成第二商业模型基准；Adult 原 train/valid/test 完全重复，旧监督结论撤回，新去重按组附录单列；Hospital 原训练标注不满足整行预算，撤回监督行预算比较；Flights 六列协议与 Raha 文献协议不同，不能宣布刷新 SOTA；官方 Magellan 的 AG 停止条件未闭环，只能使用明确命名的自建基线；E8 负责人 30 例复核、可选 E12 人工原子标注未完成；E11 为未来版本漂移，未冒充本夜完成。

新增 Jev 新鲜重复请求状态为 `{repeats['status']}`。计划是每数据集200对、每对五次H1_noul单题请求，首个请求前冻结ID；实际仅WA有948次成功，148对完成全部五次，其余3052次因余额不足返回403。在148对完整案例中，91对概率有变化，1对在0.5阈值下翻转，最大概率极差为0.12。逐请求核验确认请求正文完全相同、实际响应模型版本相同、五个服务响应ID不同，故“该接口严格确定”被反例推翻。61.5%的概率变化率只描述这148对完整WA案例，不能推广至全部200对、其他数据集或多题独立性。原采集器未在首次钱包拒绝时立即熔断，这是接手执行失误；源码已修正，原执行源码与全部拒绝记录保留，没有重跑或补齐冒充完整。离线核验见 runs/determinism_verified_20260930。

## 写论文时的组织方式

建议标题限定 Jev：**Typed Probability Interfaces for Low-Label Entity Matching: Calibration, Decomposition, and Deployment Trade-offs**。引言提出低标注下的概率接口选择问题；方法给出冻结问题、同预算抽样、训练内 OOF 阈值与 token 条件概率；实验先给完整学习曲线，再给相同 500 对的通用 LLM 对照，随后解释拆题交互、校准/选择性预测、数字边界和注入条件。错误检测与新 Adult 协议放清楚标注的扩展/附录，结论承认充分标注下微调模型的优势与数据集差异。

故事是否成立，应由表中实际胜负决定。若 Qwen/LLM 同样或更好，则贡献是受控成本、概率校准与失败边界的实证研究，不能把原子分解当成 Jev 独有能力。

## 文件与验收入口

主表位于本目录 tables/，四张 PDF/PNG 位于 figures/。原始证据保存在 runs/gpu_backup_20260930、runs/overnight_api_20260930；新诊断在 runs/qwen_diagnostics_20260930，Flights 按行实验在 runs/flights_row_recovery_20260930。详细 completeness gate 与 missing 列表优先于总结状态字符串。下一步必须视觉检查这些图、抽查表与原始预测，并更新逐问题闭环表；没有进行的人工步骤保持未完成。
'''
    (OUT/'论文故事与结果说明.md').write_text(text,encoding='utf8')
    for source in [diagnostic/'numeric_probes.csv',diagnostic/'error_detection.csv',ed/'summary.csv',diagnostic/'stability.csv',diagnostic/'attacks.csv']:
        (OUT/'tables'/source.name.replace('summary.csv','flights_row_summary.csv')).write_bytes(source.read_bytes())
    for source in [BASE/'runs/stability_utf8_verified_20260930/stability.csv',BASE/'runs/determinism_fresh_20260930/repeat_summary.csv',BASE/'runs/overnight_finalization_20260930/adult_zero_shot/metrics.csv']:
        (OUT/'tables'/('Jev_'+source.name)).write_bytes(source.read_bytes())
    for name in ['partitioned_slices.csv','confusion_reconciliation.csv','hybrid_slice_repairs.csv']:
        source=error_out/name;csv(source);(OUT/'tables'/('E8_'+name)).write_bytes(source.read_bytes())
    for name in ['owner_30_review_packet.jsonl','复核说明.md']:
        source=error_out/name;inputs[source.relative_to(BASE).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
        (OUT/name).write_bytes(source.read_bytes())
    for name in ['summary.csv','mask_counts.csv']:
        source=sensitivity_out/name;csv(source);(OUT/'tables'/('content_sensitivity_'+name)).write_bytes(source.read_bytes())
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(dict(status='CPU_ARTIFACTS_READY_REQUIRES_VISUAL_AND_INTEGRITY_REVIEW',input_sha256=inputs,
        new_api_known_cost=total_api,gpu_shutdown=backup['status'],
        outputs={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name not in ['status.json','manifest.json']}),indent=2))
    status_path.write_text(json.dumps(dict(status='CPU_ARTIFACTS_READY_REQUIRES_VISUAL_AND_INTEGRITY_REVIEW',new_api_known_cost=total_api,report=str(OUT/'论文故事与结果说明.md')),indent=2))
    print('DELIVERY_CPU_READY',str(OUT),flush=True)

if __name__=='__main__':main()
