"""Prepare reviewed local handoff documents; does not collect or train."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import numpy as np,pandas as pd

BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parent
OUT=BASE/'reports/final_delivery_20261001'
REVIEW=BASE/'reports/overnight_recovery_20261001_reviewed'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def md(frame):
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.3f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    if (OUT/'acceptance.json').exists():raise ValueError('Accepted delivery is immutable; create a new revision')
    numeric=BASE/'runs/final_integrity_20261001_reviewed/status.json'
    status=json.loads(numeric.read_text(encoding='utf8'))
    assert status['n_master_groups']==224 and status['n_prediction_groups']==1296
    assert status['n_matched_groups']==304 and status['n_flights_groups']==80
    for name,expected in status['outputs'].items():assert sha(numeric.parent/name)==expected
    figure_dir=OUT/'figures_reviewed'
    visual=json.loads((figure_dir/'visual_acceptance.json').read_text(encoding='utf8'))
    assert visual['status']=='FOUR_FIGURES_VISUALLY_REVIEWED'
    figmanifest=json.loads((figure_dir/'manifest.json').read_text(encoding='utf8'))
    for rel,expected in figmanifest['inputs'].items():assert sha(BASE/rel)==expected
    for name,expected in figmanifest['outputs'].items():assert sha(figure_dir/name)==expected
    intervals=BASE/'runs/final_interval_review_20261001'
    im=json.loads((intervals/'manifest.json').read_text(encoding='utf8'))
    for rel,expected in im['inputs'].items():assert sha(BASE/rel)==expected
    for name,expected in im['outputs'].items():assert sha(intervals/name)==expected
    tabledir=OUT/'tables';tabledir.mkdir(exist_ok=True)
    for p in list((REVIEW/'tables').glob('*.csv'))+list(intervals.glob('*.csv')):
        target=tabledir/p.name
        if target.exists():assert sha(target)==sha(p)
        else:shutil.copy2(p,target)
    master=pd.read_csv(tabledir/'full_system_summary.csv');master.budget=master.budget.astype(str)
    names=['wa','ag','da','ab']
    def pivot(frame,systems):
        f=frame[frame.system.isin(systems)].pivot(index='dataset',columns='system',values='f1_mean')
        return f.reindex(index=names,columns=systems).rename_axis('dataset').reset_index()
    b200=pivot(master[master.budget=='200'],['J-H1+LR','J-D+LR','J-D+C+LR','Q-D+LR','L2-D+LR','M2+best','Ditto'])
    b50=pivot(master[master.budget=='50'],['J-D+LR','Q-D+LR','M2+best','Ditto'])
    matched=pd.read_csv(tabledir/'matched_500_summary.csv');matched.budget=matched.budget.astype(str)
    zero=pivot(matched[matched.budget=='0'],['J-H1n','L1-H1','L2-H1','Q-H1'])
    fivehundred=pivot(matched[matched.budget=='200'],['J-D+LR','L1-D+LR','L2-D+LR','Q-D+LR','Ditto'])
    pc=pd.read_csv(BASE/'runs/overnight_synthesis_20260930/paired_comparisons/paired_system_comparisons.csv')
    delta=pc[(pc.budget.astype(str)=='200')&(pc.system_B=='Q-D+LR')].copy()
    delta['Jev_minus_Qwen']=-delta.delta_B_minus_A;delta['CI_low']=-delta.delta_ci_high;delta['CI_high']=-delta.delta_ci_low
    h2=pd.read_csv(BASE/'runs/overnight_synthesis_20260930/paired_comparisons/h2_interaction.csv')
    h2=h2[h2.comparator=='Q'][['dataset','j_decomposition_gain','comparator_decomposition_gain','interaction','ci_low','ci_high']]
    port=pd.read_csv(tabledir/'ditto_port_checks.csv')[['dataset','observed_f1','reference','gap','status']]
    attacks=pd.read_csv(tabledir/'attacks_exact_binomial.csv')
    selected=attacks[attacks.system.isin(['Ditto-full','C+LR_b200_seed0'])][['dataset','system','variant','n_eligible','n_events','estimate','cp95_low','cp95_high','fraction_truncated']]
    for c in ['estimate','cp95_low','cp95_high','fraction_truncated']:selected[c]=selected[c]*100
    selected=selected.rename(columns={'estimate':'ASR_percent','cp95_low':'exact95_low_percent','cp95_high':'exact95_high_percent','fraction_truncated':'truncated_percent'})
    cost=pd.read_csv(BASE/'runs/remote_cost_probe_inferera_20260930/matched_task_cost_latency.csv')
    cost=cost[['model','task','cost_per_1000_decisions','latency_p50_ms']].copy();cost['latency_p50_seconds']=cost.pop('latency_p50_ms')/1000
    probes=pd.read_csv(tabledir/'numeric_probes.csv');probes['accuracy_percent']=probes.accuracy*100
    story=f'''# Jev 实验接管：最终中文论证与交付说明

**结论：原始规范化实体匹配数据和大部分调用证据可以继续使用，但 Gemini 的旧结果处理、部分实验实现和总结报告存在严重错误。经过修复和补验，能够支持的故事是：冻结的类型化概率特征加轻量分类器，在低标注商品匹配中有实用价值；它的优势依赖任务、标注预算和对照系统，不能写成全面领先、拆题独有、严格确定或刷新错误检测 SOTA。**

本轮计算、离线统计核验和四张图的视觉验收已完成。40 次 Ditto、424,796 条 Qwen 原始前向、16,731 条 NLI 预测及 Ditto 注入结果均已备份；GPU 已执行官方关机命令，随后 SSH 不可达。API 主体完整，钱包耗尽后所有付费调用停止。E8 人工复核、人工语义真值、第二商业决策模型完整基准和未来版本漂移仍缺证据，本交付不将这些项目登记为完成，也不宣称原预注册全部通过。

## 研究问题与可主张的贡献

建议研究问题为：**在少量任务标注下，商业类型化接口、开源 token 概率、通用 LLM 自报概率和任务微调模型，各自在哪些数据与部署条件下值得选择？**

这次实验的价值在于给出共同样本、共同标签预算、可还原预测的比较，并展示概率解码、阈值训练、校准和部署成本如何改变结论。接口提供结构化概率方便工程接入，但结构化格式本身不能证明事实正确、校准可靠或架构优越。这里评测的是具体系统和冻结问题集，尚未覆盖两家商业决策模型，标题应限于 Jev 或明确的一项案例研究。

可用标题：**A Study of Jev Probability Interfaces for Low-Label Entity Matching: Decomposition, Calibration, and Deployment Trade-offs**。

## 先看完整测试集，避免仅与弱基线比较

四个数据集：WA=Walmart–Amazon 商品，AG=Amazon–Google 商品，DA=DBLP–ACM 论文，AB=Abt–Buy 商品。J=Jev，Q=Qwen3-8B，L1=Luna，L2=DeepSeek；H=整体问，D=拆题，C=代码符号特征，LR=逻辑回归。M2 是自建字符串特征与训练内交叉验证选择的分类器，**不是已通过验收的官方 Magellan 复现**。

下面均为 F1 百分数。b=200 指分类器训练/验证使用的 200 个匹配对标签；J/Q/L2/M2 是十种子平均，Ditto 为事先批准的三种子平均。不同算法使用训练内 OOF 或内部留出验证，标签总量相同，分配策略仍有差异。

{md(b200)}

Jev 拆题相对 Qwen 拆题的 b200 差值及描述性 95% 区间如下；保留了同一测试对跨种子的相关性，使用共同测试行与种子的交叉 bootstrap，不把尾比例当正式 p 值：

{md(delta[['dataset','Jev_minus_Qwen','CI_low','CI_high']])}

WA、AG、AB 的结果支持 Jev 在这三个商品数据上的优势；DA 区间包含零，不能主张它优于 Qwen。b=50 的完整测试点估计也给出相同商品/论文差异：

{md(b50)}

低预算 Ditto 在三个商品数据上明显不足，而 DA 已接近 90 F1。这是本次固定训练协议下的观察，不能推广成“所有少样本微调都不行”。低预算正例稀少、训练/验证分配及预训练模型和超参数都会影响表现，不能用全量移植通过替代低预算算法的全空间最优保证。

全量 Ditto 单种子仅用于检查移植：

{md(port)}

四个移植点与 [Ditto 原论文表 5](https://www.vldb.org/pvldb/vol14/p50-li.pdf) 均相差不到五点。全量 Ditto 在 AG、DA 优于本次 Jev 拆题；WA、AB 不是同样的胜负。全量点额外使用原验证标签，且只有一个种子，不能与低预算点等量解读，更不能将其当论文最佳多种子结果。

## 强零样本 LLM 是必须保留的竞争者

Luna 为每个数据集事先固定的 500 个分层测试对，b50/b200 各三个预算种子。DeepSeek 为完整 pool/test。所有包含 Luna 的直接比较，都将其他系统重新计算到完全相同的 500 个 ID，不能将 Luna 子集与别人的完整测试 F1 相减。

同 500 对的零样本整体问：

{md(zero)}

Luna 在 WA、DA、AB 高于 Jev 零样本整体问，AG 略低。它不需要拟合任务分类器，是实用选择，不能为了突出低标注分类器而省略。零样本表示没有任务标签拟合分类器，不意味着没有提示设计工作。

同 500 对、同 b200 三种子的拆题比较：

{md(fivehundred)}

Luna 的点估计在三个数据集高于 Jev 拆题，AG 低于 Jev；这些是配对子集结果，不能移用完整测试十种子区间。因而“Jev 低预算全面优于 LLM”不成立。论文应解释质量与推理成本/延迟的选择，而不是只与 Ditto 或旧字符串 LR 比较。

## 拆题的绝对质量与相对收益要分开

Jev b200 D−H 分别为 WA +1.683、AG +1.448、DA −1.408、AB +1.992 点。H1 原文要求的 3/4 正方向在修复后点估计上满足；WA/AG 的描述性区间包含零，DA 区间为负，AB 为正。这不等于三个数据集都得到显著提升。Claude 按旧表得出的 2/4 需要更新。

H2 需要计算 (Jev D−H)−(对照 D−H)，不是仅比较 D 的最终 F1。对 Qwen 的结果为：

{md(h2)}

四个交互均为负，且描述性区间不含零。即使 Jev 拆题在商品匹配上绝对分数更高，它的相对拆题收益仍比 Qwen 小。Qwen 整体问起点较弱，拆题改善更多；这个事实不支持拆题是 Jev 独有能力。对 Luna 的四个交互区间均包含零；对 DeepSeek 方向不一致。H2 不获得原先设想的普遍支持。

## 概率有效性：校准、选择性预测与格式约束

Qwen 使用逐条 BF16 完整词表前向，取 P(Yes)/(P(Yes)+P(No))。这是真实 token 概率的条件归一化，与 LLM 口头自报概率不同，也不等于无条件正确率。vLLM、批量和前缀方案未通过预设门槛的结果没有放入主实验；逐条优化通过 50/50，未降低原门槛。424,796 条返回的索引、概率范围、归一化均通过核验，最低 Yes+No 质量约 0.69549。

Qwen 的 DA 整体问默认阈值 0.5 得分很低，但训练内拟合阈值/汇总器后可超过 94 F1。这提示默认概率阈值不能直接当决策标准，不能据此把所有低原始 F1 都判为数据处理错误。Choice/Score 则确实存在旧解码错误，已从原始分布重新映射。

校准图使用 J/Q/DeepSeek 各数据集的 **2,000 个 pool 标签**拟合温度，与 b200 分类器是不同实验。提示设计、校准、错误规则审计也有资源成本；“b=50/200”只描述相应分类器的任务标签预算，不能声称整个研究只用了这些标签。Luna 校准使用预算并集，池大小及 500 对协议另列。

Jev Noul 的拟合温度四个数据集均小于 1，Qwen 均大于 1；正确解码后的 Score/Choice 方向随数据集变化，不保留旧表的普遍方向结论。温度拟合目标是 NLL，并不保证 ECE 每次下降，例如 Qwen DA、DeepSeek AG 的 ECE 反而变差。图中 ECE 区间是固定训练拟合后的测试 bootstrap，未包含所有温度拟合不确定性，温度区间单列。

选择性风险曲线使用冻结 b200 seed0 模型，按置信度排序后观察覆盖率与错误率。低覆盖的条件准确率不能替代正例召回率，负例占比较高也可能使准确率漂亮。本轮补充二项计数区间，避免零观测错误被经验 bootstrap 写成确定的零风险；仍没有分布无关的风险保证。

## 符号特征、数值与错误解释

J-D+C 相对 J-D 的 b200 改变为 WA −0.850、AG −2.712、DA +1.847、AB −0.354 点；只有 DA 的描述性区间明确为正。固定 seed0 的 DA 修复 25 个错误、引入 3 个新错，AB 修复 3 个、引入 17 个。不能仅选 DA 成功案例后宣称混合始终更好。

数值探针的 Qwen 观察如下：

{md(probes[['dataset','probe','n','accuracy_percent']])}

这些是平衡类别、冻结程序真值的数值/编号探针，不是人工语义实体匹配标签。AG version 与 WA price 暴露明显弱点，DA year 较好。它们不能替代 H5 所需的人工语义原子真值，H5 仍未完成正式检验。

E8 已修复行错配、重训模型与主实验不一致等错误，16 个系统/数据集切片的 TP/FP/FN/TN 求和全部等于对应主表。统一 None/null/nan/n/a 后，341/915 个抽取错误样本的机器规则类别变化，但主预测与 F1 不变。915 例错误展示和 30 例负责人复核包已交付。规则命中只是审计线索，不是人工确认原因；未获得人工一致率，不能编造“已复核”。

## 鲁棒性允许出现反例

Jev 的 28 个扰动分组已完整执行；旧 P2/P5 相同是排序缓存与题型实现问题，现保留实际顺序不同的正文/响应证据。Qwen 有 48 个 H/D 扰动诊断条目；整体问在 P3/P4 保持不变，因此不把这些条目当“确实被改写后的稳定性证明”，Noul 的 P5 不适用。

新鲜重复检查只获得 WA 948 次成功响应，其中 148 对有完整五次。91 对概率发生变化、1 对越过 0.5 阈值，最大极差 0.12。相同请求正文、相同响应版本、不同服务响应 ID 已核验，“该接口严格确定”被反例否定。这个 61.5% 概率变化率只描述完整 WA 案例，不能外推到四个数据集或多题联合概率。

注入比较按冻结 clean 模型执行，不重新训练攻击样本。Qwen H/D 的共同 clean-correct 子集上，WA 的 T1/T3 拆题 ASR 较低，但 T2 没有同样改善；AB 三个区间都包含零。H6 不支持普遍的抗注入能力。普通分类器的结果也必须列出：

{md(selected)}

上表使用两侧精确二项 95% 区间。例如 WA Ditto 0/384 次成功攻击的上界仍约为 0.956%，不是免疫证明。旧诊断中的零事件经验 bootstrap [0,0] 保留作历史输出，最终二项边际率使用新补充区间。AB Ditto 约 1.25–1.5% 的追加内容存在截断，可能削弱攻击；各系统自身 clean-correct 分母不同，不能直接将这些 ASR 当无条件因果优劣。Jev/DM 不新增注入结果，不把缺测冒充免疫。

## 错误检测应作为有边界的扩展

Flights 采用 5/20/50/full 整行预算，每档五种子，整行分组 OOF，80 个系统/预算/种子组全部完整。零样本 Qwen 整体问 F1 为 68.193，Jev 为 27.690；50 行预算 Qwen H/D 为 73.900/74.483，Jev H/D 为 55.124/72.709。Qwen 使用 column_profile、Jev 历史输入不带这一配置，因此这是系统协议比较，不是等提示架构因果实验。六列协议也不同于 Raha 文献，不宣布 SOTA。

Hospital 训练标签是稀疏单元格，不能当完全标注行预算；撤回旧监督行预算比较。Qwen 整体问零样本 F1 12.772、精确率 6.903、召回率 85.185，大量误报说明“高召回”本身不能证明可用。Adult 原始 train/valid/test 完全重复，旧监督泛化结论撤回；Jev 11,000 单元格零样本 F1 59.540，TP/FP/FN/TN=401/446/99/10054。新去重按组划分只作不同协议的附录探索，不将它伪装成旧结果修复。

[Narayan 等原论文表 2](https://arxiv.org/pdf/2205.09911) 的 Hospital HoloDetect/GPT-3 few-shot 为 94.4/97.8、Adult 均为 99.1。旧报告 80/73/72/66 的引用错误已撤回；该文的演示数、数据协议与本项目也不同，不能将文献数字直接拼成同条件榜单。NLI 16,731 个 pool/test 概率已经完成，评测列于完整主表和附录，不冒充另一商业决策模型。

## 成本、余额与 GPU 关机

同路由、同 48 个设计对、6 并发的受控点测量如下：

{md(cost)}

H/D 分别对应整体问/拆题。单价估计是每千次判断的推理费用，不包括标签、人力、训练、GPU 租赁或性能等价服务，也不是账户发票。Jev 的受控成本和延迟低于本次 Luna/DeepSeek 路由，不能沿用旧“省 78%、快 3.4 倍”的全局说法。48 对点测量的外推范围有限。

本夜新增已知 usage×单价估计 **$2.958075947**；用户确认钱包余额为零，网关也已拒绝请求。估计不是已核实的实付，失败请求的计费与钱包差额缺账单，原因未知。主 API 的 Luna/DeepSeek 八组、Jev 28 个扰动组和 Adult 11,000 单元格完整；免费 DM 仅少量样本、追加新鲜重复支路不完整。所有付费任务已退出，没有充值、升级或新付费重试。

重复支路第一次余额 403 后没有立即熔断，继续产生拒绝请求，是 Codex 接手执行失误。全部 3,052 次拒绝记录保留，未来源码已加停止逻辑，没有重新运行。此处与 E5 GBK/UTF8 缺失误报、最终历史/新 LLM 汇总串接错误一并公开，不能只追究 Gemini。

GPU 40 次 Ditto/Qwen/NLI/注入均完整，下载 1,781 个文件。已执行 `/usr/bin/shutdown`，随后 SSH 不可达；没有独立读取 AutoDL 控制台电源/账单，不能给出平台权威停费时刻。监控起点至计算完成约 8.982 小时，包含人工误关机后的中断，**不是计费 GPU 时长**；此前准备时间、租赁单价与账单未知。数据盘保留，未释放实例。误关机后的恢复只补余下 3,014 个 Qwen 提示，旧 421,782 条响应前缀 SHA 完全未变。

## 现在能写什么，仍缺什么

可以写完整学习曲线、同 500 对 LLM 对照、拆题交互、校准和选择性风险、符号特征的得失、受控推理成本，以及错误检测/注入的失败边界。原 H3 应改成与明确自建基线的事实比较，官方 Magellan 的 AG 超过参考五点停止条件仍未通过，不能宣称原假设的官方基线复现闭环。

H1 的方向条件部分支持但有明确负面数据集；H2 的 Jev 独特拆题收益不支持；H4 按题型和数据集分别解释；H5 人工语义真值不足；H6 仅局部条件性改善。原主要假设族不完整，未把筛出的对比假冒完整 Holm 校正族。历史问题哈希只能证明当前文件一致，不能事后补出测试前签字或公开预注册。因此验收状态是“计算交付已核验，带明确研究边界”，不是“原预注册全部通过并冻结”。

尚缺：E8 负责人 30 例人工复核；若保留 H5/E12，则须收集真实人工语义标注；第二商业决策模型完整基准目前缺免费额度，不新增付费；E11 是 2027 年投稿前的未来版本漂移。当前有限预算下可完成的补计算已经交付，未来实验或人工步骤不能虚构为今夜完成。

## 中文摘要草稿

本研究评估 Jev 类型化概率接口在低标注实体匹配中的适用范围，并与开源 Qwen token 概率、通用语言模型自报概率、字符串特征分类器及 Ditto 微调模型进行对照。实验使用四个实体匹配基准、冻结标签抽样和训练内决策阈值，保留逐请求与逐样本证据。在 200 条标注下，Jev 拆题模型在三个商品数据集相对 Qwen 拆题模型的 F1 优势为 7.1–11.6 点，在论文数据上差异不明确；Luna 在共同测试子集上的零样本及拆题结果则具有竞争力。Jev 的相对拆题收益并不普遍高于其他模型，符号特征仅在部分任务改善质量，概率校准与注入鲁棒性也呈任务差异。受控路由测量显示本次 Jev 推理成本与延迟较低。结果支持将类型化接口视为一种具有部署价值的概率特征来源，同时反对将输出格式约束等同于正确性、校准或普遍性能保证。本研究为系统选择提供可核验的质量、标签预算与推理代价比较，而不主张全面优越或错误检测 SOTA。

## 已验收成果入口

- 主表/同 500 对/附录：[tables](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/tables/full_system_summary.csv)。
- 全部细表与区间：[详细结果附录](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/详细结果附录.md)。
- 四张 PDF/PNG：[学习曲线](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/figures_reviewed/learning_curves_full_test.pdf)、[成本延迟](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/figures_reviewed/matched_task_cost_latency.pdf)、[校准](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/figures_reviewed/calibration_full_test.pdf)、[选择性风险](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/figures_reviewed/selective_risk_b200_seed0.pdf)。
- [统计和证据验收](E:/Desktop/jev/jevwrangle/reports/final_delivery_20261001/统计与证据验收.md)、[逐问题最终闭环](E:/Desktop/jev/audit_codex_2026-09-30/逐问题闭环_最终验收.md)。

原 Gemini、初版 Codex 汇总及失败验收快照全部留存。数字以 reviewed 新主表及独立重算为准；生成附录中的阶段状态是历史记录，最终验收以本目录说明和 acceptance.json 为准。
'''
    (OUT/'中文完整论证与交付说明.md').write_text(story,encoding='utf8')
    appendix=(REVIEW/'论文故事与结果说明.md').read_text(encoding='utf8')
    appendix=appendix.replace('当前状态是 **CPU 汇总完成，图表视觉验收与负责人错误样本复核尚待完成**','当前状态是 **离线数值与证据核验、四图视觉验收已完成；负责人错误样本复核尚未进行**')
    appendix=appendix.replace('## 文件与验收入口','## 原汇总阶段文件与验收入口（历史说明）')
    appendix='# 最终详细结果附录\n\n本附录来自修复后的唯一主数据源；最终解释、精确二项边际区间、GPU计费边界与验收入口以《中文完整论证与交付说明》为准。以下保留生成阶段的全部表格和状态，不将待人工步骤伪报完成。\n\n'+appendix
    (OUT/'详细结果附录.md').write_text(appendix,encoding='utf8')
    (OUT/'交付源码.py').write_bytes(Path(__file__).read_bytes())
    print('DOCUMENTS_READY',OUT)

if __name__=='__main__':main()
