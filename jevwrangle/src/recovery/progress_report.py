"""Publish an honest recovery status from retained artifacts."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parent


def main():
    now = datetime.now(timezone.utc).isoformat()
    api_dir = BASE / 'runs/api_llm_recovery_20260930'
    status = json.loads((api_dir / 'status.json').read_text(encoding='utf8'))
    manifest = json.loads((api_dir / 'manifest.json').read_text(encoding='utf8'))
    parsed = [json.loads(s) for s in (api_dir / 'parsed.jsonl').read_text(encoding='utf8').splitlines()]
    recorded, failed = len(parsed), sum(r['status'] != 'ok' for r in parsed)
    conflicts = sum(bool(r.get('answer_probability_conflict')) for r in parsed)
    repaired=0
    repair_file=BASE/'runs/api_llm_missing_20260930/parsed.jsonl'
    if repair_file.exists():repaired=sum(json.loads(s)['status']=='ok' for s in repair_file.read_text(encoding='utf8').splitlines())
    cost = 0.
    cost_rows = []
    for name in ['api_protocol_pilot_20260930', 'api_route_probe_20260930', 'api_jev_repairs_20260930', 'api_llm_recovery_20260930','api_llm_missing_20260930']:
        p = BASE / 'runs' / name / 'ledger.csv'
        if not p.exists():continue
        f = pd.read_csv(p)
        amount = f.paid_usd.sum() if 'paid_usd' in f else f.paid.sum()
        cost += amount
        cost_rows.append(f'| `{name}` | {amount:.6f} | {len(f)} |')
    jev = pd.read_csv(BASE / 'runs/recovery_repaired_20260930/tables/em_summary.csv')
    baseline = pd.read_csv(BASE / 'runs/magellan_v2_20260930/summary.csv')
    stats = pd.read_csv(BASE / 'runs/magellan_v2_20260930/paired_comparisons.csv')
    comparison = []
    temperature_text = ''
    ci_path=BASE/'runs/calibration_ci_20260930/temperature_intervals.csv'
    if ci_path.exists():
        ci=pd.read_csv(ci_path)
        temperatures=[]
        for _,row in ci.iterrows():
            temperatures.append(f"| {row.dataset} | {row.system} | {row['T']:.3f} | [{row.T_ci_low:.3f}, {row.T_ci_high:.3f}] |")
        temperature_text=f'''## 新完成的校准与选择性预测

12 个题型/数据集温度的 2000 次训练池重抽样均完成，无优化失败。测试 ECE、Brier、NLL 的 2000 次重抽样区间也已生成；测试指标区间以拟合后的 T 固定为条件，训练温度的不确定性单列。

| 数据集 | 题型 | T | 95% 区间 |
|---|---|---:|---|
{chr(10).join(temperatures)}

Noul 的 T 在四个数据集上均小于 1；Score 和 Choice 的方向随数据集改变。不能将 H4 写成全面成立。参考题型映射是 Noul 0.66、Choice 1.30、Score 1.92；该外部评测与本任务不同，方向比较是描述性的。见 [原评测仓库](https://github.com/scienthoon/jev-ood-calibration)。

另外已生成 356 组 Jev 风险—覆盖数据、0.95/0.99 筛选阈值及区间、336 组 LR 输出校准点估计，并回查 Choice/Score 原始 confidence 字段。只有二元金标准，因此“答对置信度”的目标明确限定为二元匹配判断正确性，不能冒充序数 Score 或 cannot_tell 选项的真实正确性校准。空选择集的准确率保留为缺失，不记作 100%。这些表仍待完整 LLM/Qwen 对照与绘图，不代表 E4 全体验收通过。

'''
    for ds, name in [('wa', 'Walmart–Amazon'), ('ag', 'Amazon–Google'), ('da', 'DBLP–ACM'), ('ab', 'Abt–Buy')]:
        for budget in [50, 200]:
            j = jev[(jev.dataset == ds) & (jev.system == 'J-D+LR') & (jev.budget == str(budget))].iloc[0]
            m = baseline[(baseline.dataset == ds) & (baseline.system == 'M2+best') & (baseline.budget == str(budget))].iloc[0]
            s = stats[(stats.dataset == ds) & (stats.system_A == 'M2+best') & (stats.budget == budget)].iloc[0]
            comparison.append(f'| {name} | {budget} | {j.f1_mean:.2f} | {m.f1_mean:.2f} | {s.delta_B_minus_A:+.2f} [{s.delta_ci_low:+.2f}, {s.delta_ci_high:+.2f}] |')
    body = f'''# Codex 接手修复进展（2026-09-30）

状态：**未冻结，不能据旧总结报告提交论文。** 本文件生成于 {now}；API 计数是生成时的快照，持续状态以运行目录为准。

## 已经能确认的判断

Gemini 的问题同时涉及实现、统计分析、数据划分和结论报告。实体匹配四个规范化数据集可以保留；不能把整个项目判成数据全坏。原始缓存也在本地找到了，因此多数 Jev 结果能从保留的响应重新计算，避免重复付费。

最严重的是 Adult 训练/验证/测试重复，以及 Luna 的概率语义被适配器擅自翻转。前者使旧监督学习测试失去独立性；后者使旧概率和校准比较失效。错误切片、Score/Choice 解码、训练流程和进度报告也需要修复。旧“刷新 SOTA”“全部假设成立”“100% 确定性”“结果冻结”等结论已撤销。

## 本轮已完成

1. 原代码、报告、逐样本 CSV、账本和进度文件已归档；历史缓存以只读方式使用。
2. Jev 主结果按严格概率解码和训练内 OOF 阈值重新计算；补了 7 个有明确缺失或冲突的请求，并区分新观测与旧证据。588 份预测文件的 ID、标签和阈值通过核对，错误切片混淆计数与主表一致。切片目前仍是启发式自动分类，不能替代人工错误原因标注。
3. 修复传统基线的缺失值处理与特征，并完成 LR/RF/DT/SVM 四分类器选择。224 份预测全部通过指标、ID、标签、预算内训练样本和测试隔离检查。它是 Magellan 风格的回退实现，不能标成官方 py_entitymatching 精确复现。
4. Abt–Buy 新建完整记录目录并验证连接，旧错误目录保留。新记录 ID 是内容派生 ID，不能假称上游原始 ID；传统基线直接使用正确的规范化记录。
5. Adult 旧监督测试结论撤回。生成了按整行内容分组去重的新附录实验，清楚标明审计后协议变化；不可与旧文献数字直接作 SOTA 对比。Adult 完整零样本还有 22 个无法追溯的单元格，部分结果不能冒充完整结果。
6. Ditto 已按用户同意生成 40 次训练的输入及预算内验证集，并上传恢复包。**准备输入不等于训练完成**；Qwen、Ditto、NLI 的 GPU 实验都还没有执行。
7. 新 API 保留每次原始响应、请求 ID、缓存键、版本、用量及失败；缓存保留实际 JSON 顺序。新 LLM 提示在设计集检查字段语义后固定，未按测试成绩挑选。16 项针对关键失效模式的测试已通过。Ditto 官方源码固定到提交 `52985564a93fb11308439516d3e17a033d43ec8f`，40 次输入已通过训练程序预检；新增移植代码未执行 GPU 训练，不能称已验证复现。移植依据是 [Ditto 官方仓库](https://github.com/megagonlabs/ditto)，AMP、attention mask 和实际批次数调度的差异均将记录。
8. 错误检测的六组 LLM 历史结果也已回查原始缓存：Hospital、Adult、Flights 合计每模型 6811 个单元格，全部能追溯，概率与旧 CSV 一致。实体匹配 Luna 的概率翻转问题没有出现在这六组错误检测结果里。它们仍只是旧协议中前 200 行的零样本结果，不能替代完整测试或 SOTA 对比。

## 修复后的低标注结果

下表是每个预算 10 次抽样的测试 F1 均值。传统基线的分类器用标注预算内 CV 选择；不根据测试集取 LR 与 best 中较好的成绩。区间为 2000 次配对分层 bootstrap 的 F1 差 95% 描述性区间，尚未完成全部预注册比较的 Holm 校正。

| 数据集 | 标注预算 | Jev 拆题+LR | 传统基线 M2+best | 差值与 95% 区间 |
|---|---:|---:|---:|---|
{chr(10).join(comparison)}

因此只能说三个产品数据集在这些预算点呈现明显优势。DBLP–ACM 的 b=50 没有优势，b=200 对 M2+best 的区间仍跨零。Ditto/Qwen 未完成，不能据此宣称“低标注最佳”。

H1 也不能直接接受：b=200 拆题相对整体问在三个数据集均值较高，但 Walmart–Amazon、Amazon–Google 的区间跨零，DBLP–ACM 反而下降。原假设继续保留，不按修复成绩改写。

Amazon–Google 传统基线全量异常仍在排查：旧 M+LR 的 OOF 阈值结果为 57.29，固定 0.5 为 48.02，接近文献 49.1，说明阈值是重要原因；新 M2+best（主要选 RF）为 66.36，固定 0.5 仍有 60.78，差异不能全归因于阈值。没有足够证据把它标为文献复现通过，停止条件继续保留。

{temperature_text}## API 采集与费用

- Luna/DeepSeek 整体问与拆题特征：{recorded}/{manifest['expected_calls']} 条任务已有记录，其中 {recorded-failed} 条解析成功、{failed} 条失败，{conflicts} 条整体问的文本答案与声称的概率冲突。冲突保留原值，不通过翻转概率掩盖。
- 全量采集后的定点补测：已补回 {repaired} 条，仍有 {failed-repaired} 条已记录失败未补回。定点补测只替换失败任务的“当前可用观测”，保留原失败记录；预算不超过 $0.20，且主采集与该补测合计估计不超过 $8。一旦补测再次失败即停止。程序不会覆写成功预测以改变成绩。
- 当前运行状态：`{manifest['status']}`。有记录不等于有效完成；失败请求保留为缺失，未收齐时程序阻止正式训练与比较。失败请求可能已经计费，不自动盲重试。
- Windows HTTP/2 曾触发 WinError 10035，已回到 HTTP/1.1；这个采集过程不充当 E6 的公平延迟结果。AutoDL 无卡实例据用户确认仍运行、端口未变，但 SSH 在握手阶段断开。域名和真实 IP 均试过，尚不能确定原因。
- 新请求按响应 token 和配置单价计算的费用约 **${cost:.4f}**。这是用量估计，不是供应商账单；失联请求的潜在费用未知。主 API 采集设新增 $8 上限，原项目 $40 总预算仍保留。旧账本 $7.2322 不能认定为已核实的实际支付总额。

| 新运行 | 费用估计 USD | 有用量的账本记录 |
|---|---:|---:|
{chr(10).join(cost_rows)}

## 尚未通过的必需工作

| 项目 | 当前状态 | 完成条件 |
|---|---|---|
| LLM H/D 与 H2 | 正在采集 | 全部 ID 的特征齐全、失败显式补测并留证据、预算匹配训练与统计 |
| Qwen3-8B | 待 GPU | 真正 Yes/No token 概率、推理实现一致性、同题同预算比较 |
| Ditto 40 次 | 输入准备完成，待 GPU | 真实训练/验证/测试、模型与日志留档；全量参考偏差排查 |
| decision-model-preview | 缺 API 配置 | 有凭据后验证并评测；若最终省略，论文对象和标题相应收窄 |
| E4 校准 | Jev 温度/指标区间及选择性预测已重算 | 置信度目标解释、补齐必要对照及图，LR 区间待齐 |
| E5 稳定性 | 原 P2/P5 无效，旧确定性不满足协议 | 实际顺序扰动、Choice 概率、干净模型评估及重复调用 |
| E6 成本与延迟 | 旧比较撤回 | 同地点、匹配任务的受控延迟与完整用量；GPU 成本单列 |
| E7 错误检测 | Adult/Hospital 协议问题已识别 | 完整缺失检查、Hospital 标注预算重建；文献数字只作上下文 |
| E8/E9 | 计数修复／探针仍需审查 | 人工错误原因标注、探针与实际预算/边界/ECE核对 |
| E10/H6 | 未验收 | LLM/Qwen/Ditto 合规范围内补齐；新 Jev 注入测试保持关闭 |
| E11/E12 | 未完成 | 版本漂移与负责人标注按原方案处理，不能省略却称完成 |

## 文件位置与使用规则

- 当前 Jev 结果：`jevwrangle/runs/recovery_repaired_20260930/`。
- 当前传统基线：`jevwrangle/runs/magellan_v2_20260930/`，含 `paired_comparisons.csv` 和 `prediction_integrity_checks.csv`。
- API 原始证据：`jevwrangle/runs/api_llm_recovery_20260930/`。
- GPU 输入：`jevwrangle/data/gpu_recovery_20260930/manifest.json`。
- 校准区间与选择性预测：`jevwrangle/runs/calibration_ci_20260930/`、`jevwrangle/runs/selective_recovery_20260930/`。
- 错误检测 LLM 原始概率核对：`jevwrangle/runs/ed_llm_verify_20260930/`。
- 协议修复登记：`jevwrangle/prereg/recovery_amendment_20260930.md`，不是事前注册。
- 原结果、旧 `reports/tables` 和旧图仍保存作历史记录，**不要混入修复后的论文表**。两份 `PROGRESS.md` 以本次状态替换；旧版在归档包中。

本报告不把任务状态或通过单元测试等同于完整实验验收。下一轮从这些记录断点继续，已有有效观测无需重跑。
'''
    report = ROOT / 'audit_codex_2026-09-30/CODEX_接手修复进展.md'
    report.write_text(body, encoding='utf8')
    progress = f'''# 实验进度（Codex 审计后的修复状态）

更新：{now}。**结果未冻结；撤销旧版“全部通过”的状态。**

旧进度已保存至 `jevwrangle/archive/pre_codex_repair_20260930/legacy_code_reports_progress.zip`。
完整说明见 [Codex 接手修复进展](E:/Desktop/jev/audit_codex_2026-09-30/CODEX_接手修复进展.md)。

| 实验/任务 | 状态 | 当前证据与未完成项 |
|---|---|---|
| E0 来源/接口 | 部分核查 | 原缓存已保全；7 个 Jev 修复调用和 LLM 设计集检查完成；Qwen 一致性待 GPU |
| E1 整体问 | Jev 重算完成，LLM 待新协议 | 撤销旧 Luna 概率语义；显式答案仅作次要结果 |
| E2/E3 归因/学习曲线 | 部分完成 | Jev 588 预测、传统基线 224 预测核查通过；LLM/Qwen/Ditto 缺口未齐 |
| E4 校准 | Jev 区间和选择性预测已重算 | 必要模型对照、置信度目标解释和图待齐 |
| E5 稳定性 | 未通过 | P2/P5 顺序缓存错误，需真实重跑；旧确定性不可接受 |
| E6 成本/延迟 | 未通过 | 撤销硬编码成本和不公平延迟；需受控测量 |
| E7 错误检测 | 未通过 | 撤销 SOTA；Adult 监督测试泄漏、Hospital 行预算错误，附录新协议单列 |
| E8 错误切片 | 重算计数通过 | 混淆计数与主结果一致；仍需人工检查分类原因 |
| E9 探针 | 未通过完整验收 | 样本数、边界和 ECE 修复待齐 |
| E10 注入/H6 | 未通过 | 缺必要模型；新 Jev 注入测试关闭 |
| E11/E12 | 未完成 | 版本漂移、人工原子属性标注 |
| Ditto | PREPARED_NOT_TRAINED | 用户批准 40 次方案，输入已准备上传，待 GPU |
| LLM H/D | {manifest['status']} | {recorded}/{manifest['expected_calls']} 有记录，{failed} 失败；不能把失败算作有效完成 |
| decision-model-preview | 缺配置 | 本地没有密钥，不代表供应商不可用 |
| 全量传统基线参考值 | 未验收 | AG 偏差仍需解释；回退实现不冒称官方精确复现 |

新增 API 用量费用估计约 ${cost:.4f}，失联请求可能费用未知。原 $7.2322 不是已核实供应商支付额。全部必要证据和基线齐全后才冻结。
'''
    for p in [ROOT / 'PROGRESS.md', BASE / 'PROGRESS.md']:
        p.write_text(progress, encoding='utf8')
    print(json.dumps(dict(report=str(report), recorded=recorded, failed=failed, new_estimated_usd=cost), ensure_ascii=False))


if __name__ == '__main__':
    main()
