"""Create a traceable paper-readiness packet from accepted offline artifacts."""
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
PROJECT=BASE.parent
ROOT=BASE/'runs/free_completion_20261001'
OUT=PROJECT/'论文/零新增费用补验与投稿评估_20261001'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def csvwrite(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert not (OUT/'readiness.json').exists(),'Existing accepted packet; do not overwrite'
    # The initial unaccepted packet stopped at its prospective manifest link.
    # Reuse the staged table directory only while no accepted readiness exists.
    (OUT/'tables').mkdir(exist_ok=True)
    inputs={}
    stages=['legacy_diagnostics','legacy_independent_readouts','magellan_author_library',
            'baseline_independent_verification','review_packets','baseline_population_supplements']
    for name in stages:
        folder=ROOT/name;m=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        for label,digest in m['inputs'].items():
            p=Path(label);assert sha(p)==digest,str(p);inputs[str(p)]=digest
        for label,digest in m['outputs'].items():
            p=folder/label;assert sha(p)==digest,str(p);inputs[str(p)]=digest
        inputs[str(folder/'manifest.json')]=sha(folder/'manifest.json')
    old=BASE/'reports/final_delivery_20261001'
    acceptance=json.loads((old/'acceptance.json').read_text(encoding='utf8'))
    for key,digest in acceptance['outputs'].items():assert sha(old/key)==digest,key
    for key,digest in acceptance['inputs'].items():assert sha(PROJECT/key)==digest,key
    inputs[str(old/'acceptance.json')]=sha(old/'acceptance.json')
    copies={
        'M3_summary.csv':ROOT/'magellan_author_library/summary.csv',
        'M3_metrics_by_seed.csv':ROOT/'magellan_author_library/metrics_by_seed.csv',
        'M3_reference_deviation.csv':ROOT/'magellan_author_library/reference_deviation.csv',
        'Jev_vs_M3_crossed_bootstrap.csv':ROOT/'baseline_independent_verification/paired_bootstrap.csv',
        'legacy_raw_coverage_and_metrics.csv':ROOT/'legacy_diagnostics/coverage_and_metrics.csv',
        'legacy_separate_readouts.csv':ROOT/'legacy_independent_readouts/separate_readouts.csv',
        'legacy_rawfield_injection_exact_binomial.csv':ROOT/'legacy_diagnostics/injection_exact_binomial.csv',
        'legacy_answer_injection_exact_binomial.csv':ROOT/'legacy_independent_readouts/answer_injection_exact_binomial.csv',
        'legacy_ED_verified.csv':BASE/'runs/ed_llm_verify_20260930/metrics.csv',
        'M3_matched500_summary.csv':ROOT/'baseline_population_supplements/matched500_summary.csv',
        'M3_matched500_by_seed.csv':ROOT/'baseline_population_supplements/matched500_by_seed.csv',
        'M3_content_sensitivity_summary.csv':ROOT/'baseline_population_supplements/content_sensitivity_summary.csv',
        'Jev_vs_M3_content_filtered_bootstrap.csv':ROOT/'baseline_population_supplements/content_filtered_paired_bootstrap.csv'}
    for name,p in copies.items():inputs[str(p)]=sha(p);shutil.copyfile(p,OUT/'tables'/name)
    existing=read(old/'tables/full_system_summary.csv')
    new=read(ROOT/'magellan_author_library/summary.csv')
    keys={(r['dataset'],r['system'],r['budget']):float(r['f1_mean']) for r in existing+new}
    assert len(keys)==len(existing)+len(new)
    systems=['J-H1+LR','J-D+LR','Q-D+LR','L2-D+LR','M2+LR','M2+best','M3-author+LR','M3-author+best','Ditto']
    table=[]
    for budget in ['50','200']:
        for ds in ['wa','ag','da','ab']:
            row=dict(dataset=ds,budget=budget)
            row.update({s:keys[(ds,s,budget)] for s in systems});table.append(row)
    csvwrite(OUT/'tables/full_test_low_budget_comparison.csv',table)
    matched=read(old/'tables/matched_500_summary.csv')+read(ROOT/'baseline_population_supplements/matched500_summary.csv')
    mk={(r['dataset'],r['system'],r['budget']):float(r['f1_mean']) for r in matched}
    assert len(mk)==len(matched)
    rows=[]
    for ds in ['wa','ag','da','ab']:
        row=dict(dataset=ds,budget=200,n_test=500,n_seeds=3)
        row.update({s:mk[(ds,s,'200')] for s in ['J-D+LR','Q-D+LR','L1-D+LR','L2-D+LR','M3-author+LR','M3-author+best','Ditto']})
        rows.append(row)
    csvwrite(OUT/'tables/matched500_b200_comparison.csv',rows)
    md=['# 可直接引用的新增对照表','','F1 均为百分数。新 M3 为审计后作者库回退；原主结果来自已验收唯一主表。',
        '完整测试表中 Jev/Qwen/DeepSeek/M2/M3 为十种子，Ditto 为三种子；Luna 只在共同500对表出现。','',
        '## 完整测试低预算','', '| dataset | b | '+' | '.join(systems)+' |',
        '| --- | --- | '+' | '.join(['---']*len(systems))+' |']
    for r in table:md.append('| '+r['dataset']+' | '+r['budget']+' | '+' | '.join(f'{r[s]:.3f}' for s in systems)+' |')
    matchedsystems=list(rows[0])[4:]
    md.extend(['','## 同500对、b200、同三种子','', '| dataset | '+' | '.join(matchedsystems)+' |','| --- | '+' | '.join(['---']*len(matchedsystems))+' |'])
    for r in rows:md.append('| '+r['dataset']+' | '+' | '.join(f'{r[s]:.3f}' for s in matchedsystems)+' |')
    md.extend(['','## Jev 拆题减作者库字符串基线：全部16组','',
               '| dataset | b | baseline | ΔF1 | 描述性95%下界 | 描述性95%上界 |',
               '| --- | --- | --- | --- | --- | --- |'])
    for r in read(ROOT/'baseline_independent_verification/paired_bootstrap.csv'):
        md.append('| '+r['dataset']+' | '+r['budget']+' | '+r['system_A']+' | '+
                  ' | '.join(f'{float(r[k]):.3f}' for k in ['delta_B_minus_A','delta_ci_low','delta_ci_high'])+' |')
    md.extend(['','区间采用共同测试行和种子的交叉 bootstrap，2,000 次。不是已完成的完整原假设族 Holm 检验；不使用尾比例充当 p 值。',
               '各表对应 CSV 均在本目录 tables；更完整逐样本、OOF 和模型见 runs/free_completion_20261001/magellan_author_library。',''])
    (OUT/'新增对照表.md').write_text('\n'.join(md),encoding='utf8')
    # A concrete entrypoint, with all paths verified below.
    index=PROJECT/'零新增费用补验成果索引_20261001.md'
    index.write_text('''# 零新增费用补验：已核验成果

可以用现有数据写完整的 Jev 低标注实体匹配实证论文。本次新增付费API/GPU为零，原预注册、人工标注和官方Magellan复现未冒称全部通过。

- [结论与期刊建议](E:/Desktop/jev/论文/零新增费用补验与投稿评估_20261001/结论与期刊建议.md)：可写范围、未完成项处理、JIIS/KAIS/DSE的官方费用来源。
- [论文论证蓝图](E:/Desktop/jev/论文/零新增费用补验与投稿评估_20261001/论文论证蓝图.md)：章节、证据映射、中英文摘要草稿。
- [新增对照表](E:/Desktop/jev/论文/零新增费用补验与投稿评估_20261001/新增对照表.md)：低预算全测、共同500对及全部16个新基线区间。
- [补验结果与限制](E:/Desktop/jev/论文/零新增费用补验与投稿评估_20261001/补验结果与限制.md)：19,800原始返回、224作者库基线、人工材料边界。
- [文献核验表](E:/Desktop/jev/论文/零新增费用补验与投稿评估_20261001/文献核验表.md)：相关工作与创新限制。
- [readiness.json](E:/Desktop/jev/论文/零新增费用补验与投稿评估_20261001/readiness.json)：本次验收计数与SHA。
- [E8负责人盲审表](E:/Desktop/jev/jevwrangle/runs/free_completion_20261001/review_packets/E8_负责人盲审.csv)；[E12空白标注表](E:/Desktop/jev/jevwrangle/runs/free_completion_20261001/review_packets/E12_WA150_人工标注空表.csv)。
- [原夜间成果索引](E:/Desktop/jev/最终成果索引_20261001.md)：GPU备份、原主结果、四张已验收图及逐问题闭环。

新结果保存在独立目录，未覆盖旧验收；人工E8/E12、第二商业模型、官方Magellan和未来E11仍明确留空。已经提供写作依据与摘要蓝图，尚未完成一篇投稿全文，也未投稿。
''',encoding='utf8')
    local=[]
    for p in [index]+list(OUT.glob('*.md')):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if target.startswith(('https://','http://')):continue
            t=Path(target.strip('<>'))
            if t != OUT/'readiness.json':assert t.exists(),(p,target)
            local.append(str(t))
    outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    outputs['source.py']=sha(Path(__file__))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    inputs[str(index)]=sha(index)
    packet=dict(status='ZERO_NEW_API_GPU_COST_READY_FOR_SCOPED_MANUSCRIPT_WRITING',utc=datetime.now(timezone.utc).isoformat(),
        added_paid_api_calls=0,added_gpu_calls=0,additional_api_gpu_usd=0,
        old_accepted_output_hashes_verified=len(acceptance['outputs']),
        n_legacy_groups=50,n_legacy_raw_rows=19800,n_baseline_groups=224,n_baseline_comparisons=16,
        n_new_matched500_groups=48,n_new_content_filtered_groups=224,n_new_filtered_comparisons=16,
        n_AI_E8_reviews=30,n_human_E8_reviews=0,n_E12_pairs_prepared=150,n_human_E12_labels=0,
        local_links_verified=len(local),unique_input_hashes_verified=len(inputs),inputs=inputs,outputs=outputs,
        manuscript_completed=False,submitted=False,primary_preregistration_completed=False,
        figure_status='Reuse four previously visually accepted figures; no new plots generated; new M3 shown as separate tables',
        preferred_journal='Journal of Intelligent Information Systems; subscription route no APC per official current page',
        remaining=['Real owner review if human-verified causal errors are claimed',
                   'H5/E12 human semantic gold omitted from current scoped claim',
                   'Second commercial model omitted, not completed',
                   'Official Magellan reference gate unresolved; author-library independent fallback only',
                   'Future E11 not measured','Author responsibility/data rights and full manuscript drafting'])
    (OUT/'readiness.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False),encoding='utf8')
    assert all(Path(p).exists() for p in local)
    print(json.dumps({k:packet[k] for k in ['status','unique_input_hashes_verified','old_accepted_output_hashes_verified','local_links_verified']},ensure_ascii=False))


if __name__=='__main__':main()
