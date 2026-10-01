"""Create explicitly AI-reviewed E8 and blinded, unlabelled E12 forms.

These files are audit aids. Neither is a completed human annotation study.
"""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'runs/free_completion_20261001/review_packets'

REVIEWS={
1:('Model/ID Mismatch','supported','两侧型号 NN-SD797S 与 NN-SD767S 明确不同；缺失描述并非唯一线索。'),
2:('Semantic Normalization','supported_broad','标题存在 clent/client、from/form 拼写差异；作者组织名称与会议名存在不同写法。'),
3:('Ambiguous From Supplied Fields','insufficient','两侧软件名与版本均为 Intellihance Pro 4.1；仅凭所给字段无法说明为何基准标为不匹配。不能因此改标签。'),
4:('Ambiguous From Supplied Fields','insufficient','两侧均为 Mavis Beacon 17 Deluxe；缺厂商是真实字段现象，但不足以解释基准负例。'),
5:('Package/Edition Difference','more_specific','large box 与 deluxe/mass merchant box 提供包装或版本差异线索；旧规则只归为缺厂商过于宽泛。'),
6:('Venue Alias','supported_broad','标题、作者、年份相同，会议字段为 SIGMOD 与全称。'),
7:('Author/Venue Normalization','supported_broad','作者顺序、名字简称与 HTML 字符编码不同，论文标题与年份相同。'),
8:('Product-Type Mismatch','more_specific','high school 与 elementary school 指向不同学习阶段的软件；缺厂商不是最具体差异。'),
9:('Ambiguous Licence/SKU','insufficient','两侧都写 PaperPort 11、5 users；license/complete package 可能有差别，但文本不足以确定。'),
10:('Ambiguous From Supplied Fields','insufficient','两侧软件名称均为 Adobe CS3 Web Standard；文本未提供可靠的型号或许可区别。'),
11:('Venue Alias','supported_broad','论文标题、作者、年份相同，会议简称与全称不同。'),
12:('Publication-Version Difference','more_specific','相同标题对应 1997 VLDB 与 2002 TODS，不同年份和会议/期刊均可观察；不能仅归因年份。'),
13:('Title/Author Normalization','supported_broad','panel 标记、Özsu 的 HTML 编码不同；基准为正例。'),
14:('Missing Identifier','supported','一侧含 WCC7204RK，另一侧无型号；品牌、商品类别与标题相近。'),
15:('Title Paraphrase/Encoding','supported_broad','open storage system for abstract objects 与 open abstract-object storage system 为词序改写，作者与会议字段亦需规范化。'),
16:('Ambiguous From Supplied Fields','insufficient','Safekeeper Plus 名称和厂商可对齐，缺少能解释负例的版本/SKU 信息。'),
17:('Ambiguous From Supplied Fields','insufficient','两侧标题完全一致；缺厂商不足以确认不匹配，更不能自动认定标签错误。'),
18:('Ambiguous From Supplied Fields','insufficient','Mobile Media Converter 与 Pinnacle 字段为重排，所给字段不足以解释负例。'),
19:('Ambiguous From Supplied Fields','insufficient','两侧 Network Now! Pro 名称一致，第二侧有 SKU/操作系统而第一侧未列，无法据此确证不匹配。'),
20:('Author-List/Venue Normalization','supported_broad','相同标题和年份，第二侧多列两位作者；不能把作者列表不完全相同自动当负例。'),
21:('Insufficient Model Evidence','insufficient','基准为正例，但第二侧只给通用 LG 24-inch washer/dryer 描述，缺少 WM3431W 编码；这是不可判定线索，不是已证实标签噪声。'),
22:('Version Underspecification','more_specific','第二侧写 Music Studio 7，第一侧没有版本号；缺少版本并不证明不同。'),
23:('Package/Brand Information','supported','两侧均为 Microsoft Money Home and Business 2007，包装/价格不同，缺厂商规则描述成立但不应判定有因果解释。'),
24:('Upgrade/Edition Difference','more_specific','第二侧明确 upgrade，第一侧没有；版本/许可差异比缺厂商更具体。'),
25:('Ambiguous From Supplied Fields','insufficient','两侧 Popcorn 2 软件名一致，没有足够字段解释基准负例。'),
26:('Ambiguous From Supplied Fields','insufficient','两侧均有 SubmitWolf 6.0 与 SWOLF6R；厂商表述与操作系统附加文本不足以解释负例。'),
27:('Pack-Quantity Difference','more_specific','第二侧明确 pack of2，第一侧为单个计算器描述；这比缺型号更直接。'),
28:('Licence/Full-Package Ambiguity','insufficient','第一侧 lic only、第二侧 full pk，却为基准正例；需要人核验匹配粒度，不能自动改标签。'),
29:('Publication-Version Difference','more_specific','相同标题在 2001 VLDB 与 2002 VLDB Journal，年份和出版形态共同变化。'),
30:('Size/Metadata Ambiguity','insufficient','5 与 5.25 可能是规格差异或简称；category/brand 明显不一致，仅凭字段不能确定不匹配原因。')
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=BASE/'reports/overnight_recovery_20261001_reviewed/owner_30_review_packet.jsonl'
    originals=[json.loads(s) for s in source.read_text(encoding='utf8').splitlines()]
    assert len(originals)==30 and set(REVIEWS)==set(p['review_id'] for p in originals)
    reviewed=[];blind=[]
    for x in originals:
        category,assessment,note=REVIEWS[x['review_id']]
        reviewed.append(dict(**x,ai_category=category,ai_rule_assessment=assessment,ai_observation=note,
            reviewer_identity='Codex AI; not a human reviewer',human_review_completed=False))
        blind.append(dict(review_id=x['review_id'],dataset=x['dataset'],pair_id=x['pair_id'],
            record_a=json.dumps(x['record_a'],ensure_ascii=False),record_b=json.dumps(x['record_b'],ensure_ascii=False),
            human_category='',human_match_judgment='',human_note=''))
    (OUT/'E8_AI逐例复核.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in reviewed)+'\n',encoding='utf8')
    with (OUT/'E8_负责人盲审.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(blind[0]));writer.writeheader();writer.writerows(blind)
    path=BASE/'data/canonical/wa/test.jsonl'
    pairs=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
    pairs.sort(key=lambda p:hashlib.sha256(('E12_POST_AUDIT_20261001|'+p['pair_id']).encode()).hexdigest())
    chosen=pairs[:150]
    with (OUT/'E12_WA150_人工标注空表.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['pair_id','record_a','record_b','same_brand','same_model_id','variant_mismatch','note']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in chosen:
            writer.writerow(dict(pair_id=p['pair_id'],record_a=json.dumps(p['record_a'],ensure_ascii=False),
                record_b=json.dumps(p['record_b'],ensure_ascii=False),same_brand='',same_model_id='',variant_mismatch='',note=''))
    instructions='''# 人工复核使用说明\n\nE8 的 AI 复核已经逐例完成，但不是负责人人工复核，也不是新标签真值。AI 无法从文本确定的案例明确保留不可判定，不改变任何官方标签、主表或训练模型。机器规则描述的是可观察字段模式，不证明模型错误的原因。\n\nE8_负责人盲审.csv 隐藏系统、预测、基准标签和机器规则。请先按记录填 human_match_judgment（1/0/?）、类别和说明，再与原包对照。不能把这30例当随机总体错误率：它们来自原抽取错误池。\n\nE12 空表含150个WA官方测试对、450个问题判断。所有标签留空，请只依记录填1（是）、0（否）、?（无法判断）。same_brand：同品牌/厂商；same_model_id：规范化后同型号，缺少可确认的编码填?；variant_mismatch：尺寸、容量、颜色、包装数量或版本有明确差异填1。保留?的数量和覆盖率，不把?当0。\n\n新E12抽样在模型结果已经查看之后建立，按固定ID哈希排序选择，不按预测或错误挑选；这是审计后探索性协议，不能追认成原预注册。即使未来标完，也缺完整DM比较，且语义/数值题难度不同，不能直接证明机制因果。\n\n本轮没有人类标注，E12正式检验仍未完成；目前论文正文不依赖它。\n'''
    (OUT/'人工复核说明.md').write_text(instructions,encoding='utf8')
    manifest=dict(status='AI_REVIEW_AND_BLINDED_FORMS_COMPLETE_HUMAN_PENDING',n_E8_ai=30,n_E8_human=0,
        n_E12_pairs=150,n_E12_human_labels=0,paid_api_calls=0,gpu_calls=0,
        inputs={str(source):sha(source),str(path):sha(path),str(Path(__file__)):sha(Path(__file__))},
        sampling='SHA256(E12_POST_AUDIT_20261001|pair_id); first150; defined after examining model results')
    manifest['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({k:manifest[k] for k in ['status','n_E8_ai','n_E8_human','n_E12_pairs','n_E12_human_labels']}))


if __name__=='__main__':main()
