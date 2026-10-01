"""Final read-only hash, cohort-count, numeric-claim and human-status checks."""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
PROJECT=BASE.parent
OUT=PROJECT/'论文/零新增费用补验与投稿评估_20261001'


def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    m=json.loads((OUT/'readiness.json').read_text(encoding='utf8'))
    for name,digest in m['inputs'].items():assert sha(Path(name))==digest,name
    for name,digest in m['outputs'].items():assert sha(OUT/name)==digest,name
    assert m['added_paid_api_calls']==m['added_gpu_calls']==m['additional_api_gpu_usd']==0
    coverage=read(OUT/'tables/legacy_raw_coverage_and_metrics.csv')
    assert len(coverage)==50 and sum(int(r['n_verified']) for r in coverage)==19800
    assert all(int(r['n_missing'])==0 for r in coverage)
    assert sum(int(r['n_changed']) for r in coverage)==7909
    assert sum(int(r['n_changed']) for r in coverage if r['model']=='gpt-6-luna')==6429
    assert sum(int(r['n_changed']) for r in coverage if r['model']=='deepseek-v4-flash')==1480
    assert len(read(OUT/'tables/M3_metrics_by_seed.csv'))==224
    assert len(read(OUT/'tables/M3_matched500_by_seed.csv'))==48
    assert len(read(OUT/'tables/Jev_vs_M3_crossed_bootstrap.csv'))==16
    assert len(read(OUT/'tables/Jev_vs_M3_content_filtered_bootstrap.csv'))==16
    for name in ['Jev_vs_M3_crossed_bootstrap.csv','Jev_vs_M3_content_filtered_bootstrap.csv']:
        for r in read(OUT/'tables'/name):
            assert int(r['n_seeds'])==10 and int(r['replicates'])==2000
            if r['dataset'] in ['wa','ag','ab']:assert float(r['delta_ci_low'])>0
            else:assert float(r['delta_ci_low'])<0<float(r['delta_ci_high'])
    review=BASE/'runs/free_completion_20261001/review_packets'
    human=read(review/'E12_WA150_人工标注空表.csv')
    assert len(human)==150
    assert m['n_human_E8_reviews']==m['n_human_E12_labels']==0
    result=dict(status='FINAL_PACKET_HASH_COUNTS_AND_REPORTED_CONCLUSIONS_PASSED',
                input_hashes=len(m['inputs']),output_hashes=len(m['outputs']),
                manuscript_completed=m['manuscript_completed'],submitted=m['submitted'],
                paid_api_gpu_usd=0,human_annotation_pending=True,
                readiness_sha256=sha(OUT/'readiness.json'),source_sha256=sha(Path(__file__)))
    p=BASE/'runs/free_completion_20261001/final_packet_check.json'
    assert not p.exists(),'Preserve existing final check'
    p.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
