"""Seal the local computational delivery and verify its links and hashes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re

BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parent
OUT=BASE/'reports/final_delivery_20261001'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if (OUT/'acceptance.json').exists():raise ValueError('Acceptance already exists; preserve revision')
    index=ROOT/'最终成果索引_20261001.md'
    closure=ROOT/'audit_codex_2026-09-30/逐问题闭环_最终验收.md'
    docs=list(OUT.glob('*.md'))+[index,closure]
    nlinks=0
    for p in docs:
        for target in re.findall(r'\]\((E:/[^)]+)\)',p.read_text(encoding='utf8')):
            dest=Path(target)
            if dest==OUT/'acceptance.json':continue
            if not dest.exists():raise ValueError('Broken artifact link '+target)
            nlinks+=1
    (OUT/'acceptance_source.py').write_bytes(Path(__file__).read_bytes())
    inputs=[BASE/'runs/final_integrity_20261001_reviewed/status.json',
        BASE/'runs/final_interval_review_20261001/manifest.json',
        BASE/'runs/manual_shutdown_recovery_20261001/final_gpu_backup_receipt.json',
        BASE/'runs/aggregation_qa_repair_20261001/issue.json',
        BASE/'runs/recovery_repaired_20260930/statistics_crossed_metadata_20260930/manifest.json',
        BASE/'runs/overnight_synthesis_20260930/paired_comparisons/manifest.json',index,closure]
    outputs={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    receipt=dict(status='COMPUTATIONAL_DELIVERY_ACCEPTED_WITH_EXPLICIT_RESEARCH_LIMITATIONS',
        utc=datetime.now(timezone.utc).isoformat(),
        numeric_counts=dict(primary_summary=224,canonical_prediction_seed_groups=1296,matched500_groups=304,flights_groups=80,
            unique_integrity_input_files=3655,integrity_outputs=3,gpu_backup_files=1781),
        gpu=dict(compute='COMPLETE',shutdown='OFFICIAL_SHUTDOWN_SENT_THEN_SSH_UNREACHABLE',
            console_power_and_billing_verified=False,data_disk_released=False),
        api=dict(main_groups_complete=True,new_paid_calls_allowed=False,
            usage_times_price_estimate_usd=2.958075947,account_debit_verified=False,
            wallet='User reports zero; service rejected for insufficient funds',fresh_repeat_branch='PARTIAL_WA_ONLY',second_commercial_model='INCOMPLETE_FREE_QUOTA'),
        visual='Four actual final PNG inspections passed; same matplotlib figure PDF exports; extracted PDF captions also checked',
        links_verified=nlinks,
        remaining=['E8 actual owner review of 30 cases','H5/E12 genuine human semantic gold if retained',
            'Second commercial model full benchmark unavailable under free quota','2027 E11 future drift measurement',
            'Official Magellan AG reference stopping condition not reproduced','Historical preregistration identity not recoverable','Account invoice and AutoDL power/billing authoritative readout unavailable'],
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},outputs=outputs)
    (OUT/'acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    check=json.loads((OUT/'acceptance.json').read_text(encoding='utf8'))
    for rel,expected in check['outputs'].items():assert sha(OUT/rel)==expected
    for rel,expected in check['inputs'].items():assert sha(ROOT/rel)==expected
    print(json.dumps(dict(status=receipt['status'],artifacts=len(outputs),links=nlinks),ensure_ascii=False))

if __name__=='__main__':main()
