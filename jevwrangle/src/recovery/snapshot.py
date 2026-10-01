"""Preserve pre-repair code/results without copying secrets or raw API data."""
import hashlib
import json
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DEST = BASE / "archive" / "pre_codex_repair_20260930"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "legacy_code_reports_progress.zip"
    if target.exists():
        raise FileExistsError("Snapshot already exists; never overwrite the pre-repair snapshot")
    files = []
    for folder in ["src", "reports", "prereg", "questions", "config"]:
        files.extend(p for p in (BASE / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.extend(p for p in [BASE / "PROGRESS.md", BASE.parent / "PROGRESS.md"] if p.exists())
    manifest = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            rel = p.relative_to(BASE.parent).as_posix()
            archive.write(p, rel)
            manifest[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["jevwrangle/cache/calls.sqlite"] = hashlib.sha256((BASE / "cache/calls.sqlite").read_bytes()).hexdigest()
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"SNAPSHOT {target} files={len(files)}")


if __name__ == "__main__":
    main()
