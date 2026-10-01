"""Verify public files; standard library only, no network or execution queues."""
import csv
import hashlib
from pathlib import Path
root = Path(__file__).resolve().parent
with (root / "PUBLIC_FILE_MANIFEST.csv").open(encoding="utf8", newline="") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    path = root / row["path"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    assert h.hexdigest() == row["sha256"], row["path"]
    assert path.stat().st_size == int(row["bytes"]), row["path"]
print("PASS: {} released files match SHA-256 and size; no API/GPU calls.".format(len(rows)))
