"""Check the published evidence files and standalone evidence ZIP."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ARCHIVE = ROOT / "SET-Work-evidence-20261002.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    manifest = json.loads((EVIDENCE / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in manifest["files"]}
    actual = {path.relative_to(EVIDENCE).as_posix() for path in EVIDENCE.rglob("*") if path.is_file() and path.name != "MANIFEST.json"}
    if actual != set(expected):
        raise SystemExit(f"Evidence file set mismatch: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
    for name, row in expected.items():
        data = (EVIDENCE / name).read_bytes()
        if len(data) != row["bytes"] or sha256(data) != row["sha256"]:
            raise SystemExit(f"Evidence hash mismatch: {name}")

    with zipfile.ZipFile(ARCHIVE) as bundle:
        members = {"SET-Work/evidence/" + path.relative_to(EVIDENCE).as_posix(): path for path in EVIDENCE.rglob("*") if path.is_file()}
        if set(bundle.namelist()) != set(members) or bundle.testzip() is not None:
            raise SystemExit("Evidence ZIP membership or CRC mismatch")
        for name, path in members.items():
            if sha256(bundle.read(name)) != sha256(path.read_bytes()):
                raise SystemExit(f"Evidence ZIP content mismatch: {name}")

    paper_rows = list(csv.reader(io.StringIO((EVIDENCE / "paper_20260926/results_summary/table_04.csv").read_text(encoding="utf-8"))))
    paper_tasks = [row[0] for row in paper_rows if row and row[0].startswith("T") and row[0][1:].isdigit()]
    if paper_tasks != [f"T{i:02d}" for i in range(1, 19)]:
        raise SystemExit("Retained 18-task table changed")

    replay = list(csv.DictReader(io.StringIO((EVIDENCE / "analysis/paper21x6_two_models_20260925/PAPER_BASELINE_REPLAY.csv").read_text(encoding="utf-8-sig"))))
    for method in ("original_straight", "paper_curve"):
        if {row["task"] for row in replay if row["method"] == method} != {f"T{i:02d}" for i in range(1, 22)}:
            raise SystemExit(f"Original 21-task replay incomplete: {method}")

    scores = json.loads((EVIDENCE / "analysis/all_methods126_20260926/all_comparator_scores.json").read_text(encoding="utf-8"))
    if len(scores) != 4284 or sum(row["status"] != "ok" for row in scores) != 10:
        raise SystemExit("126-case score roster or preserved failures changed")
    ledger = json.loads((EVIDENCE / "analysis/repaired126_20260927/case_ledger.json").read_text(encoding="utf-8"))
    if len(ledger) != 126:
        raise SystemExit("Repaired 126-case ledger incomplete")
    for name in ("SETSUNET_ICLR_manuscript.pdf", "supplement.pdf"):
        if not (EVIDENCE / "paper_20260926" / name).read_bytes().startswith(b"%PDF-"):
            raise SystemExit(f"Invalid paper PDF: {name}")

    print(json.dumps({"status": "passed", "manifest_files": len(expected), "archive_files": len(members), "paper_tasks": 18, "historical_tasks": 21, "all_method_score_rows": 4284, "preserved_failures": 10, "repaired_cases": 126}))


if __name__ == "__main__":
    main()
