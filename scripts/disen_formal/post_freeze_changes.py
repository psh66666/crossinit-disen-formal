#!/usr/bin/env python
"""Record which code files changed after the training freeze and why."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_state import sha256_file, write_json

REASONS = {
    "crossinit/disen_adapter.py": "cosine direction comment corrected; no computation change",
    "scripts/disen_formal/common.py": "bf16-safe tensor hash; no training computation change",
    "scripts/disen_formal/generate.py": "add --image-ids for smoke test / targeted regeneration",
    "scripts/disen_formal/check.py": "coverage checks final-state key scope; engineering wording",
    "scripts/disen_formal/aggregate.py": "corrected human aggregation, subset counts, conclusion wording",
    "scripts/disen_formal/report.py": "corrected results wording, reference, limitations",
    "scripts/disen_formal/package.py": "archive manifest filename and history audit inclusion",
    "scripts/disen_formal/deviations.py": "added audit-driven deviations",
    "scripts/disen_formal/blind.py": "repeat-key swap and page-order fixes",
    "scripts/disen_formal/fix_blind_key.py": "new repair/audit tool",
    "scripts/disen_formal/human_eval.py": "new corrected human scoring module",
    "scripts/disen_formal/audit_human_eval.py": "new synthetic acceptance tests",
    "scripts/disen_formal/fix_person_ids.py": "new orig_file person-ID correction",
    "scripts/disen_formal/history_audit.py": "orig_file mapping correction",
    "scripts/disen_formal/data.py": "orig_file mapping correction",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="locks/disen_formal_v1/freeze.json")
    ap.add_argument("--out", default="locks/disen_formal_v1/post_freeze_changes.json")
    args = ap.parse_args()
    freeze = json.loads((ROOT / args.freeze).read_text(encoding="utf-8"))
    old = freeze.get("code", {})
    rows = []
    for rel, old_hash in sorted(old.items()):
        p = ROOT / rel
        new_hash = sha256_file(p) if p.exists() else None
        if new_hash != old_hash:
            rows.append({"path": rel, "old_sha256": old_hash, "current_sha256": new_hash,
                         "reason": REASONS.get(rel, "analysis/tooling change; training math unchanged")})
    current = {}
    for d in ("crossinit", "scripts/disen_formal"):
        for p in sorted((ROOT / d).rglob("*.py")):
            current[str(p.relative_to(ROOT))] = sha256_file(p)
    result = {
        "freeze_hash": freeze.get("freeze_hash"),
        "changed": rows,
        "current_code": current,
        "training_math_changed": False,
        "note": "Formal training runs were not restarted. Changed files are inference/evaluation/"
                "documentation/tooling only; no loss, optimizer, data or model computation changed.",
    }
    write_json(ROOT / args.out, result)
    print(json.dumps({"changed": len(rows), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
