#!/usr/bin/env python
"""Final run manifest: step coverage, unique steps, timestamps, final hashes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_state import sha256_file
from scripts.disen_formal.common import formal_identities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/disen_formal_v1/run_manifest.csv")
    args = ap.parse_args()
    rows = []
    for ident in formal_identities():
        image_id = str(ident["image_id"])
        for group in ("B", "L", "D", "A"):
            run = ROOT / "runs/disen_formal_v1" / image_id / group
            mets = [json.loads(x) for x in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                    if x.strip()]
            steps = [int(r["step"]) for r in mets]
            expected = 320 if group == "B" else 2000
            final = run / "final" / "trainable.pt"
            meta = json.loads((run / "final" / "meta.json").read_text(encoding="utf-8"))
            rows.append({
                "image_id": image_id, "group": group, "updates": len(steps),
                "first_step": min(steps), "last_step": max(steps),
                "unique_steps": len(set(steps)), "expected_last": expected,
                "contiguous": int(sorted(set(steps)) == list(range(1, expected + 1))),
                "last_micro_step": mets[-1].get("micro_step", ""),
                "first_time": mets[0].get("time", ""), "last_time": mets[-1].get("time", ""),
                "final_sha256": sha256_file(final), "meta_step": meta.get("step"),
                "parent_step": meta.get("parent_step", ""),
            })
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if not r["contiguous"] or r["unique_steps"] != r["updates"]]
    print(json.dumps({"jobs": len(rows), "bad": bad}, indent=2))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
