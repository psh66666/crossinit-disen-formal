#!/usr/bin/env python
"""Post-hoc audit: every old project identity/image is disjoint from the formal set."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = Path("/root/autodl-tmp/crossinit-loc")


def main() -> None:
    ids = set()
    processed = OLD / "data" / "processed"
    if processed.exists():
        ids |= {int(x) for x in os.listdir(processed) if x.isdigit()}
    sources = {}
    for p in (OLD / "data").glob("identities*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        got = set()
        values = obj.values() if isinstance(obj, dict) else [obj]
        for v in values:
            if isinstance(v, list):
                for x in v:
                    try:
                        got.add(int(x))
                    except (TypeError, ValueError):
                        pass
        ids |= got
        if got:
            sources[str(p)] = sorted(got)
    mapping = {}
    for i, line in enumerate((ROOT / "data/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt")
                             .read_text(encoding="utf-8").splitlines()):
        if i == 0:
            continue
        a, _, orig_file = line.split()
        mapping[int(a)] = orig_file
    ident = {}
    for line in (ROOT / "data/annotations/identity_CelebA.txt").read_text(encoding="utf-8").splitlines():
        n, p = line.split()
        ident[n] = int(p)
    old_persons = {ident.get(mapping[i]) for i in ids if i in mapping}
    selected = json.loads((ROOT / "data/disen_formal/identities.json").read_text(encoding="utf-8"))["identities"]
    overlap = sorted(set(x["person_id"] for x in selected) & old_persons)
    out = {
        "historical_ids": sorted(ids), "sources": sources,
        "historical_persons": sorted(x for x in old_persons if x),
        "selected_persons": [x["person_id"] for x in selected],
        "overlap": overlap,
        "note": "Protocol exclusions were applied before selection; this audit adds image ids found in the "
                "old project processed/identities files. Empty overlap means no reselection was needed.",
    }
    (ROOT / "data/disen_formal/history_audit.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"historical_ids": out["historical_ids"], "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
