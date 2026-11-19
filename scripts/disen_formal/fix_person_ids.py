#!/usr/bin/env python
"""Correct HQ image -> CelebA person IDs using the official ``orig_file`` column.

The original selection code used ``orig_idx`` directly against
``identity_CelebA.txt``; the official mapping's filenames are ``orig_idx + 1``
(one-based vs zero-based).  This script repairs metadata only.  It does not
touch source pixels, generated images, weights, or scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.disen_formal.common import DATA_DIR, RAW_DATA, write_json


def load_mapping() -> dict[int, tuple[int, str]]:
    out = {}
    for i, line in enumerate((RAW_DATA / "CelebA-HQ-to-CelebA-mapping.txt")
                             .read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        idx, orig_idx, orig_file = line.split()
        out[int(idx)] = (int(orig_idx), orig_file)
    return out


def load_identity() -> dict[str, int]:
    out = {}
    for line in (ROOT / "data/annotations/identity_CelebA.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, pid = line.split()
        out[name] = int(pid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", default="data/disen_formal/identities.json")
    ap.add_argument("--history", default="data/disen_formal/history_audit.json")
    ap.add_argument("--out-identities", default="data/disen_formal/identities.json")
    ap.add_argument("--out-audit", default="data/disen_formal/history_audit_corrected.json")
    ap.add_argument("--out-table", default="data/disen_formal/person_id_mapping_audit.csv")
    args = ap.parse_args()
    mapping, ident = load_mapping(), load_identity()
    src = Path(args.identities)
    obj = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for item in obj["identities"]:
        image_id = int(item["image_id"])
        orig_idx, orig_file = mapping[image_id]
        if orig_file not in ident:
            raise SystemExit(f"{image_id}: orig_file {orig_file} not in identity file")
        new = ident[orig_file]
        old = int(item["person_id"])
        rows.append({"image_id": image_id, "status": item["status"],
                     "old_person_id_from_orig_idx": old, "correct_person_id": new,
                     "orig_idx": orig_idx, "orig_file": orig_file,
                     "changed": old != new})
        item["person_id_at_training"] = old
        item["person_id"] = new
        item["celeb_orig_idx"] = orig_idx
        item["celeb_orig_file"] = orig_file
    formal = [r for r in rows if r["status"] == "formal"]
    reserve = [r for r in rows if r["status"] == "reserve"]
    if len(set(r["correct_person_id"] for r in formal)) != 8:
        raise SystemExit("formal people are not 8 distinct after correction")
    if len(set(r["correct_person_id"] for r in reserve)) != 2:
        raise SystemExit("reserve people are not 2 distinct after correction")
    if len(set(r["correct_person_id"] for r in rows)) != 10:
        raise SystemExit("formal+reserve people are not 10 distinct after correction")
    obj["mapping_corrected"] = {
        "method": "CelebA-HQ mapping orig_file -> identity_CelebA.txt full filename",
        "reason": "The original code used orig_idx; official filenames are orig_idx + 1.",
        "formal_distinct": 8, "reserve_distinct": 2, "total_distinct": 10,
    }
    out = Path(args.out_identities)
    if out == src:
        shutil.copy2(src, src.with_name("identities_before_person_fix.json"))
    write_json(out, obj)

    # Correct the historical-person audit with the same orig_file rule.
    hist = json.loads(Path(args.history).read_text(encoding="utf-8"))
    hist_ids = [int(x) for x in hist.get("historical_ids", [])]
    hist_rows = []
    for image_id in hist_ids:
        if image_id not in mapping:
            continue
        orig_idx, orig_file = mapping[image_id]
        pid = ident.get(orig_file)
        hist_rows.append({"image_id": image_id, "orig_idx": orig_idx,
                          "orig_file": orig_file, "correct_person_id": pid,
                          "old_person_id_from_orig_idx": ident.get(
                              str(orig_idx).zfill(6) + ".jpg")})
    hist_persons = sorted({r["correct_person_id"] for r in hist_rows if r["correct_person_id"] is not None})
    selected = {r["correct_person_id"] for r in rows}
    overlap = sorted(selected & set(hist_persons))
    corrected = {
        "historical_ids": hist_ids, "historical_rows": hist_rows,
        "historical_persons_correct": hist_persons,
        "selected_persons_correct": [r["correct_person_id"] for r in rows],
        "overlap": overlap,
        "note": "Corrected with orig_file. The original exclusion used orig_idx and was "
                "ineffective as a person filter; the post-hoc audit finds no actual overlap.",
    }
    write_json(Path(args.out_audit), corrected)
    with open(Path(args.out_table), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({"rows": rows, "history_overlap": overlap,
                      "history_persons": hist_persons}, indent=2))
    if overlap:
        raise SystemExit("corrected historical person overlap is non-empty")


if __name__ == "__main__":
    main()
