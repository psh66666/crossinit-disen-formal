#!/usr/bin/env python
"""Semantic tests for the human blind-rating aggregation.

These tests use synthetic ratings and the repaired key; they do not inspect the
real reviewer's answers.  They are the acceptance tests required by the
2026-11-19 audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.disen_formal.human_eval import human_summary, read_csv, task_method_pref
from scripts.disen_formal.common import write_json


def write_ratings(path: Path, key_obj: dict, mode: str) -> None:
    fields = ["task_id", "identity_preference", "edit_success_A", "edit_success_B",
              "severe_artifact_A", "severe_artifact_B", "joint_preference"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for tid in key_obj["order"]:
            e = key_obj["tasks"][tid]
            groups = e["groups"]
            if mode == "all_d" and "D" in groups.values():
                side = "A" if groups["A"] == "D" else "B"
                pref = side
            elif mode == "all_d" and "L" in groups.values():
                side = "A" if groups["A"] == "L" else "B"
                pref = side
            elif mode == "all_tie":
                pref = "tie"
            else:
                pref = "A"
            w.writerow({"task_id": tid, "identity_preference": pref,
                        "edit_success_A": "yes", "edit_success_B": "yes",
                        "severe_artifact_A": "no", "severe_artifact_B": "no",
                        "joint_preference": pref})


def mirrored_key(key_obj: dict) -> dict:
    out = {"order": key_obj["order"], "tasks": {}}
    for tid, e in key_obj["tasks"].items():
        n = dict(e)
        n["groups"] = {"A": e["groups"]["B"], "B": e["groups"]["A"]}
        n["A"], n["B"] = e["B"], e["A"]
        n["A_key"], n["B_key"] = e.get("B_key"), e.get("A_key")
        out["tasks"][tid] = n
    return out


def mirrored_ratings(key_obj: dict, ratings: Path, out: Path) -> None:
    rows = {r["task_id"]: r for r in read_csv(ratings)}
    fields = ["task_id", "identity_preference", "edit_success_A", "edit_success_B",
              "severe_artifact_A", "severe_artifact_B", "joint_preference"]
    flip = {"A": "B", "B": "A", "tie": "tie", "": ""}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for tid in key_obj["order"]:
            r = rows[tid]
            w.writerow({"task_id": tid,
                        "identity_preference": flip[r["identity_preference"]],
                        "edit_success_A": r["edit_success_B"],
                        "edit_success_B": r["edit_success_A"],
                        "severe_artifact_A": r["severe_artifact_B"],
                        "severe_artifact_B": r["severe_artifact_A"],
                        "joint_preference": flip[r["joint_preference"]]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="results/disen_formal_v1/blind/key.json")
    ap.add_argument("--out", default="results/disen_formal_v1/human_eval_audit.json")
    args = ap.parse_args()
    key_obj = json.loads(Path(args.key).read_text(encoding="utf-8"))
    if len(key_obj["order"]) != 176 or len(key_obj["tasks"]) != 176:
        raise SystemExit("key must contain 176 tasks")
    tmp = Path(tempfile.mkdtemp(prefix="human_eval_audit_"))
    results = {}

    # 1. All D-side: D must win 100% of D-B and D-L joint/identity preferences.
    r = tmp / "all_d.csv"; write_ratings(r, key_obj, "all_d")
    h = human_summary(r, Path(args.key))
    results["all_d"] = {
        "status": h["status"],
        "D-B-joint": h["preference"]["B-D"]["joint"]["win_rate"],
        "D-B-identity": h["preference"]["B-D"]["identity"]["win_rate"],
        "D-L-joint": h["preference"]["D-L"]["joint"]["win_rate"],
        "D-L-identity": h["preference"]["D-L"]["identity"]["win_rate"],
        "D-B-pairs": h["preference"]["B-D"]["joint"]["n_pairs"],
        "L-B-pairs": h["preference"]["B-L"]["joint"]["n_pairs"],
        "D-L-pairs": h["preference"]["D-L"]["joint"]["n_pairs"],
        "repeat_joint": h["repeat_consistency"]["joint_agreement"],
        "unique_images": h["unique_images_scored"],
        "edit_n": {g: h["edit_success"][g]["n"] for g in ("B", "L", "D")},
    }
    assert abs(results["all_d"]["D-B-joint"] - 1.0) < 1e-12
    assert abs(results["all_d"]["D-B-identity"] - 1.0) < 1e-12
    assert abs(results["all_d"]["D-L-joint"] - 1.0) < 1e-12
    assert results["all_d"]["D-B-pairs"] == 64
    assert results["all_d"]["L-B-pairs"] == 64
    assert results["all_d"]["D-L-pairs"] == 32
    assert abs(results["all_d"]["repeat_joint"] - 1.0) < 1e-12
    assert results["all_d"]["unique_images"] == 192
    assert all(results["all_d"]["edit_n"][g] == 48 for g in ("B", "L", "D"))

    # 2. All ties: contribution is 0.5, never 1.0.
    r2 = tmp / "all_tie.csv"; write_ratings(r2, key_obj, "all_tie")
    h2 = human_summary(r2, Path(args.key))
    results["all_tie"] = {c: h2["preference"][c]["joint"]["win_rate"]
                          for c in ("B-D", "B-L", "D-L")}
    assert all(abs(v - 0.5) < 1e-12 for v in results["all_tie"].values())

    # 3. Mirror images + key + ratings: method-level result is invariant.
    mk = mirrored_key(key_obj)
    mkey = tmp / "mirrored_key.json"
    write_json(mkey, mk)
    mr = tmp / "mirrored_ratings.csv"
    mirrored_ratings(key_obj, r, mr)
    hm = human_summary(mr, mkey)
    results["mirror"] = {}
    for comp in ("B-D", "B-L", "D-L"):
        for field in ("identity", "joint"):
            a = h["preference"][comp][field]["win_rate"]
            b = hm["preference"][comp][field]["win_rate"]
            results["mirror"][f"{comp}-{field}"] = [a, b]
            assert abs(a - b) < 1e-12

    # 4. Completeness: empty and partial are not reported as available.
    empty = tmp / "empty.csv"; write_ratings(empty, key_obj, "all_tie")
    rows = read_csv(empty)
    for row in rows:
        for k in row:
            if k != "task_id":
                row[k] = ""
    with open(empty, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    he = human_summary(empty, Path(args.key))
    partial = tmp / "partial.csv"
    one = read_csv(r)[0]
    with open(partial, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(one)); w.writeheader()
        w.writerow(one)
    hp = human_summary(partial, Path(args.key))
    results["completeness"] = {"empty": he["status"], "partial": [hp["status"], hp["available"]]}
    assert he["status"] == "empty" and he["available"] is False
    assert hp["status"] == "partial" and hp["available"] is False

    # 5. Preference fields are actually consumed, not silently ignored.
    r3 = tmp / "identity_swap.csv"
    rows = read_csv(r)
    for row in rows:
        row["identity_preference"] = "tie"
    with open(r3, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    h3 = human_summary(r3, Path(args.key))
    results["identity_used"] = h3["preference"]["B-D"]["identity"]["win_rate"]
    assert abs(results["identity_used"] - 0.5) < 1e-12

    results["status"] = "PASS"
    write_json(args.out, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
