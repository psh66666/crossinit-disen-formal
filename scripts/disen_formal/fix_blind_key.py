#!/usr/bin/env python
"""Repair an already-generated blind key from the HTML's actual image paths.

This is deliberately a repair tool, not a regenerator: task ids, page order,
and images stay untouched so a human can keep her existing answers.  It is
also the independent check required before scoring: every key entry must agree
with the method encoded in the A/B image path.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def image_info(src: str) -> dict:
    m = re.search(r"generated/([^/]+)/([^/]+)/(p\d+)_seed(\d+)\.png", src)
    if not m:
        raise ValueError(f"cannot parse generated image path: {src}")
    image_id, group, prompt_id, seed = m.groups()
    return {"image_id": image_id, "group": group, "prompt_id": prompt_id,
            "seed": int(seed), "key": f"{image_id}|{group}|{prompt_id}|{seed}",
            "path": src}


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="results/disen_formal_v1/blind/index.html")
    ap.add_argument("--key", default="results/disen_formal_v1/blind/key.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    html_path, key_path = Path(args.html), Path(args.key)
    html = html_path.read_text(encoding="utf-8")
    key_obj = json.loads(key_path.read_text(encoding="utf-8"))
    key = key_obj["tasks"]

    arts = re.findall(r'<article class="task" data-id="([^"]+)">(.*?)</article>', html, re.S)
    if len(arts) != 176:
        raise SystemExit(f"expected 176 articles, found {len(arts)}")
    order, mismatches = [], []
    repaired = {}
    for tid, body in arts:
        order.append(tid)
        imgs = re.findall(r'<img src="([^"]+)">', body)
        if len(imgs) != 3:
            raise SystemExit(f"{tid}: expected source+A+B images, found {len(imgs)}")
        a, b = image_info(imgs[1]), image_info(imgs[2])
        old = key[tid]
        if old["image_id"] != a["image_id"] or old["image_id"] != b["image_id"]:
            raise SystemExit(f"{tid}: image_id mismatch between key and HTML")
        if old["prompt_id"] != a["prompt_id"] or old["prompt_id"] != b["prompt_id"]:
            raise SystemExit(f"{tid}: prompt mismatch between key and HTML")
        groups = {"A": a["group"], "B": b["group"]}
        if groups != old["groups"]:
            mismatches.append({"task_id": tid, "old": old["groups"], "new": groups,
                               "repeat_of": old.get("repeat_of")})
        repaired[tid] = {
            "groups": groups, "repeat_of": old.get("repeat_of"),
            "image_id": a["image_id"], "prompt_id": a["prompt_id"],
            "seed": a["seed"], "A": a["path"], "B": b["path"],
            "A_key": a["key"], "B_key": b["key"],
        }
    if set(repaired) != set(key):
        raise SystemExit("task id set changed; refusing to repair")
    out = Path(args.out) if args.out else key_path
    if out == key_path:
        shutil.copy2(key_path, key_path.with_name("key_before_repair.json"))
    write_json(out, {"protocol_id": key_obj.get("protocol_id"),
                     "order": order, "tasks": repaired,
                     "repair_note": "groups derived from the A/B image paths in index.html; "
                                    "task ids/order/images unchanged"})
    print(json.dumps({"tasks": len(repaired), "mismatches": len(mismatches),
                      "mismatched_repeats": sorted(m["task_id"] for m in mismatches
                                                   if m["repeat_of"]),
                      "mismatches_detail": mismatches}, indent=2))
    if mismatches:
        print("repaired key written; original preserved as key_before_repair.json")


if __name__ == "__main__":
    main()
