#!/usr/bin/env python
"""Deterministic source selection for the formal matrix.

Manual visual review cannot be performed by the executor, so this script uses
the dataset's pose and attribute annotations plus InsightFace as an objective
proxy.  The deviation is recorded in identities.json and in the report; the
generated source sheet is still available for a human to inspect later.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.face import cosine, detect_faces
from scripts.disen_formal.common import (
    DATA_DIR, RAW_DATA, image_512, protocol, sha256_file, tokens, write_json,
)

CPU = ["CPUExecutionProvider"]


def load_assets() -> dict:
    return json.loads((ROOT / "locks" / "disen_formal_v1" / "assets.json")
                      .read_text(encoding="utf-8"))


def parse_mapping() -> dict[int, str]:
    """HQ idx -> official CelebA filename from the mapping's ``orig_file`` column.

    The filename is one-based while ``orig_idx`` is zero-based; use the filename,
    never ``orig_idx``, to join the identity annotation.
    """
    out = {}
    p = RAW_DATA / "CelebA-HQ-to-CelebA-mapping.txt"
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = line.split()
        out[int(parts[0])] = parts[2]
    if len(out) != 30000:
        raise ValueError(f"mapping has {len(out)} rows, expected 30000")
    return out


def parse_identity() -> dict[str, int]:
    """Official CelebA filename -> person id."""
    out = {}
    p = ROOT / "data" / "annotations" / "identity_CelebA.txt"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, pid = line.split()
        out[name] = int(pid)
    return out


def parse_pose() -> dict[int, tuple[float, float, float]]:
    out = {}
    lines = (RAW_DATA / "CelebAMask-HQ-pose-anno.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if not parts[0].endswith(".jpg"):
            continue
        out[int(Path(parts[0]).stem)] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return out


def parse_attrs() -> tuple[list[str], dict[int, dict[str, int]]]:
    lines = (RAW_DATA / "CelebAMask-HQ-attribute-anno.txt").read_text(encoding="utf-8").splitlines()
    names = lines[1].split()
    out = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        vals = [int(x) for x in parts[1:]]
        out[int(Path(parts[0]).stem)] = dict(zip(names, vals))
    return names, out


def history_ids() -> tuple[list[int], dict[str, list[int]]]:
    proto = protocol()
    base = {int(x) for x in proto["data"]["excluded_image_ids"]}
    found: dict[str, list[int]] = {"protocol_excluded": sorted(base)}
    for p in (ROOT / "data").glob("identities*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ids = []
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    ids.extend(int(x) for x in v if str(x).isdigit())
        elif isinstance(obj, list):
            ids.extend(int(x.get("image_id", x)) for x in obj if str(x).isdigit())
        if ids:
            found[str(p.relative_to(ROOT))] = sorted(set(ids))
            base.update(ids)
    return sorted(base), found


def source_face(path: Path, det_thresh: float = 0.5):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    faces = detect_faces(im, providers=CPU, det_thresh=det_thresh)
    qual = []
    for f in faces:
        area = f["area"] / float(w * h)
        if f["det_score"] >= det_thresh and area >= 0.02:
            x1, y1, x2, y2 = f["bbox"]
            qual.append((f, area, abs((x1 + x2) / 2 - w / 2)))
    if not qual:
        return None
    qual.sort(key=lambda x: (-x[1], x[2]))
    return qual[0]


def draw_source_sheet(rows: list[dict], out: Path) -> None:
    cell, pad, header = 260, 8, 52
    cols = 5
    height = pad + ((len(rows) + cols - 1) // cols) * (cell + header + pad)
    sheet = Image.new("RGB", (pad + cols * (cell + pad), height), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6), "Formal source selection (auto proxy review; inspect before publication)", fill="black")
    for i, r in enumerate(rows):
        im = Image.open(r["source_512"]).convert("RGB").resize((cell, cell), Image.BICUBIC)
        x = pad + (i % cols) * (cell + pad)
        y = pad + header + (i // cols) * (cell + header + pad)
        sheet.paste(im, (x, y))
        label = f"{r['status']} #{r['selection_order']} id={r['image_id']} p={r['person_id']}"
        d.text((x, y - 14), label, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def candidates(args) -> None:
    proto = protocol()
    mapping = parse_mapping()
    identity = parse_identity()
    pose = parse_pose()
    _, attrs = parse_attrs()
    hist, hist_sources = history_ids()
    hist_persons = {identity[mapping[i]] for i in hist if i in mapping and mapping[i] in identity}
    print(f"history images={len(hist)} persons={len(hist_persons)} sources={hist_sources}")

    all_ids = sorted(mapping)
    rng = random.Random(int(proto["data"]["selection_seed"]))
    order = list(all_ids)
    rng.shuffle(order)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DATA_DIR / "candidate_order.json", {
        "seed": int(proto["data"]["selection_seed"]),
        "count": len(order),
        "image_ids": order,
    })

    # Historical face embeddings for the conservative near-duplicate screen.
    hist_embs = {}
    for i in hist:
        p = RAW_DATA / "CelebA-HQ-img" / f"{i}.jpg"
        if not p.exists():
            continue
        f = source_face(p, det_thresh=float(proto["data"]["source_detection_score_min"]))
        if f:
            hist_embs[i] = f[0]["emb"]

    accepted, selected_embs = [], {}
    rows = []
    used_persons = set()
    max_accept = int(proto["data"]["formal_persons"] + proto["data"]["reserve_persons"])
    for image_id in order:
        if len(accepted) >= max_accept:
            break
        src = RAW_DATA / "CelebA-HQ-img" / f"{image_id}.jpg"
        reason = ""
        if image_id in hist:
            reason = "historical_image"
        if not reason:
            pid = identity.get(mapping[image_id])
            if pid is None:
                reason = "missing_person_id"
            elif pid in hist_persons:
                reason = "historical_person"
            elif pid in used_persons:
                reason = "duplicate_person"
        yaw, pitch, roll = pose.get(image_id, (999, 999, 999))
        if not reason and (abs(yaw) > 20 or abs(pitch) > 15 or abs(roll) > 15):
            reason = f"pose_{yaw:.1f}_{pitch:.1f}_{roll:.1f}"
        a = attrs.get(image_id, {})
        if not reason and (a.get("Eyeglasses", -1) > 0 or a.get("Wearing_Hat", -1) > 0):
            reason = "glasses_or_hat"
        if not reason and a.get("Blurry", -1) > 0:
            reason = "blurry"
        face = None
        face_area = 0.0
        im = None
        if not reason:
            im = image_512(src) if src.exists() else None
            if im is None:
                reason = "missing_file"
        if not reason:
            w, h = im.size
            # Resize to 512 first, then detect; thresholds are on the 512 image.
            faces = detect_faces(im, providers=CPU,
                                 det_thresh=float(proto["evaluation"]["det_thresh"]))
            qual = []
            for f in faces:
                fa = f["area"] / float(w * h)
                if f["det_score"] >= float(proto["data"]["source_detection_score_min"]) and fa >= 0.02:
                    x1, y1, x2, y2 = f["bbox"]
                    qual.append((f, fa, abs((x1 + x2) / 2 - w / 2)))
            if not qual:
                reason = "no_face_ge_0.8"
            elif len(qual) > 1:
                reason = f"multiple_qualifying_faces_{len(qual)}"
            else:
                f, fa, _ = qual[0]
                if not (float(proto["data"]["source_face_area_fraction_min"]) <= fa <=
                        float(proto["data"]["source_face_area_fraction_max"])):
                    reason = f"face_area_{fa:.3f}"
                else:
                    face = f
                    face_area = fa
            if face is not None:
                for hid, emb in hist_embs.items():
                    if cosine(face["emb"], emb) >= float(proto["data"]["near_duplicate_arcface_cosine_threshold"]):
                        reason = f"near_duplicate_history_{hid}"
                        break
            if face is not None:
                for sid, emb in selected_embs.items():
                    if cosine(face["emb"], emb) >= float(proto["data"]["near_duplicate_arcface_cosine_threshold"]):
                        reason = f"near_duplicate_selected_{sid}"
                        break

        row = {"image_id": image_id, "person_id": identity.get(mapping[image_id], ""),
               "pose": f"{yaw:.2f},{pitch:.2f},{roll:.2f}",
               "accepted": 0, "reason": reason, "det_score": "", "face_area": ""}
        if reason:
            rows.append(row)
            continue
        pid = identity[mapping[image_id]]
        used_persons.add(pid)
        selected_embs[image_id] = face["emb"]
        out_path = DATA_DIR / "source" / f"{image_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path)
        row.update({"accepted": 1, "reason": "accepted_auto_proxy",
                    "det_score": f"{face['det_score']:.4f}", "face_area": f"{face_area:.4f}"})
        rows.append(row)
        accepted.append({
            "selection_order": len(accepted) + 1,
            "image_id": int(image_id),
            "person_id": int(pid),
            "original": str(src),
            "original_sha256": sha256_file(src),
            "source_512": str(out_path),
            "source_512_sha256": sha256_file(out_path),
            "source_face_embedding_sha256": hashlib.sha256(
                face["emb"].astype(np.float32).tobytes()).hexdigest(),
            "det_score": float(face["det_score"]),
            "face_area_fraction": float(face_area),
            "pose_yaw_pitch_roll": [float(yaw), float(pitch), float(roll)],
            "status": "formal" if len(accepted) < int(proto["data"]["formal_persons"]) else "reserve",
            "review": {
                "method": "automated_proxy_only",
                "criteria": "single face det>=0.8, area .06-.50, |yaw|<=20, |pitch|<=15, "
                            "no glasses/hat/blur attribute, no historical person or near-duplicate",
                "human_visual_review": "pending",
            },
        })

    if len(accepted) < max_accept:
        raise SystemExit(f"only {len(accepted)} accepted sources, need {max_accept}")

    with open(DATA_DIR / "candidate_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(DATA_DIR / "candidate_pool.json", {
        "history_sources": hist_sources, "history_persons": sorted(hist_persons),
        "screened": len(rows), "accepted": len(accepted),
    })
    review = {
        "protocol_id": proto["protocol_id"],
        "decisions": accepted,
        "notes": "Auto-proxy selection used because the executor cannot visually inspect images. "
                 "Every accepted source still needs a human visual check before publication.",
    }
    write_json(DATA_DIR / "source_review.json", review)
    draw_source_sheet(accepted, DATA_DIR / "source_review.png")
    print(f"screened={len(rows)} accepted={len(accepted)} -> {DATA_DIR}")


def finalize(args) -> None:
    proto = protocol()
    review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    decisions = review.get("decisions", [])
    accepted = [d for d in decisions if d.get("accepted", True)]
    formal = [d for d in accepted if d["status"] == "formal"]
    reserve = [d for d in accepted if d["status"] == "reserve"]
    if len(formal) != int(proto["data"]["formal_persons"]):
        raise SystemExit(f"need {proto['data']['formal_persons']} formal, got {len(formal)}")
    if len(reserve) != int(proto["data"]["reserve_persons"]):
        raise SystemExit(f"need {proto['data']['reserve_persons']} reserve, got {len(reserve)}")
    persons = [d["person_id"] for d in accepted]
    if len(set(persons)) != len(persons):
        raise SystemExit("person_id collision in final selection")
    for d in accepted:
        if not Path(d["source_512"]).exists():
            raise SystemExit(f"missing source {d['source_512']}")
        if sha256_file(d["source_512"]) != d["source_512_sha256"]:
            raise SystemExit(f"source hash changed: {d['source_512']}")
    obj = {
        "protocol_id": proto["protocol_id"],
        "selection_seed": proto["data"]["selection_seed"],
        "tokens": tokens(),
        "review_method": "automated_proxy_only",
        "human_visual_review_completed": False,
        "identities": accepted,
    }
    write_json(args.out, obj)
    print(json.dumps({"formal": [d["image_id"] for d in formal],
                      "reserve": [d["image_id"] for d in reserve],
                      "persons": persons}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    c.add_argument("--assets", default="locks/disen_formal_v1/assets.json")
    c.add_argument("--out", default="data/disen_formal")
    c.set_defaults(func=candidates)
    f = sub.add_parser("finalize")
    f.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    f.add_argument("--review", default="data/disen_formal/source_review.json")
    f.add_argument("--out", default="data/disen_formal/identities.json")
    f.set_defaults(func=finalize)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
