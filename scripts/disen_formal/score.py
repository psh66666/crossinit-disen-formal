#!/usr/bin/env python
"""Per-image ArcFace, CLIP, and scene diagnostics for all 512 outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.face import cosine, detect_faces
from scripts.disen_formal.common import (
    formal_identities, identity_map, prompts, protocol, scene_texts,
)

CPU = ["CPUExecutionProvider"]


def target_face(im: Image.Image, det_thresh: float = 0.5):
    w, h = im.size
    faces = detect_faces(im, providers=CPU, det_thresh=det_thresh)
    qual = []
    for f in faces:
        area = f["area"] / float(w * h)
        if f["det_score"] >= det_thresh and area >= 0.02:
            x1, y1, x2, y2 = f["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            qual.append({
                "face": f, "area": area, "cx": cx, "cy": cy,
                "edge": int(not (0.2 * w <= cx <= 0.8 * w)),
                "center_dist": abs(cx - w / 2.0),
                "box": tuple(float(v) for v in f["bbox"]),
            })
    if not qual:
        return None, faces
    qual.sort(key=lambda q: (-q["area"], q["center_dist"], q["box"]))
    return qual[0], faces


def load_clip(device):
    paths = [ROOT / "data" / "eval" / "clip-vit-large-patch14",
             ROOT / "models" / "clip-vit-large-patch14"]
    for p in paths:
        if (p / "config.json").exists():
            model = CLIPModel.from_pretrained(str(p), local_files_only=True).to(device).eval()
            proc = CLIPProcessor.from_pretrained(str(p), local_files_only=True)
            return model, proc, str(p)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, proc, "openai/clip-vit-large-patch14"


@torch.no_grad()
def text_embeddings(model, proc, texts: list[str], device) -> torch.Tensor:
    inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    e = model.get_text_features(**inp)
    return e / e.norm(dim=-1, keepdim=True)


@torch.no_grad()
def image_embedding(model, proc, im: Image.Image, device) -> torch.Tensor:
    inp = proc(images=im, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    e = model.get_image_features(**inp)
    return e / e.norm(dim=-1, keepdim=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--manifest", default="runs/disen_formal_v1/generated/manifest.jsonl")
    ap.add_argument("--out", default="results/disen_formal_v1/per_image.csv")
    args = ap.parse_args()
    proto = protocol()
    ev = proto["evaluation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    clip, proc, clip_source = load_clip(device)
    prompt_rows = prompts()
    scene = scene_texts()
    scene_ids = [p for p in ("p05", "p06", "p07", "p08") if p in scene]
    scene_text = [scene[p] for p in scene_ids]
    text_all = []
    for p in prompt_rows:
        text_all.extend([p["clip_text"], p["attribute_text"]])
    text_all.extend(scene_text)
    txt_emb = text_embeddings(clip, proc, text_all, device)
    index = {t: i for i, t in enumerate(text_all)}
    scene_emb = txt_emb[[index[t] for t in scene_text]]

    idents = identity_map()
    source_emb = {}
    source_meta = {}
    for image_id, ident in idents.items():
        im = Image.open(ident["source_512"]).convert("RGB")
        q, faces = target_face(im, float(ev["det_thresh"]))
        if q is None:
            raise SystemExit(f"source {image_id} has no target face at scoring threshold")
        source_emb[image_id] = q["face"]["emb"]
        source_meta[image_id] = {"det": q["face"]["det_score"], "area": q["area"]}

    rows = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok":
            continue
        image_id = str(rec["image_id"])
        path = ROOT / rec["image"]
        im = Image.open(path).convert("RGB")
        q, faces = target_face(im, float(ev["det_thresh"]))
        if q is None:
            arc, det, area, edge, nfaces = 0.0, 0.0, 0.0, 0, len(faces)
        else:
            arc = cosine(q["face"]["emb"], source_emb[image_id])
            det, area, edge, nfaces = q["face"]["det_score"], q["area"], q["edge"], len(faces)
        img_emb = image_embedding(clip, proc, im, device)[0]
        p = next(x for x in prompt_rows if str(x["id"]) == str(rec["prompt_id"]))
        full = float((img_emb * txt_emb[index[p["clip_text"]]]).sum())
        attr = float((img_emb * txt_emb[index[p["attribute_text"]]]).sum())
        scene_cos = (img_emb.unsqueeze(0) * scene_emb).sum(dim=-1)
        margin, acc = "", ""
        if str(p["id"]) in scene_ids:
            j = scene_ids.index(str(p["id"]))
            others = torch.cat([scene_cos[:j], scene_cos[j + 1:]])
            margin = float(scene_cos[j] - others.mean())
            acc = int(int(scene_cos.argmax()) == j)
        rows.append({
            "image_id": image_id, "group": rec["group"], "prompt_id": str(p["id"]),
            "seed": int(rec["seed"]), "image": rec["image"],
            "target": int(q is not None), "arcface": f"{arc:.8f}",
            "target_det": f"{det:.6f}", "target_area": f"{area:.8f}",
            "edge": int(edge), "n_faces": int(nfaces),
            "full_clip": f"{full:.8f}", "attr_clip": f"{attr:.8f}",
            "scene_margin": "" if margin == "" else f"{margin:.8f}",
            "scene_acc": "" if acc == "" else str(acc),
            "clip_text": p["clip_text"], "attribute_text": p["attribute_text"],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"scored {len(rows)} images -> {out}; clip={clip_source}")
    # A second CSV keeps the source-face audit binding.
    with open(out.parent / "source_faces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "source_target_det", "source_target_area"])
        w.writeheader()
        for image_id, m in source_meta.items():
            w.writerow({"image_id": image_id, "source_target_det": f"{m['det']:.6f}",
                        "source_target_area": f"{m['area']:.8f}"})


if __name__ == "__main__":
    main()
