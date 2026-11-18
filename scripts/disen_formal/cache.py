#!/usr/bin/env python
"""One-time caches: CrossInit init, VAE posteriors, CLIP image features, B320 c."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.init import celeb_names_cross_init, save_init_bundle
from crossinit.pipeline_io import (
    load_text_encoder, load_tokenizer, load_vae,
)
from scripts.disen_formal.common import (
    DATA_DIR, MODEL_PATH, NAMES_PATH, add_tokens, encode_with_token_rows,
    formal_identities, image_512, image_to_tensor_minus1, protocol,
    stream_seed, tensor_hash, tokens, write_json,
)

DEFAULT_CAPTION = "a photo of a <ci0> <ci1> person"


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for caching")
    return torch.device("cuda")


def init_bundle() -> None:
    device = _device()
    tokenizer = load_tokenizer(str(MODEL_PATH), None, local_files_only=True)
    te = load_text_encoder(str(MODEL_PATH), None, torch.float32,
                           local_files_only=True).to(device).eval()
    names = [x for x in NAMES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    embeds = celeb_names_cross_init(NAMES_PATH, tokenizer, te, n_tokens=2)
    out = DATA_DIR / "celeb_init_691.pt"
    meta = save_init_bundle(out, embeds, celeb_path=NAMES_PATH,
                            model_id=str(MODEL_PATH), revision=None, n_names=len(names))
    print(json.dumps({"init": str(out), "sha256": tensor_hash(embeds),
                      "shape": list(embeds.shape), "meta": meta}, indent=2))


def vae_cache() -> None:
    device = _device()
    vae = load_vae(str(MODEL_PATH), None, local_files_only=True).to(device, dtype=torch.float32).eval()
    scaling = float(vae.config.scaling_factor)
    pred = "epsilon"
    for ident in formal_identities() + [x for x in json.loads(
            (DATA_DIR / "identities.json").read_text(encoding="utf-8"))["identities"]
            if x["status"] == "reserve"]:
        image_id = str(ident["image_id"])
        im = image_512(ident["source_512"])
        x = image_to_tensor_minus1(im).unsqueeze(0).to(device)
        views = {"orig": x, "flip": torch.flip(x, dims=[3])}
        payload = {"scaling_factor": scaling, "prediction_type": pred, "views": {}}
        with torch.no_grad():
            for name, v in views.items():
                dist = vae.encode(v).latent_dist
                payload["views"][name] = {
                    "mean": dist.mean.detach().cpu().float(),
                    "logvar": dist.logvar.detach().cpu().float(),
                }
        out = DATA_DIR / "cache" / image_id / "vae_posterior.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)
        print(f"vae {image_id}: {payload['views']['orig']['mean'].shape}")


def clip_cache() -> None:
    import open_clip
    device = _device()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", precision="fp32", device=device)
    model.eval()
    from scripts.disen_formal.common import all_training_identities
    for ident in all_training_identities():
        image_id = str(ident["image_id"])
        im = image_512(ident["source_512"])
        x = image_to_tensor_minus1(im).unsqueeze(0).to(device)
        # Official DisenBooth preprocessing: bilinear 224, no CLIP mean/std.
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            f = model.encode_image(x, normalize=False).float().cpu().unsqueeze(1)
        out = DATA_DIR / "cache" / image_id / "clip_feature.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feature": f, "shape": list(f.shape),
                    "preprocess": "resize224_bilinear_no_mean_std",
                    "model": "ViT-H-14/laion2b_s32b_b79k", "normalize": False}, out)
        print(f"clip {image_id}: {tuple(f.shape)}")
    # Locate the actual checkpoint for the hash lock.
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                            "open_clip_pytorch_model.bin")
        write_json(ROOT / "locks" / "disen_formal_v1" / "openclip_checkpoint.json",
                   {"path": p, "size": Path(p).stat().st_size})
    except Exception as exc:  # remote cache layout may differ; do not hide the run
        print(f"warning: could not record OpenCLIP checkpoint path: {exc}")


def text_cache(only_formal: bool = True) -> None:
    device = _device()
    tokenizer = load_tokenizer(str(MODEL_PATH), None, local_files_only=True)
    te = load_text_encoder(str(MODEL_PATH), None, torch.float32,
                           local_files_only=True).to(device).eval()
    token_ids = add_tokens(tokenizer, te)
    from scripts.disen_formal.common import all_training_identities
    idents = formal_identities() if only_formal else all_training_identities()
    for ident in idents:
        image_id = str(ident["image_id"])
        run = ROOT / "runs" / "disen_formal_v1" / image_id / "B"
        payload = torch.load(run / "final" / "trainable.pt", map_location="cpu")
        rows = payload["state"]["tokens"].float()
        token_hash = tensor_hash(rows)
        with torch.no_grad():
            c = encode_with_token_rows(tokenizer, te, DEFAULT_CAPTION, token_ids,
                                       rows, device).detach().float().cpu()
        out = DATA_DIR / "cache" / image_id / "text_B320.pt"
        torch.save({
            "c": c, "caption": DEFAULT_CAPTION, "tokens": tokens(),
            "token_hash": token_hash, "shape": list(c.shape),
            "source": str(run / "final" / "trainable.pt"),
        }, out)
        print(f"text {image_id}: {tuple(c.shape)} token_hash={token_hash[:12]}")


def random_plans() -> None:
    proto = protocol()
    from scripts.disen_formal.common import all_training_identities, template_list
    templates = template_list()
    for ident in all_training_identities():
        image_id = str(ident["image_id"])
        b_seed = stream_seed(int(proto["baseline"]["train_seed"]), image_id, "B_plan")
        rng = random.Random(b_seed)
        b_steps = [{"template": rng.randrange(len(templates)),
                    "flip": int(rng.random() < 0.5)} for _ in range(1280)]
        plan = {
            "image_id": image_id,
            "B": {"seed": b_seed, "template_count": len(templates),
                  "micro_steps": 1280, "steps": b_steps},
            "adaptation": {
                "updates": 2000,
                "streams": {name: stream_seed(int(proto["adaptation"]["train_seed"]),
                                               image_id, name)
                            for name in ("vae", "noise", "timestep")},
            },
        }
        write_json(DATA_DIR / "random_plans" / f"{image_id}.json", plan)
    print(f"per-person B + adaptation plans for {len(all_training_identities())} people")


def pretrained_hashes() -> None:
    from scripts.disen_formal.assets import hardware_info
    from crossinit.init import sha256_file
    out = {"sd21": {}, "clip_eval": {}, "openclip": {}, "buffalo_l": {}}
    for p in sorted((MODEL_PATH).rglob("*")):
        if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"} and p.stat().st_size > 1000:
            out["sd21"][str(p.relative_to(MODEL_PATH))] = sha256_file(p)
    clip = ROOT / "data" / "eval" / "clip-vit-large-patch14"
    if clip.exists():
        for p in clip.rglob("*"):
            if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"}:
                out["clip_eval"][str(p.relative_to(clip))] = sha256_file(p)
    oc = ROOT / "locks" / "disen_formal_v1" / "openclip_checkpoint.json"
    if oc.exists():
        obj = json.loads(oc.read_text(encoding="utf-8"))
        out["openclip"] = {"path": obj["path"], "sha256": sha256_file(obj["path"])}
    bdir = Path.home() / ".insightface" / "models" / "buffalo_l"
    if bdir.exists():
        for p in sorted(bdir.rglob("*")):
            if p.is_file():
                out["buffalo_l"][p.name] = sha256_file(p)
    out["hardware"] = hardware_info()
    write_json(ROOT / "locks" / "disen_formal_v1" / "pretrained_hashes.json", out)
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in out.items()}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=lambda a: (init_bundle(), vae_cache(), clip_cache(),
                                                           random_plans(), pretrained_hashes()))
    t = sub.add_parser("text")
    t.add_argument("--all-formal", action="store_true")
    t.set_defaults(func=lambda a: text_cache(only_formal=True))
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
