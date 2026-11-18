#!/usr/bin/env python
"""Paired inference grid: 8 people x 8 prompts x 2 seeds x B/L/D/A.

The initial latent is generated explicitly from a fresh seed for every key and
passed to the pipeline, so every group receives byte-identical noise.  D/A
never load the training image adapter; LoRA scale is always 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_lora import attach_lora, load_lora, lora_hash
from crossinit.disen_state import load_final, sha256_file, write_json
from crossinit.pipeline_io import load_pipeline
from scripts.disen_formal.common import (
    MODEL_PATH, formal_identities, prompts, protocol, tensor_hash, tokens,
)

PROJECT = ROOT / "runs" / "disen_formal_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--freeze", default="locks/disen_formal_v1/freeze.json")
    ap.add_argument("--prompts", default="experiments/disen_formal_v1/prompts.jsonl")
    ap.add_argument("--all-formal", action="store_true")
    ap.add_argument("--groups", nargs="+", default=["B", "L", "D", "A"])
    ap.add_argument("--image-ids", nargs="+", default=None,
                    help="optional subset for smoke tests / targeted regeneration")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    proto = protocol()
    gen_cfg = proto["generation"]
    idents = formal_identities()
    if args.image_ids:
        wanted = {str(x) for x in args.image_ids}
        idents = [x for x in idents if str(x["image_id"]) in wanted]
        if not idents:
            raise SystemExit(f"no formal identity matched {sorted(wanted)}")
    prompt_rows = prompts()
    seeds = [int(x) for x in gen_cfg["seeds"]]
    groups = [g.upper() for g in args.groups]
    out_root = PROJECT / "generated"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    done = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                done[(r["image_id"], r["group"], r["prompt_id"], int(r["seed"]))] = r

    dtype = torch.bfloat16
    device = torch.device("cuda")
    pipe = load_pipeline(str(MODEL_PATH), None, dtype=dtype, device=device,
                         local_files_only=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2,
        use_karras_sigmas=False, timestep_spacing="linspace", solver_type="midpoint")
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    lora = attach_lora(pipe.unet, rank=int(proto["adaptation"]["lora_rank"]),
                       alpha=int(proto["adaptation"]["lora_alpha"]),
                       dropout=0.0, init_seed=int(proto["adaptation"]["init_seed"]))
    import hashlib
    scheduler_cfg = dict(pipe.scheduler.config)
    scheduler_hash = hashlib.sha256(
        json.dumps(scheduler_cfg, sort_keys=True, default=str).encode()).hexdigest()
    token_names = tokens()
    added = False
    write_json(out_root / "generation_config.json", {
        "groups": groups, "seeds": seeds, "prompts": [p["id"] for p in prompt_rows],
        "num_inference_steps": int(gen_cfg["num_inference_steps"]),
        "guidance_scale": float(gen_cfg["guidance_scale"]),
        "negative_prompt": gen_cfg["negative_prompt"],
        "scheduler": type(pipe.scheduler).__name__, "scheduler_hash": scheduler_hash,
        "initial_latents_explicit": True, "lora_scale": 1.0,
    })

    count = 0
    for ident in idents:
        image_id = str(ident["image_id"])
        parent_state, parent_meta = load_final(PROJECT / image_id / "B",
                                               {"group": "B", "image_id": image_id})
        parent_rows = parent_state["tokens"].float()
        parent_hash = tensor_hash(parent_rows)
        if not added:
            pipe.tokenizer.add_tokens(token_names)
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
            added = True
        token_ids = [int(pipe.tokenizer.convert_tokens_to_ids(t)) for t in token_names]
        with torch.no_grad():
            for tid, row in zip(token_ids, parent_rows):
                pipe.text_encoder.get_input_embeddings().weight.data[tid] = row.to(
                    device=pipe.text_encoder.device, dtype=pipe.text_encoder.dtype)
        for group in groups:
            run_dir = PROJECT / image_id / group
            final = run_dir / "final"
            meta = json.loads((final / "meta.json").read_text(encoding="utf-8"))
            if group != "B":
                if int(meta.get("parent_step", -1)) != 320:
                    raise SystemExit(f"{final}: parent_step != 320")
                if meta.get("parent_token_hash") != parent_hash:
                    raise SystemExit(f"{final}: parent token hash mismatch")
                state = torch.load(final / "trainable.pt", map_location="cpu")["state"]
                lora_state = {k[len("lora."):]: v for k, v in state.items() if k.startswith("lora.")}
                lora.load_state_dict(lora_state, strict=True)
                lora.set_scale(1.0)
            else:
                lora.set_scale(0.0)
            lora_state_hash = lora_hash(lora) if group != "B" else "none"
            for prompt_row in prompt_rows:
                filled = str(prompt_row["prompt"]).replace("{tokens}", " ".join(token_names))
                for seed in seeds:
                    key = (image_id, group, str(prompt_row["id"]), int(seed))
                    out_path = out_root / image_id / group / f"{prompt_row['id']}_seed{seed}.png"
                    if key in done and out_path.exists() and not args.overwrite:
                        count += 1
                        continue
                    if args.limit is not None and count >= args.limit:
                        print(f"limit {args.limit} reached")
                        return
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    gen = torch.Generator(device=device).manual_seed(int(seed))
                    latents = torch.randn((1, 4, 64, 64), generator=gen, device=device,
                                          dtype=torch.float32).to(dtype=dtype)
                    latent_hash = tensor_hash(latents)
                    with torch.no_grad():
                        image = pipe(
                            filled, num_inference_steps=int(gen_cfg["num_inference_steps"]),
                            guidance_scale=float(gen_cfg["guidance_scale"]),
                            negative_prompt=gen_cfg["negative_prompt"],
                            num_images_per_prompt=1, latents=latents,
                        ).images[0]
                    image.save(out_path)
                    rec = {
                        "status": "ok", "image": str(out_path.relative_to(ROOT)),
                        "image_id": image_id, "identity": ident.get("person_id"),
                        "group": group, "prompt_id": str(prompt_row["id"]),
                        "prompt": str(prompt_row["prompt"]), "filled_prompt": filled,
                        "clip_text": prompt_row.get("clip_text"),
                        "attribute_text": prompt_row.get("attribute_text"),
                        "seed": int(seed), "initial_latent_hash": latent_hash,
                        "image_sha256": sha256_file(out_path),
                        "parent_token_hash": parent_hash, "lora_hash": lora_state_hash,
                        "model": str(MODEL_PATH),
                        "model_index_sha256": sha256_file(MODEL_PATH / "model_index.json"),
                        "scheduler": type(pipe.scheduler).__name__, "scheduler_hash": scheduler_hash,
                        "num_inference_steps": int(gen_cfg["num_inference_steps"]),
                        "guidance_scale": float(gen_cfg["guidance_scale"]),
                        "negative_prompt": gen_cfg["negative_prompt"],
                        "lora_scale": 1.0 if group != "B" else 0.0,
                    }
                    with open(manifest_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[key] = rec
                    count += 1
                    if count % 16 == 0:
                        print(f"[{count}] {image_id}/{group}/{prompt_row['id']}/{seed}")
    print(f"generated/verified {count} images; unique ok keys={len(done)}")


if __name__ == "__main__":
    main()
