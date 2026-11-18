#!/usr/bin/env python
"""Single B/L/D/A training job for the formal matrix.

B: CrossInit two-token baseline, 320 updates x 4 micro-batches.
L: parent B320 + rank-4 U-Net attention LoRA, plain denoising.
D: L + image adapter, precise + weak denoising + cosine.
A: D without the weak denoising term.

Every job is a fresh process; no LoRA state can leak between groups.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_adapter import ImageAdapter, contrastive_cosine
from crossinit.disen_lora import (
    attach_lora, lora_hash, load_lora, save_lora, state_hash,
)
from crossinit.disen_state import (
    canonical_hash, is_complete, load_checkpoint, load_final, mark_failed,
    mark_running, rng_state, save_checkpoint, sha256_file, write_final, write_json,
)
from crossinit.logging_utils import JsonlLogger, gpu_stats, setup_logger
from crossinit.pipeline_io import (
    load_scheduler, load_text_encoder, load_tokenizer, load_unet, load_vae,
)
from scripts.disen_formal.common import (
    DATA_DIR, MODEL_PATH, NAMES_PATH, add_tokens, encode_with_token_rows,
    formal_identities, identity_map, image_512, image_to_tensor_minus1,
    protocol, stream_seed, tensor_hash, tensor_to_pil, tokens,
)

CAPTION = "a photo of a <ci0> <ci1> person"


def set_seeds(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_dirs(args) -> tuple[Path, Path]:
    # Engineering writes weights/logs to its own tree but reads the exact same
    # frozen VAE/CLIP caches as the formal jobs.
    if args.engineering:
        return ROOT / "engineering" / "disen_formal_v1", DATA_DIR / "cache"
    return ROOT / "runs" / "disen_formal_v1", DATA_DIR / "cache"


def load_freeze(args) -> dict:
    if args.engineering:
        return {}
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    expected = freeze.get("freeze_hash")
    body = dict(freeze)
    body.pop("freeze_hash", None)
    if canonical_hash(body) != expected:
        raise SystemExit("freeze lock hash does not match its contents")
    return freeze


def load_identity(image_id: str) -> dict:
    idents = identity_map()
    if image_id not in idents:
        raise SystemExit(f"image_id {image_id} is not in identities.json")
    return idents[image_id]


def _view_batch(image_id: str, flip: int, device: torch.device, gen: torch.Generator,
                cache_root: Path):
    payload = torch.load(cache_root / image_id / "vae_posterior.pt", map_location="cpu")
    v = payload["views"]["flip" if flip else "orig"]
    mean, logvar = v["mean"].float(), v["logvar"].float()
    eps = torch.randn(mean.shape, generator=gen, dtype=torch.float32).to(device)
    z = mean.to(device) + eps * torch.exp(0.5 * logvar.to(device))
    return z * float(payload["scaling_factor"])


def _text_condition(tokenizer, te, token_ids, rows, device) -> torch.Tensor:
    return encode_with_token_rows(tokenizer, te, CAPTION, token_ids, rows, device)


def _target(scheduler, latents, noise, timesteps):
    kind = scheduler.config.prediction_type
    if kind == "epsilon":
        return noise
    if kind == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"unsupported prediction_type {kind}")


def train_b(args, run_dir: Path, cache_root: Path, logger, jsonl) -> None:
    proto = protocol()["baseline"]
    image_id = str(args.image_id)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(str(MODEL_PATH), None, local_files_only=True)
    scheduler = load_scheduler(str(MODEL_PATH), None, local_files_only=True)
    te = load_text_encoder(str(MODEL_PATH), None, torch.float32,
                           local_files_only=True).to(device).eval()
    token_ids = add_tokens(tokenizer, te)
    init_payload = torch.load(DATA_DIR / "celeb_init_691.pt", map_location="cpu")
    init_rows = init_payload["embeds"].float().to(device)
    rows = torch.nn.Parameter(init_rows.clone(), requires_grad=True)
    vae = load_vae(str(MODEL_PATH), None, local_files_only=True).to(device, dtype=torch.float32).eval()
    unet = load_unet(str(MODEL_PATH), None, local_files_only=True).to(device, dtype=torch.bfloat16).eval()
    te.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    optimizer = torch.optim.AdamW([rows], lr=float(proto["learning_rate"]),
                                  betas=tuple(proto["betas"]), eps=float(proto["eps"]),
                                  weight_decay=float(proto["weight_decay"]))
    plan_path = DATA_DIR / "random_plans" / f"{image_id}.json"
    if not plan_path.exists():
        raise SystemExit(f"missing {plan_path}; run cache prepare")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    b_plan = plan["B"]
    steps_plan = b_plan["steps"]
    plan_sha = sha256_file(plan_path)
    gens = {
        "vae": torch.Generator(device="cpu").manual_seed(stream_seed(int(proto["train_seed"]), image_id, "B_vae")),
        "noise": torch.Generator(device="cpu").manual_seed(stream_seed(int(proto["train_seed"]), image_id, "B_noise")),
        "timestep": torch.Generator(device="cpu").manual_seed(stream_seed(int(proto["train_seed"]), image_id, "B_timestep")),
    }
    start = 1
    if args.resume:
        ck = ROOT / "runs" / "disen_formal_v1" / image_id / "B"
        from crossinit.disen_state import latest_checkpoint
        found = latest_checkpoint(run_dir)
        if found:
            step, path = found
            state = torch.load(path / "trainable.pt", map_location="cpu")["state"]
            with torch.no_grad():
                rows.copy_(state["tokens"].to(device=device, dtype=torch.float32))
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
            from crossinit.disen_state import restore_rng
            restore_rng(torch.load(path / "rng.pt", map_location="cpu"), gens)
            start = step + 1
            logger.info("resumed B at step %d from %s", step, path)

    max_steps = int(args.max_updates or proto["optimizer_updates"])
    accum = int(proto["gradient_accumulation"])
    init_ref = init_rows.detach()
    base_emb = te.get_input_embeddings().weight.detach()
    meta_base = {"group": "B", "image_id": image_id, "model": str(MODEL_PATH),
                 "model_index_sha256": sha256_file(MODEL_PATH / "model_index.json"),
                 "init_sha256": sha256_file(DATA_DIR / "celeb_init_691.pt"),
                 "plan_sha256": plan_sha, "run_config_hash": canonical_hash(
                     {k: proto[k] for k in ("optimizer_updates", "learning_rate", "reg_weight")})}
    t_last = time.time()
    for update in range(start, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        mses, regs = [], []
        for micro in range(accum):
            idx = (update - 1) * accum + micro
            item = steps_plan[idx % len(steps_plan)]
            from scripts.disen_formal.common import template_list
            template = template_list()[item["template"]]
            text = template.format("face of " + " ".join(tokens()) + " person")
            enc = tokenizer(text, padding="max_length", truncation=True,
                            max_length=int(tokenizer.model_max_length), return_tensors="pt")
            ids = enc.input_ids.to(device)
            with torch.no_grad():
                emb = te.get_input_embeddings()(ids)
            emb = emb.clone()
            for j, tid in enumerate(token_ids):
                pos = ids[0] == tid
                emb[0, pos] = rows[j].to(dtype=emb.dtype)
            cond = te(inputs_embeds=emb)[0].to(dtype=torch.bfloat16)

            latents = _view_batch(image_id, int(item["flip"]), device, gens["vae"], cache_root)
            noise = torch.randn(latents.shape, generator=gens["noise"], dtype=torch.float32).to(device)
            ts = torch.randint(0, int(scheduler.config.num_train_timesteps), (1,),
                               generator=gens["timestep"]).long().to(device)
            noisy = scheduler.add_noise(latents, noise, ts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(noisy, ts, cond).sample
            target = _target(scheduler, latents, noise, ts)
            mse = F.mse_loss(pred.float(), target.float())
            reg, reg_dist = torch.tensor(0.0, device=device), 0.0
            dist = F.pairwise_distance(rows, init_ref, p=2).mean()
            reg = dist * float(proto["reg_weight"])
            loss = mse + reg
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite B loss at update {update} micro {micro}")
            loss.backward()
            mses.append(float(mse.detach()))
            regs.append(float(reg.detach()))
        gnorm = float(rows.grad.norm()) if rows.grad is not None else 0.0
        optimizer.step()
        now = time.time()
        rec = {"step": update, "group": "B", "image_id": image_id,
               "mse": sum(mses) / len(mses), "reg": sum(regs) / len(regs),
               "loss": sum(mses) / len(mses) + sum(regs) / len(regs),
               "token_grad_norm": gnorm, "lr": float(optimizer.param_groups[0]["lr"]),
               "sec_per_update": round(now - t_last, 4), "micro_step": update * accum,
               **gpu_stats()}
        t_last = now
        jsonl.write(rec)
        logger.info("B step %d/%d mse=%.4f reg=%.6f gn=%.2e %.2fs",
                    update, max_steps, rec["mse"], rec["reg"], gnorm, rec["sec_per_update"])
        if update in set(int(x) for x in proto["checkpoints"]) or update == max_steps:
            state = {"tokens": rows.detach().cpu().clone()}
            save_checkpoint(run_dir, update, state, optimizer, rng_state(gens), meta_base)
        if update >= max_steps:
            break
    state = {"tokens": rows.detach().cpu().clone(), "token_hash": tensor_hash(rows.detach())}
    write_final(run_dir, max_steps, state, meta_base)


def _parent_rows(args, image_id: str, device) -> tuple[torch.Tensor, dict]:
    if args.engineering:
        payload = torch.load(DATA_DIR / "celeb_init_691.pt", map_location="cpu")
        return payload["embeds"].float().to(device), {"parent_step": 0, "engineering_b0": True}
    run = ROOT / "runs" / "disen_formal_v1" / image_id / "B"
    state, meta = load_final(run, expected_meta={"group": "B", "image_id": image_id})
    if int(meta.get("step", -1)) != 320:
        raise SystemExit(f"parent B final is step {meta.get('step')}, expected 320")
    return state["tokens"].float().to(device), meta


def _condition(args, image_id: str, rows: torch.Tensor, device, tokenizer, te, token_ids, cache_root) -> torch.Tensor:
    if args.engineering:
        with torch.no_grad():
            return _text_condition(tokenizer, te, token_ids, rows, device).detach().float().cpu()
    p = cache_root / image_id / "text_B320.pt"
    if not p.exists():
        raise SystemExit(f"missing {p}; run cache text after all B jobs")
    payload = torch.load(p, map_location="cpu")
    if payload["token_hash"] != tensor_hash(rows.cpu()):
        raise SystemExit(f"{p}: token hash does not match parent B final")
    return payload["c"]


def train_adapt(args, run_dir: Path, cache_root: Path, logger, jsonl) -> None:
    proto = protocol()["adaptation"]
    image_id = str(args.image_id)
    group = args.group
    if group not in ("L", "D", "A"):
        raise SystemExit("adaptation group must be L/D/A")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    tokenizer = load_tokenizer(str(MODEL_PATH), None, local_files_only=True)
    scheduler = load_scheduler(str(MODEL_PATH), None, local_files_only=True)
    te = load_text_encoder(str(MODEL_PATH), None, torch.float32,
                           local_files_only=True).to(device).eval()
    token_ids = add_tokens(tokenizer, te)
    rows, parent_meta = _parent_rows(args, image_id, device)
    c = _condition(args, image_id, rows, device, tokenizer, te, token_ids, cache_root).float().to(device)
    f = torch.load(cache_root / image_id / "clip_feature.pt", map_location="cpu")["feature"].float()
    if f.ndim == 2:
        f = f.unsqueeze(1)
    f = f.to(device)
    del te
    gc.collect(); torch.cuda.empty_cache()

    unet = load_unet(str(MODEL_PATH), None, local_files_only=True).to(device, dtype=torch.bfloat16)
    unet.train()
    unet.requires_grad_(False)
    lora = attach_lora(unet, rank=int(proto["lora_rank"]), alpha=int(proto["lora_alpha"]),
                       dropout=float(proto["lora_dropout"]), init_seed=int(proto["init_seed"]),
                       log=logger.info)
    lora.requires_grad_(True)
    adapter = None
    if group in ("D", "A"):
        adapter = ImageAdapter(1024, init_seed=int(proto["init_seed"])).to(device)
        adapter.train()
    params = lora.trainable_parameters() + (adapter.trainable_parameters() if adapter else [])
    optimizer = torch.optim.AdamW(params, lr=float(proto["learning_rate"]),
                                  betas=tuple(proto["betas"]), eps=float(proto["eps"]),
                                  weight_decay=float(proto["weight_decay"]))
    gens = {name: torch.Generator(device="cpu").manual_seed(
        stream_seed(int(proto["train_seed"]), image_id, name))
        for name in ("vae", "noise", "timestep")}
    meta = {
        "group": group, "image_id": image_id, "parent_step": int(parent_meta.get("step", 0)),
        "parent_token_hash": tensor_hash(rows.cpu()),
        "model_index_sha256": sha256_file(MODEL_PATH / "model_index.json"),
        "lora_init_seed": int(proto["init_seed"]),
        "config_hash": canonical_hash(proto), "engineering": bool(args.engineering),
    }
    start = 1
    if args.resume:
        from crossinit.disen_state import latest_checkpoint, restore_rng
        found = latest_checkpoint(run_dir)
        if found:
            step, path = found
            state = torch.load(path / "trainable.pt", map_location="cpu")["state"]
            lora_state = {k[len("lora."):]: v for k, v in state.items() if k.startswith("lora.")}
            lora.load_state_dict(lora_state, strict=True)
            if adapter is not None:
                adapter.load_state_dict({k[len("adapter."):]: v for k, v in state.items()
                                         if k.startswith("adapter.")}, strict=True)
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
            restore_rng(torch.load(path / "rng.pt", map_location="cpu"), gens)
            start = step + 1
            logger.info("resumed %s at step %d", group, step)

    max_steps = int(args.max_updates or proto["optimizer_updates"])
    gl = protocol()["group_losses"][group]
    cw, ww = float(gl["contrastive_weight"]), float(gl["weak_weight"])
    t_last = time.time()
    for step in range(start, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        lora.zero_grad()
        if adapter is not None:
            adapter.zero_grad()
        latents = _view_batch(image_id, 0, device, gens["vae"], cache_root)
        noise = torch.randn(latents.shape, generator=gens["noise"], dtype=torch.float32).to(device)
        ts = torch.randint(0, int(scheduler.config.num_train_timesteps), (1,),
                           generator=gens["timestep"]).long().to(device)
        noisy = scheduler.add_noise(latents, noise, ts)
        target = _target(scheduler, latents, noise, ts)

        if group == "L":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(noisy, ts, c).sample
            precise = F.mse_loss(pred.float(), target.float())
            loss1 = precise
            contrast = torch.zeros((), device=device)
            weak = torch.zeros((), device=device)
            loss1.backward()
        else:
            u = adapter(f)
            cond = c + u
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(noisy, ts, cond).sample
            precise = F.mse_loss(pred.float(), target.float())
            contrast = contrastive_cosine(c, u)
            loss1 = precise + cw * contrast
            loss1.backward()
            weak = torch.zeros((), device=device)
            if group == "D":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_weak = unet(noisy, ts, c).sample
                weak = F.mse_loss(pred_weak.float(), target.float())
                (ww * weak).backward()
        lora_gn = lora.grad_norm()
        lora.clip_grad_norm(float(proto["lora_grad_clip"]))
        adapter_gn = adapter.grad_norm() if adapter is not None else 0.0
        optimizer.step()
        now = time.time()
        total = float(precise.detach()) + float(contrast.detach()) * cw \
            + (float(weak.detach()) * ww if group == "D" else 0.0)
        rec = {"step": step, "group": group, "image_id": image_id,
               "precise": float(precise.detach()), "weak": float(weak.detach()),
               "contrast": float(contrast.detach()), "total": total,
               "lora_grad_norm": lora_gn, "adapter_grad_norm": adapter_gn,
               "lr": float(optimizer.param_groups[0]["lr"]),
               "sec_per_update": round(now - t_last, 4), "timestep": int(ts.item()),
               **gpu_stats()}
        t_last = now
        jsonl.write(rec)
        if step % 20 == 0 or step == 1:
            logger.info("%s step %d/%d precise=%.4f weak=%.4f contrast=%.4f lora_gn=%.2e %.2fs",
                        group, step, max_steps, rec["precise"], rec["weak"],
                        rec["contrast"], lora_gn, rec["sec_per_update"])
        if step in set(int(x) for x in proto["checkpoints"]) or step == max_steps:
            state = {f"lora.{k}": v for k, v in lora.state_dict().items()}
            if adapter is not None:
                state.update({f"adapter.{k}": v for k, v in adapter.state_dict().items()})
            save_checkpoint(run_dir, step, state, optimizer, rng_state(gens), meta)
        if step >= max_steps:
            break
    state = {f"lora.{k}": v for k, v in lora.state_dict().items()}
    if adapter is not None:
        state.update({f"adapter.{k}": v for k, v in adapter.state_dict().items()})
    meta["lora_hash"] = lora_hash(lora)
    meta["lora_parameters"] = lora.num_parameters()
    meta["adapter_parameters"] = adapter.num_parameters() if adapter else 0
    write_final(run_dir, max_steps, state, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--freeze", default="locks/disen_formal_v1/freeze.json")
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--group", required=True, choices=["B", "L", "D", "A"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--engineering", action="store_true")
    ap.add_argument("--max-updates", type=int, default=None)
    args = ap.parse_args()
    freeze = load_freeze(args)
    roots, cache_root = base_dirs(args)
    run_dir = roots / str(args.image_id) / args.group
    if not args.engineering and is_complete(run_dir) and not args.resume:
        print(f"{run_dir} already complete")
        return
    if not args.engineering and not args.resume and run_dir.exists():
        # Re-running an incomplete job is safe; stale final/checkpoints are ignored.
        pass
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"disen.{args.group}.{args.image_id}", run_dir / "train.log")
    jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    mark_running(run_dir, os.getpid(), 0)
    write_json(run_dir / "config.json", {
        "args": vars(args), "freeze_hash": freeze.get("freeze_hash"),
        "group": args.group, "image_id": str(args.image_id),
        "pid": os.getpid(),
    })
    try:
        if args.group == "B":
            train_b(args, run_dir, cache_root, logger, jsonl)
        else:
            train_adapt(args, run_dir, cache_root, logger, jsonl)
    except Exception as exc:
        mark_failed(run_dir, repr(exc))
        logger.exception("job failed")
        raise


if __name__ == "__main__":
    main()
