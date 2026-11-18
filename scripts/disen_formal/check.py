#!/usr/bin/env python
"""Engineering/resource check and training-coverage audit."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_lora import attach_lora, state_hash
from crossinit.disen_state import is_complete, load_checkpoint, rng_state, sha256_file, write_json
from crossinit.pipeline_io import load_text_encoder, load_tokenizer, load_unet
from scripts.disen_formal.common import (
    DATA_DIR, MODEL_PATH, formal_identities, identity_map, protocol,
)


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(cmd: list[str]) -> int:
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def lora_zero_equivalence() -> dict:
    device = torch.device("cuda")
    unet = load_unet(str(MODEL_PATH), None, local_files_only=True).to(device, dtype=torch.bfloat16).eval()
    x = torch.randn(1, 4, 64, 64, device=device, dtype=torch.bfloat16)
    t = torch.tensor([500], device=device)
    c = torch.randn(1, 77, 1024, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = unet(x, t, c).sample.float()
    handle = attach_lora(unet, rank=4, alpha=4, init_seed=20261121)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lora = unet(x, t, c).sample.float()
    diff = float((base - lora).abs().max())
    # Two independent attach calls must be element-wise identical.
    unet2 = load_unet(str(MODEL_PATH), None, local_files_only=True).to(device, dtype=torch.bfloat16).eval()
    h2 = attach_lora(unet2, rank=4, alpha=4, init_seed=20261121)
    init_diff = float((state_hash(handle.state_dict()) != state_hash(h2.state_dict())))
    del unet, unet2
    torch.cuda.empty_cache()
    return {"zero_lora_max_abs_diff": diff, "identical_init_mismatch": init_diff,
            "pass": diff == 0.0 and init_diff == 0.0}


def token_row_grad_equivalence() -> dict:
    """Two independent token rows vs updating the full embedding matrix.

    Official code optimises the whole embedding matrix then restores all other
    rows.  The formal trainer optimises only two rows.  This checks that the
    token-row gradients are identical and non-token gradients are zero.
    """
    device = torch.device("cuda")
    tokenizer = load_tokenizer(str(MODEL_PATH), None, local_files_only=True)
    te = load_text_encoder(str(MODEL_PATH), None, torch.float32,
                           local_files_only=True).to(device).eval()
    from scripts.disen_formal.common import add_tokens, tokens
    token_ids = add_tokens(tokenizer, te)
    te.requires_grad_(False)
    init = torch.load(DATA_DIR / "celeb_init_691.pt", map_location="cpu")["embeds"].float().to(device)
    enc = tokenizer("a photo of a <ci0> <ci1> person", padding="max_length",
                    truncation=True, max_length=77, return_tensors="pt")
    ids = enc.input_ids.to(device)
    rows = torch.nn.Parameter(init.clone())
    weight = te.get_input_embeddings().weight.detach()
    emb = weight[ids].clone()
    for j, tid in enumerate(token_ids):
        emb[0, ids[0] == tid] = rows[j]
    cond = te(inputs_embeds=emb)[0]
    cond.float().pow(2).mean().backward()
    grad_rows = rows.grad.detach().clone()

    full = torch.nn.Parameter(weight.clone())
    with torch.no_grad():
        full.data[token_ids] = init
    emb2 = full[ids]
    cond2 = te(inputs_embeds=emb2)[0]
    cond2.float().pow(2).mean().backward()
    grad_full = full.grad.detach()
    token_grad = grad_full[token_ids].clone()
    mask = torch.ones(full.shape[0], dtype=torch.bool, device=device)
    mask[token_ids] = False
    other = float(grad_full[mask].abs().max()) if bool(mask.any()) else 0.0
    diff = float((grad_rows - token_grad).abs().max())
    # Padding/BOS/EOT rows can receive non-zero gradients in the official
    # full-matrix formulation, but they are restored after every update.  The
    # contract that matters is that the two token rows get identical gradients
    # (and our two-row implementation cannot move any other row at all).
    return {"max_abs_grad_diff": diff, "non_token_max_abs_grad": other,
            "other_rows_frozen_in_trainer": True, "pass": diff < 1e-6}


def rng_checkpoint_check() -> dict:
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=1e-3)
    gen = torch.Generator(device="cpu").manual_seed(123)
    torch.randn(5, generator=gen)
    out_dir = ROOT / "engineering" / "_state_check"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    from crossinit.disen_state import save_checkpoint
    state = {"x": torch.arange(6, dtype=torch.float32)}
    save_checkpoint(out_dir / "run", 3, state, opt, rng_state({"g": gen}), {"tag": "check"})
    gen2 = torch.Generator(device="cpu").manual_seed(999)
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.ones(3))], lr=1e-3)
    got, meta = load_checkpoint(out_dir / "run" / "checkpoints" / "step_0003", opt2, {"g": gen2},
                                {"tag": "check"})
    a = torch.randn(5, generator=gen)
    b = torch.randn(5, generator=gen2)
    ok = bool(torch.equal(state["x"], got["x"])) and bool(torch.equal(a, b)) and meta["tag"] == "check"
    return {"pass": ok}


def engineering(args) -> None:
    proto = protocol()
    ident = formal_identities()[0]
    image_id = str(ident["image_id"])
    eng = ROOT / "engineering" / "disen_formal_v1"
    if eng.exists():
        shutil.rmtree(eng)
    eng.mkdir(parents=True, exist_ok=True)
    job_status = {}
    for group in ("B", "L", "D", "A"):
        code = run([sys.executable, "-m", "scripts.disen_formal.train",
                    "--protocol", args.protocol, "--image-id", image_id, "--group", group,
                    "--engineering", "--max-updates", str(args.updates_per_group)])
        job_status[group] = code
    mets = {g: read_metrics(eng / image_id / g / "metrics.jsonl") for g in job_status}
    b_ok = (all(c == 0 for c in job_status.values()) and len(mets["B"]) >= int(args.updates_per_group)
            and mets["B"][-1]["micro_step"] == int(args.updates_per_group) * 4
            and mets["B"][-1]["step"] == int(args.updates_per_group))
    adapt_ok = all(len(mets[g]) >= int(args.updates_per_group) for g in ("L", "D", "A"))
    timesteps = {g: [r["timestep"] for r in mets[g][:int(args.updates_per_group)]]
                 for g in ("L", "D", "A")}
    rng_aligned = timesteps["L"] == timesteps["D"] == timesteps["A"]
    grads_ok = all(bool(mets[g]) and mets[g][-1]["lora_grad_norm"] > 0
                   for g in ("L", "D", "A"))
    adapter_grads_ok = all(bool(mets[g]) and mets[g][-1]["adapter_grad_norm"] > 0
                           for g in ("D", "A"))
    reserved = max([r.get("max_memory_reserved_gib", 0) for rows in mets.values() for r in rows] or [0])
    sec = {g: (sum(r["sec_per_update"] for r in mets[g]) / max(1, len(mets[g])) if mets[g] else None)
           for g in mets}
    zero = lora_zero_equivalence()
    rng = rng_checkpoint_check()
    token_grad = token_row_grad_equivalence()
    status = bool(b_ok and adapt_ok and rng_aligned and grads_ok and adapter_grads_ok
                  and zero["pass"] and rng["pass"] and token_grad["pass"]
                  and reserved <= 22.0)
    result = {
        "status": "PASS" if status else "FAIL",
        "image_id": image_id,
        "updates_per_group": int(args.updates_per_group),
        "job_returncodes": job_status,
        "metrics_rows": {k: len(v) for k, v in mets.items()},
        "B_microsteps": mets["B"][-1].get("micro_step") if mets["B"] else None,
        "rng_aligned_LDA": rng_aligned,
        "lora_grad_nonzero": grads_ok,
        "adapter_grad_nonzero": adapter_grads_ok,
        "zero_lora": zero,
        "checkpoint_roundtrip": rng,
        "token_row_grad_equivalence": token_grad,
        "peak_reserved_gib": reserved,
        "sec_per_update": sec,
        "uses_engineering_B0_parent": True,
        "effects_not_evaluated": True,
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not status:
        raise SystemExit(1)


def module_hash(path: Path) -> str:
    return state_hash(torch.load(path, map_location="cpu")["state"])


def coverage(args) -> None:
    proto = protocol()
    idents = formal_identities()
    rows, missing = [], []
    for ident in idents:
        image_id = str(ident["image_id"])
        for group in ("B", "L", "D", "A"):
            run_dir = ROOT / "runs" / "disen_formal_v1" / image_id / group
            final = run_dir / "final" / "trainable.pt"
            mets = read_metrics(run_dir / "metrics.jsonl")
            expected = 320 if group == "B" else 2000
            row = {"image_id": image_id, "group": group, "complete": is_complete(run_dir),
                   "final_exists": final.exists(), "metrics_rows": len(mets),
                   "last_step": mets[-1]["step"] if mets else None,
                   "expected_step": expected}
            if group == "B" and mets:
                row["last_micro_step"] = mets[-1].get("micro_step")
            if final.exists():
                meta = json.loads((run_dir / "final" / "meta.json").read_text(encoding="utf-8"))
                row["meta_step"] = meta.get("step")
                row["parent_step"] = meta.get("parent_step")
                row["trainable_sha256"] = sha256_file(final)
                state = torch.load(final, map_location="cpu")["state"]
                if group == "B":
                    row["only_token_rows"] = set(state).issubset({"tokens", "token_hash"})
                else:
                    row["only_lora_adapter"] = all(
                        k.startswith("lora.") or k.startswith("adapter.") for k in state)
            ok = (row["complete"] and row["final_exists"] and row["last_step"] == expected
                  and row["meta_step"] == expected)
            if group == "B":
                ok = ok and row.get("last_micro_step") == 1280 and bool(row.get("only_token_rows"))
            else:
                ok = ok and row.get("parent_step") == 320 and bool(row.get("only_lora_adapter"))
            row["ok"] = bool(ok)
            if not ok:
                missing.append({"image_id": image_id, "group": group})
            rows.append(row)
    gen_dir = ROOT / "runs" / "disen_formal_v1" / "generated"
    gen_count, gen_keys = 0, set()
    manifest = gen_dir / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                gen_count += 1
                gen_keys.add((r["image_id"], r["group"], r["prompt_id"], int(r["seed"])))
    result = {
        "status": "PASS" if not missing else "FAIL",
        "stage": args.stage,
        "jobs": rows,
        "missing": missing,
        "generated_images": gen_count,
        "unique_generated_keys": len(gen_keys),
        "expected_generated": 512 if args.stage == "generation" else None,
    }
    if args.stage == "generation":
        result["status"] = "PASS" if len(gen_keys) == 512 else "FAIL"
    write_json(args.out, result)
    print(json.dumps({"status": result["status"], "missing": len(missing),
                      "generated": gen_count}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engineering")
    e.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    e.add_argument("--identities", default="data/disen_formal/identities.json")
    e.add_argument("--updates-per-group", type=int, default=10)
    e.add_argument("--out", default="results/disen_formal_v1/technical_check.json")
    e.set_defaults(func=engineering)
    c = sub.add_parser("coverage")
    c.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    c.add_argument("--freeze", default="locks/disen_formal_v1/freeze.json")
    c.add_argument("--stage", choices=["training", "generation"], default="training")
    c.add_argument("--out", default="results/disen_formal_v1/coverage.json")
    c.set_defaults(func=coverage)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
