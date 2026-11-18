"""Atomic checkpoints for the DisenBooth-formal matrix.

A checkpoint is a directory ``checkpoints/step_XXXX`` with:

  trainable.pt   only the tensors that are allowed to change
  optimizer.pt   AdamW state
  rng.pt         Python/NumPy/Torch CPU/CUDA plus named generator states
  meta.json      step, hashes, parent/model/config binding

``final/`` is the only inference entry point.  ``--resume`` must load the
optimizer and RNG too; loading only LoRA is not a valid resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True,
                                     default=str).encode("utf-8")).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def rng_state(generators: dict[str, torch.Generator] | None = None) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generators": {},
    }
    for name, gen in (generators or {}).items():
        state["generators"][name] = gen.get_state()
    return state


def restore_rng(state: dict, generators: dict[str, torch.Generator] | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, gen in (generators or {}).items():
        saved = state.get("generators", {}).get(name)
        if saved is None:
            raise KeyError(f"checkpoint has no RNG state for generator {name!r}")
        gen.set_state(saved)


def _step_dirs(ckpt_root: Path) -> list[tuple[int, Path]]:
    out = []
    for p in ckpt_root.glob("step_*"):
        m = re.fullmatch(r"step_(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def save_checkpoint(run_dir: str | Path, step: int, trainable: dict[str, torch.Tensor],
                    optimizer: torch.optim.Optimizer, rng: dict, meta: dict) -> Path:
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoints" / f"step_{int(step):04d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"state": trainable}, ckpt / "trainable.pt")
    atomic_torch_save(optimizer.state_dict(), ckpt / "optimizer.pt")
    atomic_torch_save(rng, ckpt / "rng.pt")
    meta = dict(meta)
    meta.update({
        "step": int(step),
        "trainable_sha256": sha256_file(ckpt / "trainable.pt"),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    write_json(ckpt / "meta.json", meta)
    return ckpt


def write_final(run_dir: str | Path, step: int, trainable: dict[str, torch.Tensor],
                meta: dict) -> Path:
    run_dir = Path(run_dir)
    final = run_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"state": trainable}, final / "trainable.pt")
    meta = dict(meta)
    meta.update({"step": int(step), "trainable_sha256": sha256_file(final / "trainable.pt")})
    write_json(final / "meta.json", meta)
    write_json(run_dir / "status.json", {"state": "completed", "step": int(step),
                                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return final


def latest_checkpoint(run_dir: str | Path, upto: int | None = None) -> tuple[int, Path] | None:
    items = _step_dirs(Path(run_dir) / "checkpoints")
    if upto is not None:
        items = [(s, p) for s, p in items if s <= int(upto)]
    return items[-1] if items else None


def load_checkpoint(path: str | Path, optimizer: torch.optim.Optimizer | None,
                    generators: dict[str, torch.Generator] | None,
                    expected_meta: dict | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    path = Path(path)
    trainable = torch.load(path / "trainable.pt", map_location="cpu")["state"]
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if expected_meta:
        for key, value in expected_meta.items():
            if key in meta and meta[key] != value:
                raise ValueError(f"{path}: {key}={meta[key]!r} expected {value!r}")
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
    if generators is not None:
        restore_rng(torch.load(path / "rng.pt", map_location="cpu"), generators)
    return trainable, meta


def load_final(run_dir: str | Path, expected_meta: dict | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    final = Path(run_dir) / "final"
    meta_path = final / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing final checkpoint: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_meta:
        for key, value in expected_meta.items():
            if key in meta and meta[key] != value:
                raise ValueError(f"{final}: {key}={meta[key]!r} expected {value!r}")
    if sha256_file(final / "trainable.pt") != meta["trainable_sha256"]:
        raise ValueError(f"{final}: trainable.pt hash does not match meta.json")
    return torch.load(final / "trainable.pt", map_location="cpu")["state"], meta


def status_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "status.json"


def mark_running(run_dir: str | Path, pid: int, step: int = 0) -> None:
    write_json(status_path(run_dir), {
        "state": "running", "pid": int(pid), "step": int(step),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def mark_failed(run_dir: str | Path, error: str, step: int = 0) -> None:
    write_json(status_path(run_dir), {
        "state": "failed", "error": str(error)[:4000], "step": int(step),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def is_complete(run_dir: str | Path) -> bool:
    final = Path(run_dir) / "final"
    if not (final / "meta.json").exists():
        return False
    try:
        meta = json.loads((final / "meta.json").read_text(encoding="utf-8"))
        return sha256_file(final / "trainable.pt") == meta["trainable_sha256"]
    except (OSError, KeyError, ValueError):
        return False
