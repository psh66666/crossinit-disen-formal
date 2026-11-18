"""Rank-4 LoRA on every U-Net attention projection (DisenBooth training unit).

Only ``to_q/to_k/to_v/to_out.0`` inside diffusers ``CrossAttention`` modules
are wrapped. The frozen base Linear stays in place, so the zero-initialised up
projection makes the wrapped module byte-identical to the unwrapped one at
initialisation.  LoRA tensors stay FP32 even when the U-Net runs BF16: the
rank-4 matmuls are tiny, and FP32 keeps the optimizer/checkpoint numerics
independent of autocast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


def _seed_for(name: str, init_seed: int) -> int:
    h = hashlib.sha256(f"{init_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, name: str, rank: int = 4, alpha: int = 4,
                 dropout: float = 0.0, init_seed: int = 20261121):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.name = name
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(_seed_for(name, init_seed))
        with torch.no_grad():
            self.down.weight.normal_(0.0, 1.0 / max(1, rank), generator=gen)
            self.up.weight.zero_()
        # Keep the adapter on the same device as the frozen base.
        self.to(device=base.weight.device)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # diffusers 0.21.2 calls ``to_q(x, scale=attention_scale)``.  That scale
        # is applied by the attention module itself; the base Linear ignores it
        # and our residual must not apply it a second time.
        y = self.base(x)
        # Always do the low-rank path in FP32. Casting back to y.dtype keeps
        # the residual in the U-Net's activation dtype.
        delta = self.up(self.down(self.dropout(x.float())))
        return y + (self.scale * delta).to(dtype=y.dtype)

    def extra_repr(self) -> str:
        return f"{self.name} r={self.rank} alpha={self.alpha} scale={self.scale}"


def _attention_modules(unet: nn.Module):
    # diffusers 0.21.2 names the class ``Attention`` (both attn1 and attn2);
    # later versions call the cross-attention class ``CrossAttention``.  Match
    # the structural contract instead of a version-specific class name.
    for name, module in unet.named_modules():
        if not all(hasattr(module, a) for a in ("to_q", "to_k", "to_v", "to_out")):
            continue
        if not hasattr(module, "to_out") or len(module.to_out) == 0:
            continue
        yield name, module


def attach_lora(unet: nn.Module, rank: int = 4, alpha: int = 4,
                dropout: float = 0.0, init_seed: int = 20261121,
                log=print) -> "LoRAHandle":
    if getattr(unet, "_disen_lora", None) is not None:
        raise RuntimeError("LoRA is already attached to this U-Net")
    layers: dict[str, LoRALinear] = {}
    for name, module in _attention_modules(unet):
        for attr in ("to_q", "to_k", "to_v"):
            base = getattr(module, attr)
            key = f"{name}.{attr}"
            wrapped = LoRALinear(base, key, rank, alpha, dropout, init_seed)
            setattr(module, attr, wrapped)
            layers[key] = wrapped
        base = module.to_out[0]
        key = f"{name}.to_out.0"
        wrapped = LoRALinear(base, key, rank, alpha, dropout, init_seed)
        module.to_out[0] = wrapped
        layers[key] = wrapped
    handle = LoRAHandle(unet, layers)
    unet._disen_lora = handle
    log(f"LoRA attached: {len(layers)} projections, rank={rank}, alpha={alpha}, "
        f"trainable={handle.num_parameters()}")
    return handle


class LoRAHandle:
    def __init__(self, unet: nn.Module, layers: dict[str, LoRALinear]):
        self.unet = unet
        self.layers = layers
        self._scale = 1.0

    def names(self) -> list[str]:
        return list(self.layers)

    def trainable_parameters(self) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        for layer in self.layers.values():
            out.extend([layer.down.weight, layer.up.weight])
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, layer in self.layers.items():
            out[f"{name}.down"] = layer.down.weight.detach().cpu().clone()
            out[f"{name}.up"] = layer.up.weight.detach().cpu().clone()
        return out

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> None:
        missing = [n for n in self.layers if f"{n}.down" not in state or f"{n}.up" not in state]
        extra = [k for k in state if k.rsplit(".", 1)[0] not in self.layers]
        if strict and (missing or extra):
            raise ValueError(f"LoRA state mismatch: missing={missing[:4]} extra={extra[:4]}")
        with torch.no_grad():
            for name, layer in self.layers.items():
                if f"{name}.down" not in state:
                    continue
                layer.down.weight.copy_(state[f"{name}.down"].to(
                    device=layer.down.weight.device, dtype=torch.float32))
                layer.up.weight.copy_(state[f"{name}.up"].to(
                    device=layer.up.weight.device, dtype=torch.float32))

    def grad_norm(self) -> float:
        vals = [p.grad.detach().norm() for p in self.trainable_parameters() if p.grad is not None]
        return float(torch.stack(vals).norm()) if vals else 0.0

    def zero_grad(self) -> None:
        for p in self.trainable_parameters():
            p.grad = None

    def clip_grad_norm(self, max_norm: float) -> float:
        params = [p for p in self.trainable_parameters() if p.grad is not None]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    def set_scale(self, scale: float) -> None:
        self._scale = float(scale)
        for layer in self.layers.values():
            layer.scale = float(scale)

    def requires_grad_(self, flag: bool) -> None:
        for p in self.trainable_parameters():
            p.requires_grad_(flag)

    def summary(self) -> dict:
        return {
            "rank": next(iter(self.layers.values())).rank if self.layers else None,
            "alpha": next(iter(self.layers.values())).alpha if self.layers else None,
            "projections": self.names(),
            "trainable_parameters": self.num_parameters(),
        }


def lora_hash(handle: LoRAHandle) -> str:
    h = hashlib.sha256()
    for name in sorted(handle.layers):
        layer = handle.layers[name]
        h.update(name.encode())
        h.update(layer.down.weight.detach().cpu().numpy().tobytes())
        h.update(layer.up.weight.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def save_lora(path: str | Path, handle: LoRAHandle, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": handle.state_dict(), "meta": meta or {}, "summary": handle.summary()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_lora(path: str | Path, handle: LoRAHandle, expected: dict | None = None) -> dict:
    payload = torch.load(path, map_location="cpu")
    summary = payload.get("summary", {})
    if expected:
        for key in ("rank", "alpha"):
            if key in expected and summary.get(key) != expected[key]:
                raise ValueError(f"{path}: {key}={summary.get(key)} expected {expected[key]}")
    handle.load_state_dict(payload["state"], strict=True)
    return payload.get("meta", {})
