"""DisenBooth image adapter (official two-Linear residual form).

Source: DisenBooth official code at commit 9956ab6.  The frozen CLIP image
feature is masked by a learned sigmoid gate and passed through a two-layer
1024-wide residual MLP.  Output shape [1,1,1024] is broadcast over all 77 text
positions by the trainer.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAdapter(nn.Module):
    def __init__(self, dim: int = 1024, init_seed: int = 20261121):
        super().__init__()
        self.dim = int(dim)
        self.mask_logits = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.linear1 = nn.Linear(dim, dim, dtype=torch.float32)
        self.linear2 = nn.Linear(dim, dim, dtype=torch.float32)
        self._init_identical(init_seed)

    def _init_identical(self, init_seed: int) -> None:
        """Identical initial tensors for D and A, independent of module order."""
        gen = torch.Generator(device="cpu").manual_seed(int(init_seed))
        with torch.no_grad():
            for m in (self.linear1, self.linear2):
                bound = 1.0 / (m.in_features ** 0.5)
                m.weight.uniform_(-bound, bound, generator=gen)
                m.bias.uniform_(-bound, bound, generator=gen)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # feature: [B,1,1024] frozen CLIP image embedding.
        masked = torch.sigmoid(self.mask_logits) * feature.float()
        return masked + self.linear2(F.relu(self.linear1(masked)))

    def trainable_parameters(self) -> list[nn.Parameter]:
        out: list[nn.Parameter] = [self.mask_logits]
        for p in self.linear1.parameters():
            out.append(p)
        for p in self.linear2.parameters():
            out.append(p)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "mask_logits": self.mask_logits.detach().cpu().clone(),
            "linear1.weight": self.linear1.weight.detach().cpu().clone(),
            "linear1.bias": self.linear1.bias.detach().cpu().clone(),
            "linear2.weight": self.linear2.weight.detach().cpu().clone(),
            "linear2.bias": self.linear2.bias.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> None:  # type: ignore[override]
        expected = {"mask_logits", "linear1.weight", "linear1.bias",
                    "linear2.weight", "linear2.bias"}
        if strict and set(state) != expected:
            raise ValueError(f"adapter state keys mismatch: {sorted(state)}")
        with torch.no_grad():
            self.mask_logits.copy_(state["mask_logits"].to(self.mask_logits.device, torch.float32))
            for name in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
                layer, param = name.split(".")
                getattr(getattr(self, layer), param).copy_(
                    state[name].to(device=self.mask_logits.device, dtype=torch.float32))

    def grad_norm(self) -> float:
        vals = [p.grad.detach().norm() for p in self.trainable_parameters() if p.grad is not None]
        return float(torch.stack(vals).norm()) if vals else 0.0

    def zero_grad(self) -> None:
        for p in self.trainable_parameters():
            p.grad = None


def contrastive_cosine(c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Cosine term used with a positive weight, i.e. c and u are pushed apart.

    c: [B,77,1024], u: [B,1,1024].  Mean over the 77 text positions; no padding
    mask, matching the released implementation.  The loss adds
    ``+0.001 * cosine(c.mean(1), u)``, so optimisation lowers this similarity;
    it is a disentanglement/contrast term, not an alignment term.
    """
    return F.cosine_similarity(c.mean(dim=1), u.squeeze(1), dim=-1).mean()


def save_adapter(path: str | Path, adapter: ImageAdapter, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": adapter.state_dict(), "meta": meta or {},
               "params": adapter.num_parameters()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_adapter(path: str | Path, adapter: ImageAdapter) -> dict:
    payload = torch.load(path, map_location="cpu")
    adapter.load_state_dict(payload["state"], strict=True)
    return payload.get("meta", {})
