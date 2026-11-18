"""Causality-preserved adapter for the CLIP text encoder.

The adapter is a rank-r residual on selected Linear layers. Its mask is zero
before the first new token and on padding positions, so prefix words see the
base encoder exactly. The base Linear is frozen; only down/up are trainable.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CPALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 1):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.down = nn.Linear(base.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, base.out_features, bias=False)
        # A private generator keeps CPA init from consuming the C/M paired
        # data-order RNG.
        gen = torch.Generator(device="cpu").manual_seed(0x0CAFE)
        with torch.no_grad():
            self.down.weight.normal_(0.0, 1.0 / max(1, self.rank), generator=gen)
        nn.init.zeros_(self.up.weight)
        self.to(device=base.weight.device, dtype=base.weight.dtype)
        self._mask = None

    def set_mask(self, mask: torch.Tensor | None):
        self._mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._mask is None:
            return out
        delta = self.up(self.down(x))
        m = self._mask.to(device=x.device, dtype=x.dtype)
        if m.ndim == 2:
            m = m.unsqueeze(-1)
        return out + m * delta


class CPA:
    """Registry of CPALinear wrappers plus save/load helpers.

    Deliberately not an nn.Module: the wrappers already live inside the text
    encoder's module tree, and registering them twice would duplicate parameters.
    """

    def __init__(self):
        self.layers: dict[str, CPALinear] = {}

    @property
    def names(self) -> list[str]:
        return list(self.layers.keys())

    def set_mask(self, mask: torch.Tensor | None):
        for layer in self.layers.values():
            layer.set_mask(mask)

    def trainable_parameters(self):
        return [param for layer in self.layers.values()
                for group in (layer.down.parameters(), layer.up.parameters())
                for param in group if param.requires_grad]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {}
        for name, layer in self.layers.items():
            obj[f"{name}.down.weight"] = layer.down.weight.detach().cpu()
            obj[f"{name}.up.weight"] = layer.up.weight.detach().cpu()
        torch.save(obj, path)

    def load(self, path: str | Path) -> None:
        obj = torch.load(path, map_location="cpu")
        missing = [n for n in self.layers if f"{n}.down.weight" not in obj]
        if missing:
            raise ValueError(f"{path}: missing CPA layers: {missing[:5]}")
        with torch.no_grad():
            for name, layer in self.layers.items():
                layer.down.weight.copy_(obj[f"{name}.down.weight"].to(layer.down.weight.dtype))
                layer.up.weight.copy_(obj[f"{name}.up.weight"].to(layer.up.weight.dtype))


def attach_cpa(text_encoder: nn.Module, rank: int = 1) -> CPA:
    """Wrap every transformer-block MLP fc2 with a masked residual adapter."""
    cpa = CPA()
    targets = []
    for name, module in text_encoder.named_modules():
        if name.endswith("mlp.fc2") and isinstance(module, nn.Linear):
            targets.append((name, module))
    if not targets:
        raise RuntimeError("no mlp.fc2 Linear modules found in text encoder")
    for name, module in targets:
        parent_name, _, child = name.rpartition(".")
        parent = text_encoder.get_submodule(parent_name) if parent_name else text_encoder
        wrapped = CPALinear(module, rank)
        setattr(parent, child, wrapped)
        cpa.layers[name] = wrapped
    text_encoder._cpa = cpa
    return cpa


def get_cpa(text_encoder: nn.Module) -> CPA | None:
    return getattr(text_encoder, "_cpa", None)


def cpa_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_ids: list[int],
) -> torch.Tensor:
    """True from the first new token onward, excluding padding."""
    ids = torch.as_tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
    is_new = (input_ids.unsqueeze(-1) == ids.view(1, 1, -1)).any(dim=-1)
    has = is_new.any(dim=1)
    first = torch.argmax(is_new.long(), dim=1)
    pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    return (pos >= first.unsqueeze(1)) & has.unsqueeze(1) & attention_mask.bool()
