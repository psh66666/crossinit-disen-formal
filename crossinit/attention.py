"""Capture cross-attention probabilities for selected attn2 layers.

Why this is not a one-line hook: diffusers 0.21.2 uses AttnProcessor2_0 (SDPA) by
default, and SDPA never materialises the probability matrix. xFormers likewise
returns only the output. For the selected layers we therefore install the plain
`AttnProcessor` and wrap `Attention.get_attention_scores`, which is the exact
tensor the forward pass already uses -- so lambda=0 does not change the math,
only which tensors we keep a reference to.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

from diffusers.models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor


def _is_cross_attention(name: str, mod: Attention) -> bool:
    # The `attn1`/`attn2` suffix is the only reliable signal in SD UNet:
    # diffusers sometimes exposes cross_attention_dim on self-attention too.
    if name.endswith("attn2"):
        return True
    if name.endswith("attn1"):
        return False
    flag = getattr(mod, "is_cross_attention", None)
    if flag is not None:
        return bool(flag)
    return getattr(mod, "cross_attention_dim", None) is not None


def _all_cross_attn(unet) -> dict[str, Attention]:
    return {
        name: mod
        for name, mod in unet.named_modules()
        if isinstance(mod, Attention) and _is_cross_attention(name, mod)
    }


def all_cross_attention_names(unet) -> list[str]:
    """All attn2 module names in registration/forward order (16 for SD 2.1)."""
    return list(_all_cross_attn(unet))


def _seq_len(hidden_states: torch.Tensor) -> int:
    if hidden_states.ndim == 4:
        return int(hidden_states.shape[-2] * hidden_states.shape[-1])
    return int(hidden_states.shape[1])


@torch.no_grad()
def discover_query_resolutions(
    unet,
    device,
    dtype=torch.float16,
    latent_size: int = 64,
) -> dict[str, int]:
    """Run one dummy UNet forward and record each cross-attn query length.

    The module name alone is not a reliable resolution indicator if the UNet
    config changes; this probes the real forward order.
    """
    modules = _all_cross_attn(unet)
    shapes: dict[str, int] = {}
    handles = []

    def make_hook(name):
        def hook(_module, args, _kwargs):
            if args:
                shapes[name] = _seq_len(args[0])

        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name), with_kwargs=True))

    was_training = unet.training
    unet.eval()
    try:
        # SD 2.1 has 4 latent channels; cross-attn conditioning is 1024-d.
        latents = torch.zeros(1, 4, latent_size, latent_size, device=device, dtype=dtype)
        t = torch.zeros(1, device=device, dtype=torch.long)
        enc = torch.zeros(1, 77, 1024, device=device, dtype=dtype)
        unet(latents, t, enc)
    finally:
        for h in handles:
            h.remove()
        unet.train(was_training)
    return shapes


def resolve_layer_names(
    unet,
    spec,
    *,
    resolutions: Iterable[int] = (32, 16),
    device=None,
    dtype=torch.float16,
) -> list[str]:
    """Resolve config `loc.layers` (explicit names or "auto") to module names."""
    modules = _all_cross_attn(unet)
    if isinstance(spec, (list, tuple)) and len(spec) > 0:
        names = [str(s) for s in spec]
    elif spec in (None, "auto"):
        probe_device = device if device is not None else torch.device("cpu")
        probe_dtype = torch.float32 if probe_device.type == "cpu" else dtype
        shapes = discover_query_resolutions(unet, probe_device, dtype=probe_dtype)
        # Stable order = registration order = forward order for down/mid/up blocks.
        names = []
        for res in resolutions:
            q = int(res) ** 2
            match = next((n for n in shapes if shapes[n] == q), None)
            if match is None:
                raise ValueError(f"no cross-attention layer with query {res}x{res} found")
            names.append(match)
    else:
        raise ValueError(f"loc.layers must be a list of names or 'auto', got {spec!r}")

    for name in names:
        if name not in modules:
            raise KeyError(f"{name} is not a cross-attention module in this UNet")
    return names


class AttentionCapture:
    """Install capture on selected attn2 layers and restore them afterwards."""

    def __init__(self, unet, layer_names: Iterable[str]):
        self.unet = unet
        self.layer_names = list(layer_names)
        self.heads: dict[str, int] = {}
        self.probs: dict[str, torch.Tensor] = {}
        self._saved: list[tuple[Attention, object, object]] = []

    @staticmethod
    def _set_processor(mod: Attention, processor) -> None:
        if hasattr(mod, "set_processor"):
            mod.set_processor(processor)
        else:  # older diffusers
            mod.processor = processor

    def install(self) -> None:
        processors = self.unet.attn_processors
        modules = _all_cross_attn(self.unet)
        for name in self.layer_names:
            mod = modules[name]
            # `unet.attn_processors` keys are "<module>.processor"; get_processor()
            # is the version-independent way to read the current one.
            current = mod.get_processor() if hasattr(mod, "get_processor") else processors.get(f"{name}.processor")
            if isinstance(current, XFormersAttnProcessor):
                raise RuntimeError(
                    f"{name} uses xFormers attention, which never materialises "
                    "probabilities. Do not enable xformers when loc.enabled."
                )
            # Save both the processor and the bound method we are about to wrap.
            orig_scores = mod.get_attention_scores
            self._set_processor(mod, AttnProcessor())
            self.heads[name] = int(mod.heads)

            def wrapper(query, key, attention_mask=None, *args, _orig=orig_scores,
                        _name=name, **kwargs):
                probs = _orig(query, key, attention_mask, *args, **kwargs)
                self.probs[_name] = probs
                return probs

            mod.get_attention_scores = wrapper
            self._saved.append((mod, orig_scores, current))

    def clear(self) -> None:
        self.probs.clear()

    def remove(self) -> None:
        for mod, orig_scores, orig_processor in reversed(self._saved):
            mod.get_attention_scores = orig_scores
            if orig_processor is not None:
                self._set_processor(mod, orig_processor)
        self._saved.clear()
        self.probs.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()


def token_mask_from_ids(input_ids: torch.Tensor, token_ids: Iterable[int]) -> torch.Tensor:
    """(B, L) bool mask of the two placeholder token positions.

    Positions differ across samples because the training template is sampled at
    random, so the loss must not assume a fixed index.
    """
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for tid in token_ids:
        mask |= input_ids == int(tid)
    return mask


def grid_shape(q_len: int) -> tuple[int, int]:
    h = int(round(math.sqrt(q_len)))
    if h * h != q_len:
        raise ValueError(f"cross-attention query length {q_len} is not a square grid")
    return h, h
