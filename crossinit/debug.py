"""Shared fixed-input setup for stage B/C checks (equivalence, gradients)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from crossinit.dataset import TextualInversionMaskDataset
from crossinit.init import load_init_bundle
from crossinit.logging_utils import seed_everything
from crossinit.pipeline_io import (
    dtype_from_name,
    load_scheduler,
    load_text_encoder,
    load_tokenizer,
    load_unet,
    load_vae,
    resolve_model_source,
)


@dataclass
class DebugSetup:
    cfg: dict
    identity: str
    device: torch.device
    dtype: torch.dtype
    tokenizer: object
    text_encoder: object
    unet: object
    vae: object
    scheduler: object
    tokens: list[str]
    token_ids: list[int]
    input_ids: torch.Tensor
    pixel: torch.Tensor
    mask: torch.Tensor
    latents: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    noisy: torch.Tensor


@torch.no_grad()
def build_debug_setup(
    cfg: dict,
    identity: str,
    seed: int = 20261117,
    device: str | None = None,
) -> DebugSetup:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = dtype_from_name(cfg["model"]["weight_dtype"]) if device.type == "cuda" else torch.float32
    source, revision = resolve_model_source(cfg)
    local_only = bool(cfg["model"].get("local_files_only", False))

    tokenizer = load_tokenizer(source, revision, local_files_only=local_only)
    variant = cfg["model"].get("variant")
    text_encoder = load_text_encoder(source, revision, torch.float32, local_files_only=local_only, variant=variant)
    text_encoder.to(device).eval()
    unet = load_unet(source, revision, local_files_only=local_only, variant=variant).to(device, dtype=dtype).eval()
    unet.requires_grad_(False)
    vae = load_vae(source, revision, local_files_only=local_only, variant=variant).to(device, dtype=dtype).eval()
    vae.requires_grad_(False)
    scheduler = load_scheduler(source, revision, local_files_only=local_only)

    n_tokens = int(cfg["train"]["n_tokens"])
    tokens = [f"<{identity}>_v{i}" for i in range(n_tokens)]
    tokenizer.add_tokens(tokens)
    token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in tokens]
    text_encoder.resize_token_embeddings(len(tokenizer))
    init_embeds, _ = load_init_bundle(cfg["train"]["init_embeds"], device=device, dtype=torch.float32)
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight.data[token_ids] = init_embeds

    # Match train.py: only the token embedding matrix is trainable.
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    seed_everything(seed)
    ds = TextualInversionMaskDataset(
        data_root=Path(cfg["data"]["root"]) / identity,
        tokenizer=tokenizer,
        placeholder_tokens=tokens,
        size=int(cfg["data"]["resolution"]),
        repeats=1,
        flip_p=float(cfg["data"]["flip_p"]),
        center_crop=bool(cfg["data"]["center_crop"]),
        text_template=str(cfg["data"]["text_template"]),
        require_shifted=False,
    )
    item = ds[0]
    pixel = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
    mask = item["mask"].unsqueeze(0).to(device, dtype=torch.float32)
    input_ids = item["input_ids"].unsqueeze(0).to(device)

    latents = vae.encode(pixel).latent_dist.mode().detach() * vae.config.scaling_factor
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    noise = torch.randn(latents.shape, generator=gen, device=device, dtype=latents.dtype)
    timesteps = torch.tensor([500], device=device, dtype=torch.long)
    noisy = scheduler.add_noise(latents, noise, timesteps)

    return DebugSetup(
        cfg=cfg, identity=identity, device=device, dtype=dtype,
        tokenizer=tokenizer, text_encoder=text_encoder, unet=unet, vae=vae,
        scheduler=scheduler, tokens=tokens, token_ids=token_ids,
        input_ids=input_ids, pixel=pixel, mask=mask, latents=latents,
        noise=noise, timesteps=timesteps, noisy=noisy,
    )
