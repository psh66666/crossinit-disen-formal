"""Model / embedding I/O shared by training, generation and diagnostics."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTokenizer

from crossinit.clip_model import CLIPTextModel
from crossinit.init import sha256_file

DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return DTYPES[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype {name!r}; use fp16/bf16/fp32") from exc


def resolve_model_source(cfg: dict) -> tuple[str, str | None]:
    model = cfg["model"]
    local = model.get("local_model_dir")
    if local and Path(local).exists():
        return str(local), None
    return str(model["pretrained_model_name_or_path"]), model.get("revision")


def load_tokenizer(source: str, revision: str | None, local_files_only: bool = False) -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(
        source, subfolder="tokenizer", revision=revision, local_files_only=local_files_only
    )


def load_text_encoder(source: str, revision: str | None, dtype, local_files_only: bool = False, variant=None):
    # transformers 4.25.1 does not accept `variant`; for an fp16-only mirror we
    # create text_encoder/model.safetensors as a symlink to model.fp16.safetensors.
    return CLIPTextModel.from_pretrained(
        source,
        subfolder="text_encoder",
        revision=revision,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )


def load_scheduler(source: str, revision: str | None, local_files_only: bool = False):
    return DDPMScheduler.from_pretrained(
        source, subfolder="scheduler", revision=revision, local_files_only=local_files_only
    )


def load_vae(source: str, revision: str | None, local_files_only: bool = False, variant=None):
    return AutoencoderKL.from_pretrained(
        source, subfolder="vae", revision=revision, variant=variant, local_files_only=local_files_only
    )


def load_unet(source: str, revision: str | None, local_files_only: bool = False, variant=None):
    return UNet2DConditionModel.from_pretrained(
        source, subfolder="unet", revision=revision, variant=variant, local_files_only=local_files_only
    )


def load_pipeline(
    source: str,
    revision: str | None,
    dtype,
    device: str | torch.device = "cuda",
    local_files_only: bool = False,
    variant=None,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        source,
        revision=revision,
        torch_dtype=dtype,
        variant=variant,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def sort_token_names(tokens) -> list[str]:
    """`<id>_v10` must come after `<id>_v2`; dict order is not a contract."""
    def key(tok: str):
        try:
            return int(str(tok).rsplit("_v", 1)[1])
        except (IndexError, ValueError):
            return 0
    return sorted([str(t) for t in tokens], key=key)


def add_tokens_and_load_embeds(
    tokenizer,
    text_encoder,
    embeds: dict[str, torch.Tensor],
    dtype=torch.float16,
) -> list[int]:
    """Add the placeholder tokens and copy the saved rows into the embedding matrix.

    Token strings are identity-specific (`<28017>_v0`), so reusing one process for
    several identities only appends new rows; base rows are never overwritten.
    """
    tokens = sort_token_names(embeds.keys())
    tokenizer.add_tokens(tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    ids = []
    weight = text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        for token in tokens:
            tid = tokenizer.convert_tokens_to_ids(token)
            weight.data[tid] = embeds[token].to(device=weight.device, dtype=weight.dtype)
            ids.append(int(tid))
    return ids


def load_embed_dict(path: str | Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a dict of token -> tensor")
    return {str(k): v.float() for k, v in obj.items()}


def save_embed_rows(path: str | Path, tokens: list[str], rows: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {tok: row.detach().cpu().clone() for tok, row in zip(tokens, rows)}
    torch.save(obj, path)


def model_file_hashes(source: str) -> dict[str, str]:
    """Best-effort hashes of the weights that define the experiment."""
    root = Path(source)
    if not root.exists():
        return {}
    rels = [
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.bin",
        "text_encoder/model.safetensors",
        "text_encoder/pytorch_model.bin",
        "text_encoder/model.fp16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.bin",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "model_index.json",
    ]
    out = {}
    for rel in rels:
        p = root / rel
        if p.exists():
            out[rel] = sha256_file(p)
    return out
