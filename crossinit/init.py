"""Cross Initialization (v_init = E(v_bar_691)) and checkpoint I/O.

Ported from the official implementation (MIT):
  https://github.com/lyuPang/CrossInitialization  commit 0b1e39a
  utils.py: celeb_names_cross_init / token_cross_init

The paper (Eq. 3) says "initialize with E(v_bar_691)"; the precise construction
is: for each of the 691 names, take the content token at position 1 and the one
at position 2, average each column separately, put the two means after BOS/EOS/PAD
into the text encoder, and read output positions 1..2. (The paper describes these
as first/last name; the released code simply takes the first two content tokens.)
Averaging the raw lookup rows and stopping there is NOT the method, and neither is
encoding the mean sequence without reading the encoder output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch

from crossinit.clip_model import CLIPTextModel


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def celeb_names_cross_init(
    celeb_path: str | Path,
    tokenizer,
    text_encoder: CLIPTextModel,
    n_tokens: int = 2,
) -> torch.Tensor:
    """Return (n_tokens, hidden) initialization from the 691-name list."""
    names = [n for n in Path(celeb_path).read_text(encoding="utf-8").splitlines() if n.strip()]
    if len(names) == 0:
        raise ValueError(f"empty celebrity name file: {celeb_path}")

    columns: list[list[torch.Tensor]] = [[] for _ in range(n_tokens)]
    weight = text_encoder.get_input_embeddings().weight
    for name in names:
        ids = tokenizer(
            name,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        rows = weight.data[ids.to(weight.device)]
        # Official utils.py: range(1, min(len-1, n_column+1)). One-token names
        # (Cher, Drake, ...) contribute only to column 0 instead of crashing.
        for pos in range(1, min(ids.shape[0] - 1, n_tokens + 1)):
            columns[pos - 1].append(rows[pos].unsqueeze(0))

    empty = [i for i, col in enumerate(columns) if len(col) == 0]
    if empty:
        raise ValueError(f"no celebrity name contributed to columns {empty}; check tokenizer/names")
    col_means = torch.cat(
        [torch.cat(col).mean(dim=0, keepdim=True) for col in columns], dim=0
    )  # (n_tokens, 1024)

    embeds = text_encoder.get_input_embeddings().weight.data
    bos, eos, pad = embeds[[tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]]
    n_pad = tokenizer.model_max_length - col_means.shape[0] - 2
    inputs_embeds = torch.cat(
        [bos.unsqueeze(0), col_means, eos.unsqueeze(0), pad.repeat(n_pad, 1)], dim=0
    ).unsqueeze(0)
    # Cross Initialization: run the text encoder on the mean embedding sequence.
    out = text_encoder(inputs_embeds=inputs_embeds)[0][0, 1 : 1 + n_tokens]
    return out.detach().clone()


@torch.no_grad()
def token_cross_init(
    tokens: str | Sequence[str],
    tokenizer,
    text_encoder: CLIPTextModel,
    return_first_embeds: bool = False,
) -> torch.Tensor:
    """Initialize from arbitrary token(s); useful for sanity checks."""
    if not isinstance(tokens, str):
        tokens = " ".join(tokens)
    ids = tokenizer(
        tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device)
    if return_first_embeds:
        embeds = text_encoder.get_input_embeddings().weight.data[ids[0]]
    else:
        embeds = text_encoder(ids)[0][0]
    return embeds[1:-1].detach().clone()


def save_init_bundle(
    path: str | Path,
    embeds: torch.Tensor,
    *,
    celeb_path: str | Path,
    model_id: str,
    revision: str | None,
    n_names: int,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "cross_init_celeb_mean",
        "celeb_path_name": Path(celeb_path).name,
        "celeb_sha256": sha256_file(celeb_path),
        "n_names": int(n_names),
        "model_id": model_id,
        "revision": revision,
        "token_count": int(embeds.shape[0]),
        "hidden_size": int(embeds.shape[1]),
        "dtype": str(embeds.dtype),
    }
    torch.save({"embeds": embeds.detach().cpu(), "meta": meta}, path)
    return meta


def load_init_bundle(path: str | Path, device="cpu", dtype=torch.float32) -> tuple[torch.Tensor, dict]:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "embeds" not in bundle:
        raise ValueError(f"{path}: expected a bundle with an 'embeds' tensor")
    embeds = bundle["embeds"].to(device=device, dtype=dtype)
    return embeds, dict(bundle.get("meta", {}))
