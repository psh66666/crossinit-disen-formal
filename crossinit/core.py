"""Minimal CoRe CER port for CrossInit's two learnable tokens.

CoRe's official code replaces one concept token with a learnable embedding and
compares the *context-token* outputs of a personalized prompt against the same
prompt with the category token. CrossInit has two learnable tokens and a
trailing class word (``person``), so the concept span here is
``<tok0> <tok1> person`` and the reference span is the single ``person``.
Everything outside those spans must tokenize identically; otherwise the pair is
rejected instead of silently comparing the wrong positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

# This tokenizer pads with 0 (observed in SD 2.1 CLIP), not EOT; exclude both.
SPECIAL_IDS = {0, 49406, 49407}  # padding / BOS / EOT


@dataclass(frozen=True)
class CorePair:
    template_id: str
    object_ids: tuple[int, ...]
    reference_ids: tuple[int, ...]
    object_span: tuple[int, int]       # [start, end) in object_ids
    reference_span: tuple[int, int]    # [start, end) in reference_ids
    object_context: tuple[int, ...]    # positions in object_ids
    reference_context: tuple[int, ...] # positions in reference_ids
    context_text: tuple[str, ...]


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(
        text, padding="max_length", truncation=True,
        max_length=int(tokenizer.model_max_length), return_tensors="pt",
    ).input_ids[0].tolist()


def _find(seq: Sequence[int], sub: Sequence[int], start: int = 0) -> int:
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if list(seq[i:i + n]) == list(sub):
            return i
    return -1


def _context_positions(ids: Sequence[int], span: tuple[int, int]) -> tuple[int, ...]:
    return tuple(i for i, t in enumerate(ids)
                 if t not in SPECIAL_IDS and not (span[0] <= i < span[1]))


def build_pair(tokenizer, token_ids: Sequence[int], template: dict,
               category_token: str = "person") -> CorePair:
    template_id = str(template.get("id", "template"))
    raw = str(template["prompt"]).strip()
    if raw.count("{}") != 1:
        raise ValueError(f"{template_id}: template must contain exactly one '{{}}'")
    object_text = raw.format(" ".join(template["_tokens"]) + " " + category_token)
    reference_text = raw.format(category_token)
    object_ids = _ids(tokenizer, object_text)
    reference_ids = _ids(tokenizer, reference_text)

    start = _find(object_ids, token_ids)
    if start < 0:
        raise ValueError(f"{template_id}: object prompt has no token sequence")
    cat_ids = tokenizer(category_token, add_special_tokens=False).input_ids
    if not cat_ids:
        raise ValueError("category token tokenizes to nothing")
    # The class word must sit immediately after the two learnable tokens.
    cat = _find(object_ids, cat_ids, start + len(token_ids))
    if cat != start + len(token_ids):
        raise ValueError(f"{template_id}: class word is not adjacent to tokens")
    object_span = (start, cat + len(cat_ids))

    ref_cat = _find(reference_ids, cat_ids)
    if ref_cat < 0:
        raise ValueError(f"{template_id}: reference has no category token")
    reference_span = (ref_cat, ref_cat + len(cat_ids))

    object_context = _context_positions(object_ids, object_span)
    reference_context = _context_positions(reference_ids, reference_span)
    obj_ctx_ids = [object_ids[i] for i in object_context]
    ref_ctx_ids = [reference_ids[i] for i in reference_context]
    if obj_ctx_ids != ref_ctx_ids:
        raise ValueError(
            f"{template_id}: context token ids are not aligned\n"
            f"  object={obj_ctx_ids}\n  reference={ref_ctx_ids}\n"
            f"  object_text={object_text!r}\n  reference_text={reference_text!r}"
        )
    text = tuple(tokenizer.convert_ids_to_tokens(i) for i in obj_ctx_ids)
    return CorePair(
        template_id=template_id,
        object_ids=tuple(object_ids),
        reference_ids=tuple(reference_ids),
        object_span=object_span,
        reference_span=reference_span,
        object_context=object_context,
        reference_context=reference_context,
        context_text=text,
    )


def build_pairs(tokenizer, tokens: Sequence[str], token_ids: Sequence[int],
                templates: Sequence[dict], category_token: str = "person") -> list[CorePair]:
    out: list[CorePair] = []
    for t in templates:
        item = dict(t)
        item["_tokens"] = list(tokens)
        out.append(build_pair(tokenizer, token_ids, item, category_token))
    if not out:
        raise ValueError("empty CoRe template list")
    return out


def context_cosines(model, token_rows: torch.Tensor, pair: CorePair) -> torch.Tensor:
    """Per-context-token cosine similarity (object vs reference branch)."""
    emb = model.get_input_embeddings()
    device = emb.weight.device
    dtype = emb.weight.dtype
    obj_ids = torch.tensor(pair.object_ids, device=device)
    ref_ids = torch.tensor(pair.reference_ids, device=device)

    rows = emb(obj_ids)  # differentiable lookup
    start, end = pair.object_span
    token_rows = token_rows.to(device=device, dtype=dtype)
    class_rows = rows[start + len(token_rows):end]
    obj_inputs = torch.cat([rows[:start], token_rows, class_rows, rows[end:]], dim=0)
    obj_out = model(inputs_embeds=obj_inputs.unsqueeze(0))[0][0]

    with torch.no_grad():
        ref_inputs = emb(ref_ids).detach()
        ref_out = model(inputs_embeds=ref_inputs.unsqueeze(0))[0][0]

    obj_ctx = obj_out[list(pair.object_context)]
    ref_ctx = ref_out[list(pair.reference_context)]
    return F.cosine_similarity(obj_ctx, ref_ctx, dim=-1)


def core_inputs_embeds(model, token_rows: torch.Tensor, pair: CorePair):
    """Object/reference inputs_embeds; object branch keeps gradient to token_rows."""
    emb = model.get_input_embeddings()
    device = emb.weight.device
    dtype = emb.weight.dtype
    obj_ids = torch.tensor(pair.object_ids, device=device)
    ref_ids = torch.tensor(pair.reference_ids, device=device)
    rows = emb(obj_ids)
    start, end = pair.object_span
    token_rows = token_rows.to(device=device, dtype=dtype)
    class_rows = rows[start + len(token_rows):end]
    obj_inputs = torch.cat([rows[:start], token_rows, class_rows, rows[end:]], dim=0)
    with torch.no_grad():
        ref_inputs = emb(ref_ids).detach()
    return obj_inputs.unsqueeze(0), ref_inputs.unsqueeze(0)


def attention_means(capture) -> dict[str, torch.Tensor]:
    """Mean cross-attention per token: [B, tokens] for every captured layer."""
    out = {}
    for name, probs in capture.probs.items():
        heads = int(capture.heads[name])
        b = probs.shape[0] // heads
        s, t = probs.shape[1], probs.shape[2]
        out[name] = probs.reshape(b, heads, s, t).mean(dim=(1, 2))
    return out


def car_loss(obj_means: dict[str, torch.Tensor],
             cat_means: dict[str, torch.Tensor],
             pair: CorePair) -> torch.Tensor:
    """CoRe CAR: MSE between context-token attention means, averaged over layers."""
    vals = []
    op = list(pair.object_context)
    rp = list(pair.reference_context)
    for name, om in obj_means.items():
        if name not in cat_means:
            raise KeyError(f"CAR layer {name} missing in category maps")
        vals.append(F.mse_loss(om[:, op], cat_means[name][:, rp]))
    if not vals:
        raise RuntimeError("CAR captured no cross-attention layers")
    return torch.stack(vals).mean()


def postconcept_cosines(model, token_rows: torch.Tensor, pair: CorePair) -> torch.Tensor:
    """Cosine similarities only for context tokens AFTER the concept span.

    Tokens before the concept cannot depend on the identity rows under CLIP's
    causal attention; including them dilutes the diagnostic.
    """
    emb = model.get_input_embeddings()
    device = emb.weight.device
    dtype = emb.weight.dtype
    obj_ids = torch.tensor(pair.object_ids, device=device)
    ref_ids = torch.tensor(pair.reference_ids, device=device)
    rows = emb(obj_ids)
    start, end = pair.object_span
    token_rows = token_rows.to(device=device, dtype=dtype)
    class_rows = rows[start + len(token_rows):end]
    obj_inputs = torch.cat([rows[:start], token_rows, class_rows, rows[end:]], dim=0)
    obj_out = model(inputs_embeds=obj_inputs.unsqueeze(0))[0][0]
    with torch.no_grad():
        ref_out = model(inputs_embeds=emb(ref_ids).detach().unsqueeze(0))[0][0]
    op = [i for i in pair.object_context if i > end]
    rp = [i for i in pair.reference_context if i > pair.reference_span[1]]
    if not op or len(op) != len(rp):
        return torch.empty(0, device=device)
    return F.cosine_similarity(obj_out[op], ref_out[rp], dim=-1)


def cer_loss(model, token_rows: torch.Tensor, pair: CorePair) -> torch.Tensor:
    """1 - mean cosine similarity of aligned context-token outputs.

    The reference branch is detached; gradients reach ``token_rows`` through
    the object branch.
    """
    return (1.0 - context_cosines(model, token_rows, pair)).mean()

