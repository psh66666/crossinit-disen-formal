"""Loss terms.

L = L_denoise + 1e-5 * L_reg_official + lambda_loc * L_loc

L_reg_official is the official implementation, not the paper equation:
  F.pairwise_distance(v, v_init, p=2).mean() * 1e-5
i.e. a NON-squared L2 norm, whereas Eq. 4 writes ||v - v_init||_2^2. The plan
keeps the official code form for B/M/S and discloses the difference in the report.

The selected formulation (A1, 2026-11-17) is:

    q(p) = a(p) / (sum_j a(j) + eps)
    L_loc = mean_l [ 1 - sum_p q_l(p) M_l(p) ]        # `out_frac`

This is collapse-resistant: if the two tokens give up all attention, q is zero
and the loss is 1, worse than a well-localised ~0.5. It is also scale-free across
16x16 and 32x32 grids. `raw`/`patch_mean` are kept for comparison; `neg_in_frac`
is the same gradient as out_frac without the constant; `kl` is an alternative.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _grid_shape(q_len: int) -> tuple[int, int]:
    h = int(round(math.sqrt(q_len)))
    if h * h != q_len:
        raise ValueError(f"query length {q_len} is not a square grid")
    return h, h


def localization_loss(
    probs: dict[str, torch.Tensor],
    heads: dict[str, int],
    token_mask: torch.Tensor,
    masks: torch.Tensor,
    *,
    norm: str = "raw",
    eps: float = 1e-6,
    sample_weights: torch.Tensor | None = None,
    detach_denominator: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """probs: name -> (B*H, Q, L); token_mask: (B, L); masks: (B, 1, H, W)."""
    if not probs:
        raise RuntimeError("localization_loss called without captured attention probabilities")
    if norm not in {"raw", "patch_mean", "out_frac", "neg_in_frac", "kl"}:
        raise ValueError(norm)

    batch = masks.shape[0]
    layer_losses: list[torch.Tensor] = []
    stats_acc = {"in_mass": 0.0, "out_mass": 0.0, "total_mass": 0.0, "leak_ratio": 0.0}

    for name, prob in probs.items():
        h = heads[name]
        b, q, l = prob.shape[0] // h, prob.shape[1], prob.shape[2]
        if b != batch:
            raise ValueError(f"{name}: prob batch {b} vs mask batch {batch}")
        p = prob.float().view(batch, h, q, l)
        tm = token_mask.to(device=p.device, dtype=p.dtype).unsqueeze(1).unsqueeze(2)
        q_mass = (p * tm).sum(dim=-1).mean(dim=1)  # (B, Q)

        gh, gw = _grid_shape(q)
        m = F.interpolate(masks.float(), size=(gh, gw), mode="area").flatten(1)  # (B, Q)
        in_mass = (q_mass * m).sum(dim=-1)
        out_mass = (q_mass * (1.0 - m)).sum(dim=-1)
        total = in_mass + out_mass
        denom = total.detach() if detach_denominator else total

        if norm == "raw":
            per = out_mass
        elif norm == "patch_mean":
            per = out_mass / float(q)
        elif norm == "out_frac":
            # A1 formula: normalise the token attention over space, then
            # L = 1 - in_mask_fraction. A zero-mass token gives 1, not 0.
            qn = q_mass / (denom.unsqueeze(-1) + eps)
            per = 1.0 - (qn * m).sum(dim=-1)
        elif norm == "neg_in_frac":
            per = -in_mass / (denom + eps)
        else:  # kl: target distribution is the mask, distribution is normalised q
            target = m / (m.sum(dim=-1, keepdim=True) + eps)
            qn = q_mass / (total.unsqueeze(-1) + eps)
            per = (target * (torch.log(target + eps) - torch.log(qn + eps))).sum(dim=-1)

        if sample_weights is not None:
            w = sample_weights.to(device=per.device, dtype=per.dtype)
            layer_loss = (per * w).sum() / w.sum().clamp_min(1.0)
        else:
            layer_loss = per.mean()
        layer_losses.append(layer_loss)

        stats_acc["in_mass"] += float(in_mass.mean().detach())
        stats_acc["out_mass"] += float(out_mass.mean().detach())
        stats_acc["total_mass"] += float(total.mean().detach())
        stats_acc["leak_ratio"] += float((out_mass / (total + eps)).mean().detach())
        stats_acc.setdefault("in_frac", 0.0)
        stats_acc["in_frac"] += float((in_mass / (total + eps)).mean().detach())

    n = float(len(layer_losses))
    loss = torch.stack(layer_losses).mean()
    stats = {k: v / n for k, v in stats_acc.items()}
    stats["loc"] = float(loss.detach())
    return loss, stats


def reg_loss(
    token_rows: torch.Tensor,
    init_rows: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, float]:
    """Official regulariser: mean non-squared L2 distance, then multiply by lambda."""
    dist = F.pairwise_distance(token_rows, init_rows.to(token_rows), p=2).mean()
    return dist * float(weight), float(dist.detach())


def denoise_mse(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE with an optional per-pixel weight.

    ``weight`` is a [B,1,H,W] map interpolated to the latent grid. The weighted
    mean is normalized by ``sum(weight)`` so the loss scale stays comparable to
    the unweighted MSE when the mean weight is not 1.
    """
    diff = (model_pred.float() - target.float()).pow(2)
    if weight is None:
        return diff.mean()
    w = weight.float()
    if w.shape[-2:] != diff.shape[-2:]:
        w = F.interpolate(w, size=diff.shape[-2:], mode="area")
    w = w.expand_as(diff)
    return (diff * w).sum() / w.sum().clamp_min(1e-6)
