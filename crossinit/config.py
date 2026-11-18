"""YAML config loading with one-level `_base_` inheritance and dotted overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    parent = cfg.pop("_base_", None)
    if parent:
        # Relative to the config that declares it, so group configs stay relocatable.
        base_cfg = load_config((path.parent / parent).resolve())
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_dotted_overrides(items: Iterable[str] | None) -> dict:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def resolve_identities(cfg: dict, split: str = "formal") -> list[str]:
    """Config list first; otherwise read data/identities.json written by selection."""
    ids = get_dotted(cfg, "data.identities", []) or []
    if ids:
        return [str(x) for x in ids]
    path = Path(get_dotted(cfg, "data.identities_file", "data/identities.json"))
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            return [str(x) for x in obj.get(split, [])]
        except (json.JSONDecodeError, AttributeError):
            return []
    return []


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


REQUIRED_KEYS = (
    "run.name",
    "model.pretrained_model_name_or_path",
    "data.root",
    "train.n_tokens",
    "train.max_train_steps",
    "loc.norm",
)


def validate_config(cfg: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if get_dotted(cfg, k) is None]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")

    norm = get_dotted(cfg, "loc.norm")
    if norm not in {"raw", "patch_mean", "out_frac", "neg_in_frac", "kl"}:
        raise ValueError(f"loc.norm={norm!r} is not one of raw/patch_mean/out_frac/neg_in_frac/kl")

    mask_mode = get_dotted(cfg, "loc.mask_mode", "correct")
    if mask_mode not in {"correct", "shifted", "none"}:
        raise ValueError(f"loc.mask_mode={mask_mode!r} is not one of correct/shifted/none")
