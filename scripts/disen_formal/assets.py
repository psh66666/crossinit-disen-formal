#!/usr/bin/env python
"""Register frozen assets and write the pre-training freeze lock.

The server project is independent of old runs.  Large models and the raw
CelebAMask-HQ tree are reused through symlinks, never copied.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_state import canonical_hash, sha256_file, write_json

MODEL_FILE_PATTERNS = ("model_index.json", "unet/diffusion_pytorch_model.safetensors",
                       "unet/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.safetensors",
                       "vae/diffusion_pytorch_model.bin")
CODE_DIRS = ("crossinit", "scripts/disen_formal")
CODE_FILES = ("train.py", "generate.py", "requirements.txt",
              "experiments/disen_formal_v1/protocol.json",
              "experiments/disen_formal_v1/prompts.jsonl",
              "experiments/disen_formal_v1/human_rubric.json")


def load_protocol(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_model_dirs(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("model_index.json"):
        if any(part in {"runs", "data_raw", "celeba_meta"} for part in p.parts):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if obj.get("_class_name") != "StableDiffusionPipeline":
            continue
        if not (p.parent / "unet").is_dir() or not (p.parent / "vae").is_dir():
            continue
        out.append(p.parent)
    return sorted(set(out))


def _find_data_dir(root: Path) -> Path:
    for p in root.rglob("CelebA-HQ-img"):
        if p.is_dir() and (p.parent / "CelebA-HQ-to-CelebA-mapping.txt").exists():
            return p.parent
    raise SystemExit("CelebAMask-HQ not found: no CelebA-HQ-img + mapping pair")


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        raise SystemExit(f"{link} already exists and does not point to {target}")
    os.symlink(target, link)


def _model_hashes(model: Path) -> dict[str, str]:
    out = {}
    for rel in MODEL_FILE_PATTERNS:
        p = model / rel
        if p.exists() and p.is_file():
            out[rel] = sha256_file(p)
    if not out:
        raise SystemExit(f"{model}: no recognised weight files")
    return out


def _history_model_binding(project: Path) -> tuple[str | None, list[str]]:
    """Read the old project's recorded model source, if present."""
    found, sources = None, []
    for p in project.rglob("run_config.json"):
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            src = cfg.get("model_source")
        except (OSError, json.JSONDecodeError):
            continue
        if src:
            sources.append(str(src))
            found = str(src)
    return found, sorted(set(sources))


def register(args) -> None:
    protocol = load_protocol(args.protocol)
    root = Path(args.discover_under)
    candidates = _candidate_model_dirs(root)
    if not candidates:
        raise SystemExit(f"no SD2.1 pipeline under {root}")
    history_src, all_sources = _history_model_binding(ROOT)
    chosen = None
    if history_src:
        for c in candidates:
            try:
                if c.resolve() == Path(history_src).resolve():
                    chosen = c
                    break
            except OSError:
                pass
    if chosen is None and len(candidates) == 1:
        chosen = candidates[0]
    if chosen is None:
        old = [s for s in all_sources if Path(s).exists()]
        for c in candidates:
            if any(Path(s).resolve() == c.resolve() for s in old):
                chosen = c
                break
    if chosen is None:
        raise SystemExit(
            "multiple SD2.1 candidates and none match a historical model_source; "
            f"candidates={[str(c) for c in candidates]} history={all_sources}"
        )

    data_dir = _find_data_dir(root)
    project_data = ROOT / "data" / "CelebAMask-HQ"
    _link(data_dir, project_data)
    model_link = ROOT / "models" / "sd2-1-base"
    _link(chosen, model_link)
    eval_clip = None
    for cand in (root / "models" / "clip-vit-large-patch14", chosen.parent / "clip-vit-large-patch14"):
        if (cand / "config.json").exists():
            eval_clip = cand
            break
    if eval_clip is not None:
        _link(eval_clip, ROOT / "data" / "eval" / "clip-vit-large-patch14")
    mapping = data_dir / "CelebA-HQ-to-CelebA-mapping.txt"
    identity = ROOT / "data" / "annotations" / "identity_CelebA.txt"
    if not identity.exists():
        raise SystemExit(f"missing {identity}; build it from CelebA identity annotations first")
    # data_dir itself is linked to data/CelebAMask-HQ, so the mapping is visible
    # through the link without a second copy.

    model_hashes = _model_hashes(chosen)
    data_files = {
        "mapping": str(mapping), "mapping_sha256": sha256_file(mapping),
        "identity": str(identity), "identity_sha256": sha256_file(identity),
        "image_count": len(list((data_dir / "CelebA-HQ-img").glob("*.jpg"))),
        "pose": str(data_dir / "CelebAMask-HQ-pose-anno.txt"),
        "pose_sha256": sha256_file(data_dir / "CelebAMask-HQ-pose-anno.txt"),
        "attributes": str(data_dir / "CelebAMask-HQ-attribute-anno.txt"),
        "attributes_sha256": sha256_file(data_dir / "CelebAMask-HQ-attribute-anno.txt"),
    }
    assets = {
        "protocol_id": protocol["protocol_id"],
        "model": {
            "path": str(model_link), "resolved": str(chosen), "hashes": model_hashes,
            "model_index_sha256": model_hashes["model_index.json"],
        },
        "data": data_files,
        "history_model_source": history_src,
        "history_model_sources": all_sources,
        "hardware": hardware_info(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_info(),
    }
    write_json(args.out, assets)
    print(json.dumps({"model": str(chosen), "data": str(data_dir),
                      "assets_hash": canonical_hash(assets)}, indent=2))


def gpu_info() -> list[dict]:
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append({"index": i, "name": p.name, "total_memory_bytes": p.total_memory})
    return out


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10)
        info["nvidia_smi"] = out.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = []
    return info


def code_hashes() -> dict[str, str]:
    out = {}
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            out[str(p.relative_to(ROOT))] = sha256_file(p)
    for rel in CODE_FILES:
        p = ROOT / rel
        if p.exists():
            out[rel] = sha256_file(p)
    return out


def freeze(args) -> None:
    protocol = load_protocol(args.protocol)
    identities = json.loads(Path(args.identities).read_text(encoding="utf-8"))
    assets = json.loads(Path(args.assets).read_text(encoding="utf-8"))
    technical = json.loads(Path(args.engineering).read_text(encoding="utf-8"))
    if technical.get("status") != "PASS":
        raise SystemExit(f"engineering check is not PASS: {technical.get('status')}")

    from scripts.disen_formal.common import template_list
    prompts = [json.loads(x) for x in Path("experiments/disen_formal_v1/prompts.jsonl")
               .read_text(encoding="utf-8").splitlines() if x.strip()]
    rubric = json.loads(Path("experiments/disen_formal_v1/human_rubric.json")
                        .read_text(encoding="utf-8"))
    init_path = ROOT / "data" / "disen_formal" / "celeb_init_691.pt"
    if not init_path.exists():
        raise SystemExit(f"missing {init_path}; run cache prepare first")
    lock = {
        "protocol": protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "assets": assets,
        "assets_hash": canonical_hash(assets),
        "identities": identities,
        "identities_sha256": sha256_file(args.identities),
        "prompts": prompts,
        "prompts_sha256": sha256_file("experiments/disen_formal_v1/prompts.jsonl"),
        "rubric": rubric,
        "rubric_sha256": sha256_file("experiments/disen_formal_v1/human_rubric.json"),
        "templates_81": template_list(),
        "init": {"path": str(init_path), "sha256": sha256_file(init_path)},
        "code": code_hashes(),
        "technical_check": technical,
        "expected": protocol["expected_counts"],
    }
    lock["freeze_hash"] = canonical_hash(lock)
    write_json(args.out, lock)
    print(json.dumps({"freeze": str(args.out), "freeze_hash": lock["freeze_hash"]}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    r.add_argument("--discover-under", default="/root/autodl-tmp")
    r.add_argument("--out", default="locks/disen_formal_v1/assets.json")
    r.set_defaults(func=register)
    f = sub.add_parser("freeze")
    f.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    f.add_argument("--assets", default="locks/disen_formal_v1/assets.json")
    f.add_argument("--identities", default="data/disen_formal/identities.json")
    f.add_argument("--engineering", default="results/disen_formal_v1/technical_check.json")
    f.add_argument("--out", default="locks/disen_formal_v1/freeze.json")
    f.set_defaults(func=freeze)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
