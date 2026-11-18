"""Shared helpers for the formal matrix."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments" / "disen_formal_v1" / "protocol.json"
PROMPTS_PATH = ROOT / "experiments" / "disen_formal_v1" / "prompts.jsonl"
RUBRIC_PATH = ROOT / "experiments" / "disen_formal_v1" / "human_rubric.json"
MODEL_PATH = ROOT / "models" / "sd2-1-base"
DATA_DIR = ROOT / "data" / "disen_formal"
RAW_DATA = ROOT / "data" / "CelebAMask-HQ"
NAMES_PATH = ROOT / "third_party" / "official" / "examples" / "wiki_names_v2.txt"


def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def prompts() -> list[dict]:
    return read_jsonl(PROMPTS_PATH)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def stream_seed(global_seed: int, image_id: str, stream: str) -> int:
    h = hashlib.sha256(f"{int(global_seed)}:{image_id}:{stream}".encode()).digest()
    return int.from_bytes(h[:4], "little")


_TEMPLATES: list[str] | None = None


def template_list() -> list[str]:
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    path = ROOT / "third_party" / "official" / "textual_inversion_dataset.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "imagenet_templates_small":
                    values = ast.literal_eval(node.value)
                    if len(values) != 81:
                        raise ValueError(f"official template list has {len(values)} entries, expected 81")
                    _TEMPLATES = list(values)
                    return _TEMPLATES
    raise ValueError(f"{path}: imagenet_templates_small not found")


def tokens() -> list[str]:
    return ["<ci0>", "<ci1>"]


def add_tokens(tokenizer, text_encoder) -> list[int]:
    names = tokens()
    tokenizer.add_tokens(names)
    token_ids = [int(tokenizer.convert_tokens_to_ids(t)) for t in names]
    text_encoder.resize_token_embeddings(len(tokenizer))
    return token_ids


def encode_with_token_rows(tokenizer, text_encoder, text: str, token_ids: list[int],
                           rows: torch.Tensor | None, device) -> torch.Tensor:
    """CLIP text features with the two token rows replaced; supports gradients."""
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=int(tokenizer.model_max_length), return_tensors="pt")
    ids = enc.input_ids.to(device)
    emb = text_encoder.get_input_embeddings()(ids)
    if rows is not None:
        pos = (ids[0] == int(token_ids[0]))
        pos1 = (ids[0] == int(token_ids[1]))
        if int(pos.sum()) != 1 or int(pos1.sum()) != 1:
            raise ValueError(f"prompt {text!r} does not contain each token exactly once")
        emb = emb.clone()
        emb[0, pos] = rows[0].to(device=emb.device, dtype=emb.dtype)
        emb[0, pos1] = rows[1].to(device=emb.device, dtype=emb.dtype)
        out = text_encoder(inputs_embeds=emb)[0]
    else:
        out = text_encoder(ids)[0]
    return out


def image_512(path: str | Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize((512, 512), Image.BICUBIC)


def image_to_tensor_minus1(im: Image.Image) -> torch.Tensor:
    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    a = ((x.clamp(-1, 1).float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    return Image.fromarray(a.round().astype(np.uint8))


def identity_map() -> dict[str, dict]:
    data = json.loads((DATA_DIR / "identities.json").read_text(encoding="utf-8"))
    return {str(x["image_id"]): x for x in data["identities"]}


def formal_identities() -> list[dict]:
    data = json.loads((DATA_DIR / "identities.json").read_text(encoding="utf-8"))
    out = [x for x in data["identities"] if x["status"] == "formal"]
    if len(out) != 8:
        raise ValueError(f"expected 8 formal identities, found {len(out)}")
    return out


def all_training_identities() -> list[dict]:
    return list(identity_map().values())


def scene_texts() -> dict[str, str]:
    return {p["id"]: p["attribute_text"] for p in prompts() if p["id"] in {"p05", "p06", "p07", "p08"}}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    # bfloat16 has no NumPy dtype; force float32 before hashing.
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()
