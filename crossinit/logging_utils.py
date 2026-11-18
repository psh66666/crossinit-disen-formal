"""Small helpers: seeding, JSONL run logs, GPU memory, atomic saves."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The official setup_seed also forces deterministic cudnn. Keep it: all
    # groups and all identities must see the same augmentation order.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("time", datetime.now().isoformat(timespec="seconds"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return {
        "gpu": props.name,
        "vram_total_gib": round(total / 2**30, 2),
        "vram_free_gib": round(free / 2**30, 2),
        "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated(idx) / 2**30, 3),
        "max_memory_reserved_gib": round(torch.cuda.max_memory_reserved(idx) / 2**30, 3),
    }


def atomic_torch_save(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
