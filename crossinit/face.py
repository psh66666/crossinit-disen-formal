"""InsightFace (ArcFace) wrapper with a process-level cache.

Used for: candidate filtering (single detectable face), identity similarity,
and two-person detection counts. InsightFace is a research-only dependency; it
is not needed for training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_APPS: dict[tuple[str, tuple[str, ...]], object] = {}


def _providers(explicit=None) -> tuple[str, ...]:
    if explicit:
        return tuple(explicit)
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except ImportError:
        avail = set()
    out = []
    if "CUDAExecutionProvider" in avail:
        out.append("CUDAExecutionProvider")
    out.append("CPUExecutionProvider")
    return tuple(out)


def _limit_onnx_threads() -> None:
    """Avoid thousands of ORT threads on shared CPU containers."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if getattr(ort.InferenceSession, "_cpa_mix_thread_limited", False):
        return
    original = ort.InferenceSession.__init__

    def limited(self, *args, **kwargs):
        opts = kwargs.get("sess_options")
        if opts is None:
            opts = ort.SessionOptions()
            kwargs["sess_options"] = opts
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        return original(self, *args, **kwargs)

    ort.InferenceSession.__init__ = limited
    ort.InferenceSession._cpa_mix_thread_limited = True


def get_app(model_name: str = "buffalo_l", providers=None, det_thresh: float = 0.2,
            allowed_modules=None):
    _limit_onnx_threads()
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "insightface is not installed. Remote install: pip install insightface onnxruntime-gpu"
        ) from exc
    modules = tuple(allowed_modules) if allowed_modules else None
    key = (model_name, _providers(providers), float(det_thresh), modules)
    if key not in _APPS:
        app = FaceAnalysis(
            name=model_name, providers=list(key[1]), det_thresh=float(det_thresh),
            allowed_modules=list(modules) if modules else None,
        )
        app.prepare(ctx_id=0 if "CUDAExecutionProvider" in key[1] else -1, det_size=(640, 640))
        # FaceAnalysis.__init__(det_thresh=...) is not reliably forwarded to the
        # SCRFD detector in insightface 0.7.3; set it explicitly.
        if hasattr(app, "det_model"):
            app.det_model.det_thresh = float(det_thresh)
        _APPS[key] = app
    return _APPS[key]


def detect_faces(
    image: Image.Image | str | Path,
    model_name: str = "buffalo_l",
    providers=None,
    det_thresh: float = 0.2,
    allowed_modules=None,
) -> list[dict]:
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    arr = np.asarray(image.convert("RGB"))
    faces = get_app(model_name, providers, det_thresh=det_thresh,
                    allowed_modules=allowed_modules).get(arr[:, :, ::-1])
    out = []
    for f in faces:
        raw = getattr(f, "normed_embedding", None)
        if raw is None:
            emb = np.zeros(512, dtype=np.float32)
        else:
            emb = np.asarray(raw, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-12)
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        out.append({
            "emb": emb,
            "det_score": float(f.det_score),
            "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
            "bbox": [x1, y1, x2, y2],
        })
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
