"""CelebAMask-HQ mask utilities (numpy + PIL only, no torch/scipy).

The training mask must follow exactly the same resize / crop / flip as the image.
`prepare.py` stores both at the native mask resolution; `dataset.py` applies the
same transform to both tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Official CelebAMask-HQ class names. Index is the numeric suffix in the PNG name.
CELEBA_MASK_CLASSES = [
    "background",  # 0
    "skin",        # 1
    "nose",        # 2
    "eye_g",       # 3
    "l_eye",       # 4
    "r_eye",       # 5
    "l_brow",      # 6
    "r_brow",      # 7
    "l_ear",       # 8
    "r_ear",       # 9
    "mouth",       # 10
    "u_lip",       # 11
    "l_lip",       # 12
    "hair",        # 13
    "hat",         # 14
    "ear_r",       # 15
    "neck",        # 16
    "neck_l",      # 17
    "cloth",       # 18
]


# C1 (2026-11-17): head region only. Glasses/hat/earring/necklace/cloth are
# NOT part of the person mask; clothing is background for this study.
HEAD_CLASSES = [
    "skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow",
    "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "neck",
]


def foreground_class_names(include_cloth: bool = False) -> list[str]:
    """Classes merged into the person mask (head/hair/neck; no accessories)."""
    names = list(HEAD_CLASSES)
    if include_cloth:  # not used by the locked protocol
        names.append("cloth")
    return names


def _mask_candidates(mask_anno_root: Path, image_id: int, class_name: str) -> list[Path]:
    folder = mask_anno_root / str(image_id // 2000)
    # The official archive zero-pads to 5 digits; some conversions do not.
    names = [
        f"{image_id:05d}_{class_name}.png",
        f"{image_id}_{class_name}.png",
    ]
    return [folder / n for n in names]


def load_celeba_class_mask(
    mask_anno_root: str | Path,
    image_id: int,
    class_name: str,
) -> np.ndarray | None:
    """Return a bool mask for one class, or None if that class file is absent."""
    root = Path(mask_anno_root)
    for path in _mask_candidates(root, image_id, class_name):
        if path.exists():
            arr = np.asarray(Image.open(path).convert("L"))
            return arr > 127
    return None


def merge_celeba_masks(
    mask_anno_root: str | Path,
    image_id: int,
    include_cloth: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Merge per-class PNGs into one binary person mask."""
    masks: list[np.ndarray] = []
    used: list[str] = []
    for name in foreground_class_names(include_cloth=include_cloth):
        m = load_celeba_class_mask(mask_anno_root, image_id, name)
        if m is None:
            continue
        masks.append(m)
        used.append(name)
    if not masks:
        raise FileNotFoundError(
            f"no CelebAMask-HQ class masks found for id={image_id} under {mask_anno_root}. "
            "Expected <root>/<id//2000>/<id:05d>_skin.png etc."
        )
    shape = masks[0].shape
    for m in masks:
        if m.shape != shape:
            raise ValueError(f"class masks for id={image_id} have inconsistent shapes")
    merged = np.zeros(shape, dtype=bool)
    for m in masks:
        merged |= m
    return merged, used


def dilate_binary(mask: np.ndarray, iterations: int = 0) -> np.ndarray:
    """8-neighbourhood binary dilation. O(iterations) and dependency-free."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(iterations)):
        p = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            p[0:-2, 0:-2] | p[0:-2, 1:-1] | p[0:-2, 2:]
            | p[1:-1, 0:-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, 0:-2] | p[2:, 1:-1] | p[2:, 2:]
        )
    return m


def erode_binary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return ~dilate_binary(~np.asarray(mask, dtype=bool), iterations)


def soften_mask(binary: np.ndarray, dilate_px: int = 15, blur_sigma: float = 5.0) -> np.ndarray:
    """Dilate then Gaussian-blur a binary mask to a soft [0,1] target.

    Dilation makes the boundary tolerant to small attention offsets; blur avoids
    a hard step in the loss. Both are applied identically to the correct and the
    shifted (S) mask, so the comparison stays fair.
    """
    m = dilate_binary(binary, iterations=dilate_px)
    img = Image.fromarray((m.astype(np.uint8) * 255))
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    soft = np.asarray(img).astype(np.float32) / 255.0
    return np.clip(soft, 0.0, 1.0)


def area_ratio(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(m.sum()) / float(m.size)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def find_translation(
    mask: np.ndarray,
    max_iou: float = 0.15,
    step: int = 1,
) -> dict | None:
    """Find a pure integer translation (dy, dx) with IoU <= max_iou.

    Area is preserved exactly. The shifted mask must stay fully inside the frame.
    Uses FFT autocorrelation, so the full 512x512 shift space is searched at once.

    Returns None when no valid translation exists. This is the feasibility check
    the plan is missing for 40-55% foreground images: a large connected blob
    often cannot be moved anywhere in-frame without overlapping itself >15%.
    """
    m = np.asarray(mask, dtype=bool)
    area = int(m.sum())
    if area == 0:
        return None
    ys, xs = np.nonzero(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = m.shape

    dy_min, dy_max = -y0, h - y1
    dx_min, dx_max = -x0, w - x1
    if dy_max < dy_min or dx_max < dx_min:
        return None

    mf = m.astype(np.float32)
    spec = np.fft.rfft2(mf)
    corr = np.fft.irfft2(spec * np.conj(spec), s=m.shape)
    # corr[dy % H, dx % W] = sum_{y,x} m[y,x] * m[y-dy, x-dx]
    dys = np.arange(dy_min, dy_max + 1, step, dtype=np.int64)
    dxs = np.arange(dx_min, dx_max + 1, step, dtype=np.int64)
    inter = np.clip(corr[np.ix_(dys % h, dxs % w)], 0.0, None)
    union = 2.0 * area - inter
    iou = inter / np.maximum(union, 1e-9)

    valid = iou <= max_iou
    if not valid.any():
        return None

    # Prefer the most separated placement (lowest IoU), then the longest shift.
    cand_y, cand_x = np.nonzero(valid)
    cand_iou = iou[cand_y, cand_x]
    cand_dist = np.abs(dys[cand_y]) + np.abs(dxs[cand_x])
    # np.lexsort sorts by the LAST key first: primary=iou, then longest shift.
    order = np.lexsort((cand_x, cand_y, -cand_dist, cand_iou))
    best = int(order[0])
    return {
        "dy": int(dys[cand_y[best]]),
        "dx": int(dxs[cand_x[best]]),
        "iou": float(cand_iou[best]),
        "shift_l1": int(cand_dist[best]),
    }


def translate_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate without wrap. Caller must ensure the bbox stays inside."""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m)
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return out
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ty0, ty1 = y0 + dy, y1 + dy
    tx0, tx1 = x0 + dx, x1 + dx
    if ty0 < 0 or tx0 < 0 or ty1 > h or tx1 > w:
        raise ValueError("translation would move the mask outside the frame")
    out[ty0:ty1, tx0:tx1] = m[y0:y1, x0:x1]
    return out


def mask_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    inner = erode_binary(m, iterations=max(1, int(width)))
    return m & ~inner


def overlay_mask_contour(image: Image.Image, mask: np.ndarray, color=(255, 0, 0)) -> Image.Image:
    """QA overlay: draw the mask boundary on the image (no cv2 dependency)."""
    img = image.convert("RGB").copy()
    if mask.shape != (img.height, img.width):
        mask_img = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255))
        mask_img = mask_img.resize(img.size, resample=Image.BILINEAR)
        mask = np.asarray(mask_img) > 127
    edge = mask_boundary(mask, width=2)
    arr = np.asarray(img).copy()
    arr[edge] = color
    return Image.fromarray(arr)


def background_content_score(image: Image.Image, mask: np.ndarray) -> dict:
    """Cheap proxy for "the background has something in it".

    Plain white studio backgrounds are what the paper's CelebA-HQ samples often
    have; with no background content the leakage hypothesis is untestable.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if mask.shape != gray.shape:
        m_img = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255))
        m_img = m_img.resize(image.size, resample=Image.BILINEAR)
        mask = np.asarray(m_img) > 127
    bg = ~np.asarray(mask, dtype=bool)
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    return {
        "bg_std": float(gray[bg].std()) if bg.any() else 0.0,
        "bg_edge_mean": float(edge[bg].mean()) if bg.any() else 0.0,
        "bg_fraction": float(bg.mean()),
    }


def load_soft_mask(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_soft_mask(path: str | Path, mask: np.ndarray) -> None:
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def resize_soft_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    img = img.resize((int(size[0]), int(size[1])), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def summarize_mask(mask: np.ndarray) -> dict:
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return {"area": 0, "ratio": 0.0, "bbox": None}
    return {
        "area": int(m.sum()),
        "ratio": float(m.mean()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
    }
