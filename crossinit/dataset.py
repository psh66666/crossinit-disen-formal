"""Single-image dataset with a synchronized soft person mask.

Mirrors the official TextualInversionDataset (resize -> optional center crop ->
horizontal flip, random ImageNet-style template, prompt = "face of {tokens} person")
and adds exactly the same geometric transform for the mask.

The official script uses one image and `repeats=100` with batch size 8, i.e. each
update sees 8 augmented copies of the SAME image, not 8 different identities.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Verbatim from the official textual_inversion_dataset.py (MIT).
IMAGENET_TEMPLATES_SMALL = (
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
    "an illustration of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "an illustration of a clean {}",
    "an illustration of a dirty {}",
    "a dark photo of the {}",
    "an illustration of my {}",
    "an illustration of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "an illustration of the {}",
    "a good photo of the {}",
    "an illustration of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "an illustration of the clean {}",
    "a rendition of a {}",
    "an illustration of a nice {}",
    "a good photo of a {}",
    "an illustration of the nice {}",
    "an illustration of the small {}",
    "an illustration of the weird {}",
    "an illustration of the large {}",
    "an illustration of a cool {}",
    "an illustration of a small {}",
    "a depiction of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a depiction of a clean {}",
    "a depiction of a dirty {}",
    "a dark photo of the {}",
    "a depiction of my {}",
    "a depiction of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a depiction of the {}",
    "a good photo of the {}",
    "a depiction of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a depiction of the clean {}",
    "a rendition of a {}",
    "a depiction of a nice {}",
    "a good photo of a {}",
    "a depiction of the nice {}",
    "a depiction of the small {}",
    "a depiction of the weird {}",
    "a depiction of the large {}",
    "a depiction of a cool {}",
    "a depiction of a small {}",
)


def _load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img


def _load_soft(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0


class TextualInversionMaskDataset(Dataset):
    """Expects `data_root/image.png`, `mask.png`, optional `mask_shift.png`."""

    def __init__(
        self,
        data_root: str | Path,
        tokenizer,
        placeholder_tokens: list[str],
        size: int = 512,
        repeats: int = 100,
        flip_p: float = 0.5,
        center_crop: bool = False,
        interpolation: str = "bicubic",
        templates: tuple[str, ...] = IMAGENET_TEMPLATES_SMALL,
        text_template: str = "face of {tokens} person",
        require_shifted: bool = False,
        full_templates: bool = False,
    ):
        self.root = Path(data_root)
        self.image_path = self.root / "image.png"
        self.mask_path = self.root / "mask.png"
        self.shift_path = self.root / "mask_shift.png"
        if not self.image_path.exists():
            raise FileNotFoundError(f"missing {self.image_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"missing {self.mask_path}")
        if require_shifted and not self.shift_path.exists():
            raise FileNotFoundError(
                f"mask_mode=shifted but {self.shift_path} is missing; "
                "run data/prepare.py for this identity or check shift feasibility"
            )

        self.tokenizer = tokenizer
        self.placeholder_tokens = list(placeholder_tokens)
        self.size = int(size)
        self.repeats = int(repeats)
        self.flip_p = float(flip_p)
        self.center_crop = bool(center_crop)
        self.templates = tuple(templates)
        self.text_template = text_template
        self.require_shifted = require_shifted
        self.full_templates = bool(full_templates)
        self._length = max(1, self.repeats)

        resample = {
            "linear": Image.BILINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
            "nearest": Image.NEAREST,
        }[interpolation]
        self.resample = resample
        # Load once: one identity == one source image in this project.
        self.image = _load_rgb(self.image_path)
        self.mask = _load_soft(self.mask_path)
        self.shift = _load_soft(self.shift_path) if self.shift_path.exists() else None

        if self.mask.shape != (self.image.height, self.image.width):
            # Masks are 512x512 in CelebAMask-HQ while CelebA-HQ images can be 1024.
            # Resize the mask to the image grid once; the per-item resize below then
            # applies the same normalized geometry to both.
            m_img = Image.fromarray((self.mask * 255.0 + 0.5).astype(np.uint8))
            m_img = m_img.resize(self.image.size, resample=Image.BILINEAR)
            self.mask = np.asarray(m_img).astype(np.float32) / 255.0
        if self.shift is not None and self.shift.shape != self.mask.shape:
            s_img = Image.fromarray((self.shift * 255.0 + 0.5).astype(np.uint8))
            s_img = s_img.resize((self.mask.shape[1], self.mask.shape[0]), resample=Image.BILINEAR)
            self.shift = np.asarray(s_img).astype(np.float32) / 255.0

    def __len__(self) -> int:
        return self._length

    def _crop_box(self, w: int, h: int) -> tuple[int, int, int, int] | None:
        if not self.center_crop:
            return None
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        return left, top, left + side, top + side

    def __getitem__(self, index: int) -> dict:
        tokens = " ".join(self.placeholder_tokens)
        template = random.choice(self.templates)
        text = template.format(tokens=tokens) if self.full_templates else template.format(
            self.text_template.format(tokens=tokens)
        )
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0]

        img = np.asarray(self.image).astype(np.uint8)
        mask = self.mask
        shift = self.shift
        box = self._crop_box(img.shape[1], img.shape[0])
        if box is not None:
            x0, y0, x1, y1 = box
            img = img[y0:y1, x0:x1]
            # Map the image crop onto the mask grid (sizes can differ by 2x).
            mh, mw = mask.shape
            sx, sy = mw / self.image.width, mh / self.image.height
            mx0, my0 = int(round(x0 * sx)), int(round(y0 * sy))
            mx1, my1 = int(round(x1 * sx)), int(round(y1 * sy))
            mask = mask[my0:my1, mx0:mx1]
            if shift is not None:
                shift = shift[my0:my1, mx0:mx1]

        image = Image.fromarray(img).resize((self.size, self.size), resample=self.resample)
        mask_img = Image.fromarray((mask * 255.0 + 0.5).astype(np.uint8)).resize(
            (self.size, self.size), resample=Image.BILINEAR
        )
        mask = np.asarray(mask_img).astype(np.float32) / 255.0
        if shift is not None:
            shift_img = Image.fromarray((shift * 255.0 + 0.5).astype(np.uint8)).resize(
                (self.size, self.size), resample=Image.BILINEAR
            )
            shift = np.asarray(shift_img).astype(np.float32) / 255.0

        # One RNG draw, applied to image and both masks, exactly like the official
        # torchvision RandomHorizontalFlip (which also consumes torch.rand(1)).
        do_flip = torch.rand(1).item() < self.flip_p
        img_np = np.asarray(image).astype(np.uint8)
        if do_flip:
            img_np = img_np[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
            if shift is not None:
                shift = shift[:, ::-1].copy()

        img_np = (img_np / 127.5 - 1.0).astype(np.float32)
        out = {
            "input_ids": input_ids,
            "attention_mask": encoded.attention_mask[0],
            "pixel_values": torch.from_numpy(img_np).permute(2, 0, 1),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if shift is not None:
            out["mask_shift"] = torch.from_numpy(shift).unsqueeze(0)
        return out
