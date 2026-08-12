# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Dataset resolution and image loading helpers."""
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def resolve_dataset_root(base_dir: Path, split_name: str = "test", require_defect: bool = True) -> Path:
    """Walk *base_dir* to find the anomalib split root containing ``good/`` (and optionally ``defect/``)."""
    direct_candidates = [base_dir / split_name, base_dir]
    for candidate in direct_candidates:
        if (candidate / "good").exists() and (not require_defect or (candidate / "defect").exists()):
            return candidate

    for good_dir in base_dir.rglob("good"):
        parent = good_dir.parent
        if require_defect and not (parent / "defect").exists():
            continue
        return parent

    raise ValueError(f"Could not resolve dataset root under {base_dir} for split={split_name}")


def list_images(folder: Path) -> list:
    """Return sorted image paths in *folder* (deterministic order shared with ``load_images``)."""
    return [p for p in sorted(folder.glob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Load a single image, resize, normalise to [0,1] and return as CHW float32 array."""
    arr = np.asarray(Image.open(path).convert("RGB").resize((image_size, image_size)), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_images(folder: Path, image_size: int) -> np.ndarray:
    """Load all images from *folder*, resize, normalise to [0,1] and return as NCHW float32 array."""
    items = [load_image(p, image_size) for p in list_images(folder)]
    if not items:
        raise ValueError(f"No images found in {folder}")
    return np.stack(items).astype(np.float32)
