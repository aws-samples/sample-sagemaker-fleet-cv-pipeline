# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Dataset resolution and image loading helpers."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _has_images(folder: Path) -> bool:
    """True if *folder* exists and holds at least one image."""
    return folder.is_dir() and any(
        p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir() if p.is_file()
    )


def _is_split_root(candidate: Path, require_defect: bool) -> bool:
    'True if *candidate* looks like an anomalib split root with real content.'
    if not _has_images(candidate / "good"):
        return False
    return _has_images(candidate / "defect") if require_defect else True


KNOWN_SPLITS = ("train", "validation", "test")


def resolve_dataset_root(base_dir: Path, split_name: str = "test", require_defect: bool = True) -> Path:
    'Resolve the anomalib split root under *base_dir*.'
    other_splits = {s for s in KNOWN_SPLITS if s != split_name}

    if base_dir.name in other_splits:
        raise ValueError(
            f"Refusing to resolve split={split_name} under {base_dir}: the base "
            f"directory is named for the {base_dir.name!r} split, so anything "
            f"inside it belongs to that split. Pass the dataset root, or the "
            f"{split_name!r} directory itself."
        )

    # The explicitly named split is always acceptable.
    named = base_dir / split_name
    if _is_split_root(named, require_defect):
        return named

    if base_dir.name not in other_splits and _is_split_root(base_dir, require_defect):
        return base_dir

    def _under_a_different_split(candidate: Path) -> bool:
        'True if ANY path component between base_dir and candidate names another split.'
        try:
            relative = candidate.relative_to(base_dir)
        except ValueError:
            return False
        return any(part in other_splits for part in relative.parts)

    matches = sorted(
        {
            good_dir.parent
            for good_dir in base_dir.rglob("good")
            if _is_split_root(good_dir.parent, require_defect)
            and good_dir.parent.name not in other_splits
            and not _under_a_different_split(good_dir.parent)
        }
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous dataset root under {base_dir} for split={split_name}: "
            f"found {len(matches)} candidates ({', '.join(str(m) for m in matches)}). "
            f"Point at the intended split directory explicitly -- selecting one "
            f"arbitrarily could evaluate the wrong split."
        )

    # Report whether a differently-named split was rejected, because "no test
    # data" and "only train data is here" need different corrective actions.
    rejected = sorted(
        good_dir.parent.name
        for good_dir in base_dir.rglob("good")
        if good_dir.parent.name in other_splits
    )
    hint = (
        f" Found directories for other split(s) ({', '.join(sorted(set(rejected)))}) "
        f"but refused to use them: returning another split's data here would "
        f"evaluate the model on data it may have been trained on."
        if rejected else ""
    )
    raise ValueError(
        f"Could not resolve dataset root under {base_dir} for split={split_name}: "
        f"expected a directory containing good/"
        f"{' and defect/' if require_defect else ''} with at least one image.{hint}"
    )


def list_images(folder: Path) -> list:
    """Return sorted image paths in *folder* (deterministic order shared with ``load_images``)."""
    return [p for p in sorted(folder.glob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]


# Domain tag for the split content hash. Bumping it changes every dataset id, which is
# why it is versioned rather than implicit.
_SPLIT_HASH_DOMAIN = b"fleet-cv/split-content/v2\x00"


def split_content_hash(split_dir: Path, class_names=("defect", "good")) -> str:
    "SHA-256 over a split's class names, filenames and image content, UNAMBIGUOUSLY."
    digest = hashlib.sha256()
    digest.update(_SPLIT_HASH_DOMAIN)
    for class_name in sorted(class_names):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        encoded = class_name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(files).to_bytes(4, "big"))
        for path in files:
            name = path.name.encode()
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            # Fixed 32 bytes. This is what removes the ambiguity: variable-length content
            # can no longer extend into the next record.
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def decode_rgb(path: Path):
    'Open *path* and decode it to RGB, the one operation the model input starts from.'
    return Image.open(path).convert("RGB")


def load_image(path: Path, image_size: int) -> np.ndarray:
    """Load a single image, resize, normalise to [0,1] and return as CHW float32 array."""
    arr = np.asarray(decode_rgb(path).resize((image_size, image_size)), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


# Domain tag for the canonical image digest. Included in the hash so a future change
# to the canonical form cannot silently produce the same digest for a different
# definition of "the same image".
_IMAGE_DIGEST_DOMAIN = b"fleet-cv/canonical-rgb/v1\x00"


def canonical_image_digest(path: Path) -> str:
    'SHA-256 of what the MODEL sees, not of the file that carries it.'
    image = decode_rgb(path)
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(_IMAGE_DIGEST_DOMAIN)
    digest.update(width.to_bytes(4, "big"))
    digest.update(height.to_bytes(4, "big"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def load_images(folder: Path, image_size: int) -> np.ndarray:
    """Load all images from *folder*, resize, normalise to [0,1] and return as NCHW float32 array."""
    items = [load_image(p, image_size) for p in list_images(folder)]
    if not items:
        raise ValueError(f"No images found in {folder}")
    return np.stack(items).astype(np.float32)


def assert_splits_disjoint(split_dirs: dict, class_names=("good", "defect")) -> None:
    'Raise if any image CONTENT appears in more than one split.'
    seen = {}
    collisions = []
    for split_name in sorted(split_dirs):
        root = Path(split_dirs[split_name])
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                key = canonical_image_digest(path)
                if key in seen:
                    prior_split, prior_name = seen[key]
                    if prior_split != split_name:
                        collisions.append(
                            f"  {prior_split}/{prior_name} == {split_name}/{path.name}"
                        )
                else:
                    seen[key] = (split_name, path.name)
    if collisions:
        raise RuntimeError(
            f"Splits share image content ({len(collisions)} collision(s)); refusing "
            f"to publish a dataset whose splits overlap, because every quality "
            f"metric computed on it would be meaningless:\n"
            + "\n".join(collisions[:20])
            + ("\n  ..." if len(collisions) > 20 else "")
        )

DATASET_ID_LENGTH = 32


def derive_dataset_id(split_content_hashes: dict) -> str:
    'The dataset id implied by these per-split content hashes.'
    import hashlib
    import json as _json

    canonical = _json.dumps(dict(sorted(split_content_hashes.items())),
                            sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:DATASET_ID_LENGTH]


# The split scheme's name and the VERSIONS this module can assign under. A version is
# not a label: it selects which digest the assignment is derived from, so reading an old
# manifest under a new version would reinterpret its splits.
SPLIT_SCHEME = "content-addressed-sha256"
SPLIT_SCHEME_V1_ENCODED_BYTES = 1
SPLIT_SCHEME_V2_DECODED_RGB = 2
SUPPORTED_SPLIT_SCHEME_VERSIONS = (
    SPLIT_SCHEME_V1_ENCODED_BYTES,
    SPLIT_SCHEME_V2_DECODED_RGB,
)


def assign_split(image_path: Path, train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 scheme_version: int = SPLIT_SCHEME_V2_DECODED_RGB) -> str:
    'Return the split an image belongs to, derived from ITS OWN CONTENT.'
    if scheme_version not in SUPPORTED_SPLIT_SCHEME_VERSIONS:
        raise ValueError(
            f"Unsupported split scheme version {scheme_version!r}; this module can "
            f"assign under {list(SUPPORTED_SPLIT_SCHEME_VERSIONS)}. Refusing to guess: "
            f"assigning under the wrong version silently reinterprets which images are "
            f"held out."
        )
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio} val={val_ratio} "
            f"(both must be in (0,1) and sum to less than 1)"
        )
    if scheme_version == SPLIT_SCHEME_V1_ENCODED_BYTES:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    else:
        digest = canonical_image_digest(image_path)
    # 8 hex chars is 32 bits of the digest; ample resolution for a ratio split.
    position = int(digest[:8], 16) / 0xFFFFFFFF
    if position < train_ratio:
        return "train"
    if position < train_ratio + val_ratio:
        return "validation"
    return "test"
