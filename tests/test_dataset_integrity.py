# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Validation must cover the data the trainers actually read.'
import importlib.util
import json
import logging
import sys
import tarfile
from pathlib import Path

from PIL import Image

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


from utils.dataset import derive_dataset_id  # noqa: E402
from utils.lineage import load_dataset_manifest  # noqa: E402

logging.disable(logging.INFO)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preprocess = _load("src/data/preprocess_cv.py", "preprocess_integrity")


def _publish(tmp_path, *, nested=False, embedded_overrides=None, omit_embedded=False):
    """Build a staging directory of archives plus the external manifest."""
    source = tmp_path / "source"
    splits = {}
    for split in ("train", "validation", "test"):
        preprocess.create_anomalib_dataset(
            source, 8, split, anomaly_ratio=0.25, image_size=32
        )
        root = source / split
        if nested and split == "train":
            for class_name in ("good", "defect"):
                nested_dir = root / "train" / class_name
                nested_dir.mkdir(parents=True)
                for item in (root / class_name).iterdir():
                    (nested_dir / item.name).write_bytes(item.read_bytes())
        splits[split] = root

    manifest = {
        "split_scheme": "generated-disjoint",
        "split_scheme_version": 2,
        "label_mapping": "synthetic",
        "image_size": 32,
        "split_ratios": {"train": 0.666667, "validation": 0.166667,
                         "test": 0.166666},
        "splits": {},
    }
    for split, root in splits.items():
        manifest["splits"][split] = {
            "counts": preprocess.class_counts(root),
            "content_sha256": preprocess.content_hash(root),
        }
    manifest["dataset_id"] = derive_dataset_id(
        {split: entry["content_sha256"]
         for split, entry in manifest["splits"].items()}
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    for split, root in splits.items():
        if not omit_embedded:
            embedded = dict(manifest)
            embedded.update(embedded_overrides or {})
            (root / preprocess.MANIFEST_NAME).write_text(json.dumps(embedded))
        with tarfile.open(staging / f"{split}.tar.gz", "w:gz") as tar:
            tar.add(root, arcname=split)
    return staging, manifest


def test_a_clean_dataset_validates(tmp_path):
    staging, manifest = _publish(tmp_path)
    preprocess._validate_downloaded_dataset(staging, manifest)


def test_nested_split_layout_is_refused(tmp_path):
    """Validation reads train/, the resolver would read train/train/."""
    staging, manifest = _publish(tmp_path, nested=True)
    with pytest.raises(RuntimeError, match="nested layout"):
        preprocess._validate_downloaded_dataset(staging, manifest)


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "SOMETHING-ELSE"),
    ("split_scheme", "content-addressed-sha256"),
    ("split_scheme_version", 99),
    ("label_mapping", "platelets"),
    ("image_size", 256),
])
def test_embedded_manifest_must_match_the_published_one(tmp_path, field, value):
    """The embedded copy is what the trainers read."""
    staging, manifest = _publish(tmp_path, embedded_overrides={field: value})
    with pytest.raises(RuntimeError, match="disagrees with"):
        preprocess._validate_downloaded_dataset(staging, manifest)


def test_a_missing_embedded_manifest_is_refused(tmp_path):
    staging, manifest = _publish(tmp_path, omit_embedded=True)
    with pytest.raises(RuntimeError, match="no dataset_manifest"):
        preprocess._validate_downloaded_dataset(staging, manifest)


_BASE = {
    "dataset_id": "D1",
    "split_scheme": "content-addressed-sha256",
    "split_scheme_version": 1,
    "label_mapping": "platelets",
    "image_size": 256,
}


def _channel(tmp_path, name, manifest):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (directory / "dataset_manifest.json").write_text(
            manifest if isinstance(manifest, str) else json.dumps(manifest)
        )
    return directory


def test_agreeing_channels_resolve(tmp_path):
    a = _channel(tmp_path, "train", _BASE)
    b = _channel(tmp_path, "validation", _BASE)
    assert load_dataset_manifest(a, b)["dataset_id"] == "D1"


def test_conflicting_channels_are_refused(tmp_path):
    """Returning the first match let the walk order pick the model's identity."""
    a = _channel(tmp_path, "train", _BASE)
    b = _channel(tmp_path, "validation", dict(_BASE, dataset_id="D2"))
    with pytest.raises(ValueError, match="disagree about the dataset identity"):
        load_dataset_manifest(a, b)


def test_a_malformed_channel_manifest_is_refused(tmp_path):
    """Ignoring it let the next channel's manifest stand in for the corrupt one."""
    a = _channel(tmp_path, "train", "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_dataset_manifest(a)


def test_no_manifest_anywhere_returns_empty(tmp_path):
    """Absent is distinct from corrupt: the caller decides what absence means."""
    assert load_dataset_manifest(_channel(tmp_path, "train", None)) == {}


def _png_bytes(colour, size=(4, 4)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _ambiguous_pair(tmp_path: Path):
    'Two split layouts that the OLD concatenation hashed identically.'
    a, b, c = (_png_bytes((200, 10, 10)), _png_bytes((10, 200, 10)),
               _png_bytes((10, 10, 200)))

    x = tmp_path / "x" / "good"
    x.mkdir(parents=True)
    (x / "00.png").write_bytes(c)
    (x / "01.png").write_bytes(a + b"02.png" + b)

    y = tmp_path / "y" / "good"
    y.mkdir(parents=True)
    (y / "00.png").write_bytes(c + b"01.png" + a)
    (y / "02.png").write_bytes(b)

    return tmp_path / "x", tmp_path / "y"


def test_the_shared_hash_separates_them(tmp_path):
    """The fix: length-prefixed fields and a fixed-width content digest."""
    from utils.dataset import split_content_hash

    x, y = _ambiguous_pair(tmp_path)
    assert split_content_hash(x) != split_content_hash(y), (
        "different decoded image sets still hash identically, so two datasets can share "
        "one dataset id and a parent's training images can sit in a child's holdout"
    )


def test_moving_a_file_between_classes_changes_the_hash(tmp_path):
    """The per-class file count is part of the record for this reason."""
    from utils.dataset import split_content_hash

    root = tmp_path / "split"
    for class_name in ("good", "defect"):
        (root / class_name).mkdir(parents=True)
    (root / "good" / "a.png").write_bytes(_png_bytes((1, 2, 3)))
    (root / "good" / "b.png").write_bytes(_png_bytes((4, 5, 6)))
    (root / "defect" / "c.png").write_bytes(_png_bytes((7, 8, 9)))
    before = split_content_hash(root)

    (root / "good" / "b.png").rename(root / "defect" / "b.png")
    assert split_content_hash(root) != before
