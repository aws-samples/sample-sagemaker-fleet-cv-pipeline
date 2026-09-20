# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Tests for split stability across dataset versions and warm-start compatibility.'
import importlib.util
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.lineage import TRAINING_MANIFEST_NAME, find_parent_manifest, warm_start_allowed, write_training_manifest


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bccd = _load("scripts/download_bccd.py", "download_bccd")


def _images(tmp_path: Path, count: int, start: int = 0):
    'Distinct, REAL images with deterministic content.'
    out = []
    for i in range(start, start + count):
        p = tmp_path / f"img_{i:04d}.png"
        image = Image.new("RGB", (8, 8), ((i * 7) % 256, (i * 13) % 256, (i * 29) % 256))
        image.putpixel((0, 0), (i % 256, (i // 256) % 256, 0))
        image.save(p, format="PNG")
        out.append((p, "good" if i % 3 else "defect"))
    return out


def test_assignment_depends_only_on_the_image(tmp_path):
    """The same image must get the same split regardless of its neighbours."""
    small = _images(tmp_path, 40)
    large = small + _images(tmp_path, 60, start=40)

    a = {p: s for s, items in bccd.content_addressed_split(small).items()
         for p, _ in items}
    b = {p: s for s, items in bccd.content_addressed_split(large).items()
         for p, _ in items}
    moved = [p for p in a if a[p] != b[p]]
    assert moved == [], f"{len(moved)} image(s) changed split when the set grew"


def test_no_image_migrates_from_train_to_test_across_versions(tmp_path):
    'The exact contamination path, asserted directly.'
    v1 = _images(tmp_path, 200)
    v2 = v1 + _images(tmp_path, 60, start=200)

    old = bccd.content_addressed_split(v1)
    new = bccd.content_addressed_split(v2)
    old_train = {p for p, _ in old["train"]}
    new_test = {p for p, _ in new["test"]}
    assert old_train & new_test == set(), (
        f"{len(old_train & new_test)} of the new version's test images were in the "
        f"old version's training split"
    )


CURRENT = {
    "dataset_id": "abc123",
    "split_scheme": "content-addressed-sha256",
    "split_scheme_version": 2,
    "label_mapping": "platelets -> defect",
    "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
}


def _parent(**overrides):
    base = {
        "split_scheme": "content-addressed-sha256",
        "split_scheme_version": 2,
        "label_mapping": "platelets -> defect",
        "image_size": 256,
        "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
    }
    base.update(overrides)
    return base


def test_matching_contract_allows_warm_start():
    allowed, reason = warm_start_allowed(CURRENT, _parent(), image_size=256)
    assert allowed, reason


@pytest.mark.parametrize("override,field", [
    ({"split_scheme": "seeded-shuffle"}, "split_scheme"),
    ({"split_scheme_version": 0}, "split_scheme_version"),
    ({"label_mapping": "something else"}, "label_mapping"),
    ({"image_size": 128}, "image_size"),
])
def test_contract_mismatch_refuses_warm_start(override, field):
    allowed, reason = warm_start_allowed(CURRENT, _parent(**override), image_size=256)
    assert not allowed
    assert field in reason


def test_parent_without_lineage_refuses_warm_start():
    """A pre-metadata parent cannot be shown to be safe."""
    allowed, reason = warm_start_allowed(CURRENT, {}, image_size=256)
    assert not allowed
    assert TRAINING_MANIFEST_NAME in reason


def test_training_manifest_survives_a_tarball_round_trip(tmp_path):
    """The fetch step delivers the parent as a tarball, so it must survive that."""
    src = tmp_path / "model"
    src.mkdir()
    write_training_manifest(src, CURRENT, image_size=256, model_name="m")
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in src.iterdir():
            tar.add(item, arcname=item.name)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted, filter="data")
    assert find_parent_manifest(extracted)["split_scheme"] == CURRENT["split_scheme"]


def test_warm_start_refuses_a_differently_sized_synthetic_parent(tmp_path):
    'The consequence: that parent must not be warm-started from.'
    synthetic = {
        "split_scheme": "generated-disjoint",
        "split_scheme_version": 2,
        "label_mapping": "synthetic",
        "split_ratios": {"train": 0.666667, "validation": 0.166667,
                         "test": 0.166666},
    }
    parent = dict(synthetic, dataset_id="parent-600", image_size=64)
    current = dict(synthetic, dataset_id="child-400")
    allowed, reason = warm_start_allowed(current, parent, image_size=64)
    assert not allowed
    assert "does not keep an image in the same split as the dataset changes" in reason


def test_warm_start_allows_the_same_synthetic_dataset(tmp_path):
    """The default path still works: fixed seed and count give the same id."""
    synthetic = {
        "dataset_id": "same-id",
        "split_scheme": "generated-disjoint",
        "split_scheme_version": 2,
        "label_mapping": "synthetic",
        "split_ratios": {"train": 0.666667, "validation": 0.166667,
                         "test": 0.166666},
    }
    allowed, reason = warm_start_allowed(
        synthetic, dict(synthetic, image_size=64), image_size=64
    )
    assert allowed, reason


def test_content_addressed_scheme_still_allows_cross_version_warm_start():
    """The asymmetry is the point: only a stable scheme may cross versions."""
    scheme = {
        "split_scheme": "content-addressed-sha256",
        "split_scheme_version": 2,
        "label_mapping": "platelets",
        "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
    }
    allowed, reason = warm_start_allowed(
        dict(scheme, dataset_id="new"),
        dict(scheme, dataset_id="old", image_size=256),
        image_size=256,
    )
    assert allowed, reason


from utils.dataset import SPLIT_SCHEME_V1_ENCODED_BYTES, assert_splits_disjoint, assign_split, canonical_image_digest

NOT_STABLE = "does not keep an image in the same split as the dataset changes"


def _metadata_variant(source: Path, dest: Path) -> Path:
    """The same decoded image, re-saved with different PNG text metadata."""
    from PIL import PngImagePlugin

    image = Image.open(source).convert("RGB")
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "a different comment entirely")
    image.save(dest, format="PNG", pnginfo=info)
    return dest


def test_a_metadata_rewrite_does_not_change_the_canonical_digest(tmp_path):
    'The digest must describe the pixels, not the container.'
    original = _images(tmp_path, 1)[0][0]
    variant = _metadata_variant(original, tmp_path / "variant.png")

    assert original.read_bytes() != variant.read_bytes(), (
        "the fixture did not actually change the file bytes, so this proves nothing"
    )
    assert canonical_image_digest(original) == canonical_image_digest(variant)


def _variant_that_moves_under_v1(tmp_path: Path):
    'An image plus a metadata-only variant whose V1 split DIFFERS.'
    from PIL import PngImagePlugin

    source = _images(tmp_path, 1)[0][0]
    original_split = assign_split(source, 0.7, 0.15,
                                  scheme_version=SPLIT_SCHEME_V1_ENCODED_BYTES)
    image = Image.open(source).convert("RGB")
    for attempt in range(200):
        variant = tmp_path / f"variant_{attempt}.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", f"metadata variant {attempt}")
        image.save(variant, format="PNG", pnginfo=info)
        moved = assign_split(variant, 0.7, 0.15,
                            scheme_version=SPLIT_SCHEME_V1_ENCODED_BYTES)
        if moved != original_split:
            return source, variant, original_split, moved
    raise AssertionError(
        "no metadata variant changed the v1 split in 200 attempts; the search is "
        "broken, so any v2 assertion built on it would prove nothing"
    )


def test_the_same_image_in_two_splits_is_refused(tmp_path):
    'Duplicate detection must see through the container too.'
    source, variant, _, _ = _variant_that_moves_under_v1(tmp_path)

    split_dirs = {}
    for split, image in (("train", source), ("test", variant)):
        class_dir = tmp_path / split / "good"
        class_dir.mkdir(parents=True)
        (class_dir / "img.png").write_bytes(image.read_bytes())
        split_dirs[split] = tmp_path / split

    with pytest.raises(RuntimeError, match="share image content"):
        assert_splits_disjoint(split_dirs, ("good", "defect"))


def test_an_unsupported_scheme_version_is_refused(tmp_path):
    """Assigning under a version this code does not implement would be a guess."""
    image = _images(tmp_path, 1)[0][0]
    with pytest.raises(ValueError, match="Unsupported split scheme version"):
        assign_split(image, 0.7, 0.15, scheme_version=99)
