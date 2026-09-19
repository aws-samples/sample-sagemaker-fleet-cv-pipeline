# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Tests for dataset root resolution and synthetic image generation.'
import importlib.util
import json
import random
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.dataset import assert_splits_disjoint, resolve_dataset_root  # noqa: E402


def _load_preprocess():
    """Load preprocess_cv.py by path -- it is a job entry point, not a package."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_cv", REPO_ROOT / "src" / "data" / "preprocess_cv.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess_cv = _load_preprocess()


@pytest.mark.parametrize("image_size", [16, 32, 48, 64, 72, 80, 96, 224, 256])
def test_anomalous_image_generates_at_any_size(image_size):
    for seed in range(25):
        random.seed(seed)
        img = preprocess_cv.create_anomalous_image((image_size, image_size))
        assert img.size == (image_size, image_size)


@pytest.mark.parametrize("image_size", [16, 64, 256])
def test_normal_image_generates_at_any_size(image_size):
    random.seed(0)
    img = preprocess_cv.create_normal_image((image_size, image_size))
    assert img.size == (image_size, image_size)


def test_anomalous_image_differs_from_normal():
    """The anomaly must actually mark the image, or the AUROC gate is vacuous."""
    random.seed(7)
    normal = preprocess_cv.create_normal_image((64, 64))
    random.seed(7)
    anomalous = preprocess_cv.create_anomalous_image((64, 64))
    assert list(normal.getdata()) != list(anomalous.getdata())


def _make_split(root: Path, split: str, n: int, anomaly_ratio: float):
    preprocess_cv.create_anomalib_dataset(
        root, n, split, anomaly_ratio=anomaly_ratio, image_size=32
    )


def test_empty_defect_dir_does_not_satisfy_require_defect(tmp_path):
    'An empty defect/ must not make a train directory look like a test root.'
    root = tmp_path / "data"
    _make_split(root, "test", 4, 0.0)
    assert (root / "test" / "defect").is_dir()
    assert not any((root / "test" / "defect").iterdir())

    with pytest.raises(ValueError, match="Could not resolve dataset root"):
        resolve_dataset_root(root, "test", require_defect=True)


def test_train_split_still_resolves_without_defects(tmp_path):
    root = tmp_path / "data"
    _make_split(root, "train", 4, 0.0)
    assert resolve_dataset_root(root, "train", require_defect=False).name == "train"


def test_named_split_is_preferred_over_recursive_search(tmp_path):
    root = tmp_path / "data"
    _make_split(root, "validation", 8, 0.25)
    _make_split(root, "test", 8, 0.25)
    _make_split(root, "payload-aaa", 8, 0.25)
    assert resolve_dataset_root(root, "test", require_defect=True).name == "test"
    assert resolve_dataset_root(root, "validation", require_defect=True).name == "validation"


def test_missing_test_split_does_not_fall_back_to_train(tmp_path):
    """Asking for test with only a complete train present must raise."""
    root = tmp_path / "data"
    _make_split(root, "train", 8, 0.25)          # both classes, as is now normal
    with pytest.raises(ValueError) as excinfo:
        resolve_dataset_root(root, "test", require_defect=True)
    assert "train" in str(excinfo.value)
    assert "refused" in str(excinfo.value)


def test_a_base_named_for_another_split_is_refused(tmp_path):
    "The BASE's own name is checked, not only the names below it."
    root = tmp_path / "data" / "test"
    _make_split(root, "nested", 8, 0.25)
    with pytest.raises(ValueError) as excinfo:
        resolve_dataset_root(root, "train", require_defect=False)
    assert "test" in str(excinfo.value)


def test_a_named_split_inside_a_foreign_base_is_also_refused(tmp_path):
    'The named-split branch must not bypass the base check.'
    root = tmp_path / "data" / "test"
    _make_split(root, "train", 8, 0.25)
    with pytest.raises(ValueError) as excinfo:
        resolve_dataset_root(root, "train", require_defect=False)
    assert "test" in str(excinfo.value)


def test_an_unrelated_ancestor_name_is_not_refused(tmp_path):
    "Only the supplied base is judged; ancestors above it are the caller's choice."
    root = tmp_path / "test" / "data"
    _make_split(root, "train", 8, 0.25)
    assert resolve_dataset_root(root, "train", require_defect=False).name == "train"


def test_incomplete_test_split_does_not_fall_back_to_train(tmp_path):
    """An empty test/good must not silently resolve to train either."""
    root = tmp_path / "data"
    _make_split(root, "train", 8, 0.25)
    (root / "test" / "good").mkdir(parents=True)
    with pytest.raises(ValueError):
        resolve_dataset_root(root, "test", require_defect=True)


def test_validation_lookup_does_not_return_train_or_test(tmp_path):
    root = tmp_path / "data"
    _make_split(root, "train", 8, 0.25)
    _make_split(root, "test", 8, 0.25)
    with pytest.raises(ValueError):
        resolve_dataset_root(root, "validation", require_defect=True)


def test_unnamed_nested_root_is_still_found(tmp_path):
    'A split root under an arbitrary directory name is legitimate.'
    root = tmp_path / "data" / "extracted_test"
    _make_split(root, "test", 8, 0.25)
    assert resolve_dataset_root(tmp_path / "data", "test", require_defect=True).name == "test"


def test_ambiguous_root_is_rejected(tmp_path):
    'Two candidate roots and no named match must raise, not pick one.'
    root = tmp_path / "data"
    _make_split(root, "batch_a", 8, 0.25)
    _make_split(root, "batch_b", 8, 0.25)
    with pytest.raises(ValueError, match="Ambiguous dataset root"):
        resolve_dataset_root(root, "test", require_defect=True)


def test_split_root_directly_supplied(tmp_path):
    """base_dir may itself be the split root, as it is in the pipeline."""
    root = tmp_path / "data"
    _make_split(root, "test", 8, 0.25)
    assert resolve_dataset_root(root / "test", "test", require_defect=True) == root / "test"


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Could not resolve dataset root"):
        resolve_dataset_root(tmp_path, "test", require_defect=True)


def _run_preprocess(out_dir: Path, tmp_dir: Path, **kw):
    import subprocess
    args = {
        "--num-samples": "40", "--image-size": "32", "--anomaly-ratio": "0.25",
        "--output-dir": str(out_dir), "--temp-dir": str(tmp_dir),
    }
    args.update({k: str(v) for k, v in kw.items()})
    argv = [sys.executable, str(REPO_ROOT / "src" / "data" / "preprocess_cv.py")]
    for k, v in args.items():
        argv += [k, v]
    import os
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((out_dir / "dataset_manifest.json").read_text())


def test_manifest_is_inside_every_split_tarball(tmp_path):
    'The manifest must be in the tarball, not only at the output root.'
    manifest = _run_preprocess(tmp_path / "out", tmp_path / "tmp")
    assert manifest["dataset_id"]
    for split in ("train", "validation", "test"):
        tar_path = tmp_path / "out" / split / f"{split}.tar.gz"
        with tarfile.open(tar_path, "r:gz") as tar:
            names = tar.getnames()
        assert f"{split}/dataset_manifest.json" in names, (
            f"{split}.tar.gz does not carry the dataset manifest: {names[:5]}"
        )


def test_dataset_id_is_stable_for_the_same_inputs(tmp_path):
    'Same seed and parameters -> same dataset_id.'
    a = _run_preprocess(tmp_path / "a", tmp_path / "at")
    b = _run_preprocess(tmp_path / "b", tmp_path / "bt")
    assert a["dataset_id"] == b["dataset_id"]
    assert a["splits"] == b["splits"]


def test_dataset_id_changes_when_the_data_changes(tmp_path):
    """A different seed produces different images and so a different identity."""
    a = _run_preprocess(tmp_path / "a", tmp_path / "at")
    b = _run_preprocess(tmp_path / "b", tmp_path / "bt", **{"--seed": 7})
    assert a["dataset_id"] != b["dataset_id"]


def test_splits_do_not_share_content(tmp_path):
    """Every split hash must be distinct -- the invariant the gate depends on."""
    manifest = _run_preprocess(tmp_path / "out", tmp_path / "tmp")
    hashes = [info["content_sha256"] for info in manifest["splits"].values()]
    assert len(set(hashes)) == len(hashes)


def test_disjointness_catches_a_single_leaked_image(tmp_path):
    'One shared image must fail, not just wholly identical splits.'
    root = tmp_path / "d"
    for split in ("train", "validation", "test"):
        _make_split(root, split, 8, 0.25)
    dirs = {s: root / s for s in ("train", "validation", "test")}

    assert_splits_disjoint(dirs)  # clean

    import shutil
    leaked = sorted((root / "train" / "good").iterdir())[0]
    shutil.copy2(leaked, root / "test" / "good" / "LEAKED.png")
    with pytest.raises(RuntimeError, match="share image content"):
        assert_splits_disjoint(dirs)
