# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Tests that classifier training can never consume the test channel.'
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402

try:
    from model.train_classifier import prepare_classifier_data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch/torchvision not installed -- pip install -r requirements-dev.txt",
)


def _write_images(folder: Path, count: int, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (16, 16), color=color).save(folder / f"img_{i:03d}.png")


@pytest.fixture()
def channel_dirs(tmp_path):
    """Train and validation channels, both carrying both classes."""
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    _write_images(train_root / "good", 20, (10, 10, 10))
    _write_images(train_root / "defect", 6, (180, 30, 30))
    _write_images(val_root / "good", 10, (20, 20, 20))
    _write_images(val_root / "defect", 4, (200, 20, 20))
    return train_root, val_root


@needs_torch
def test_train_comes_from_train_split_only(channel_dirs):
    """Training arrays must be exactly the train split -- no validation images."""
    train_root, val_root = channel_dirs
    train_x, train_y, val_x, val_y = prepare_classifier_data(
        train_root, val_root, image_size=16
    )
    assert len(train_y) == 26          # 20 good + 6 defect, the whole train split
    assert int(train_y.sum()) == 6     # its defects, not borrowed from validation
    assert len(val_y) == 14            # validation untouched and complete
    assert int(val_y.sum()) == 4

    def _channels(x):
        "A per-image signature that separates the fixture's four colours."
        flat = x.reshape(len(x), -1)
        return {(round(float(v.min()), 3), round(float(v.max()), 3)) for v in flat}

    def _sig(color):
        v = min(color) / 255.0, max(color) / 255.0
        return (round(v[0], 3), round(v[1], 3))

    train_expected = {_sig((10, 10, 10)), _sig((180, 30, 30))}
    val_expected = {_sig((20, 20, 20)), _sig((200, 20, 20))}
    assert _channels(train_x) == train_expected, (
        f"training tensors are not the TRAIN channel's images: got {sorted(_channels(train_x))}, "
        f"expected {sorted(train_expected)}"
    )
    assert _channels(val_x) == val_expected, (
        f"validation tensors are not the VALIDATION channel's images: "
        f"got {sorted(_channels(val_x))}, expected {sorted(val_expected)}"
    )


@needs_torch
def test_validation_is_not_reduced_by_training(channel_dirs):
    """Every validation image must reach the metric.

    The old 70/30 borrow left only 30% of validation for measuring.
    """
    train_root, val_root = channel_dirs
    _, _, val_x, val_y = prepare_classifier_data(train_root, val_root, image_size=16)
    on_disk = sum(
        1 for c in ("good", "defect") for _ in (val_root / c).glob("*.png")
    )
    assert len(val_y) == on_disk


@needs_torch
def test_prepare_is_deterministic(channel_dirs):
    train_root, val_root = channel_dirs
    a = prepare_classifier_data(train_root, val_root, image_size=16)
    b = prepare_classifier_data(train_root, val_root, image_size=16)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@needs_torch
def test_rejects_defect_free_train(tmp_path):
    """Zero training defects must fail loudly, not train a single-class model."""
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    _write_images(train_root / "good", 5, (10, 10, 10))
    _write_images(val_root / "good", 5, (20, 20, 20))
    _write_images(val_root / "defect", 2, (200, 20, 20))
    with pytest.raises(ValueError, match=r"train split .* has no defect"):
        prepare_classifier_data(train_root, val_root, image_size=16)


@needs_torch
def test_rejects_defect_free_validation(tmp_path):
    """Validation without defects cannot yield per-class recall."""
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    _write_images(train_root / "good", 5, (10, 10, 10))
    _write_images(train_root / "defect", 2, (180, 30, 30))
    _write_images(val_root / "good", 5, (20, 20, 20))
    with pytest.raises(ValueError, match=r"validation split .* has no defect"):
        prepare_classifier_data(train_root, val_root, image_size=16)


def test_training_script_has_no_test_channel():
    source = (REPO_ROOT / "src" / "model" / "train_classifier.py").read_text()
    assert "SM_CHANNEL_TEST" not in source
    assert '"--test"' not in source and "'--test'" not in source


def test_anomaly_training_script_has_no_test_channel():
    'The autoencoder reports AUROC on validation, so it must not see test.'
    source = (REPO_ROOT / "src" / "model" / "train_anomalib.py").read_text()
    assert "SM_CHANNEL_TEST" not in source
    assert '"--test"' not in source and "'--test'" not in source


def test_pipeline_does_not_mount_test_into_classifier_training():
    """The classifier InputData list must contain train and validation only."""
    source = (REPO_ROOT / "sagemaker_cv_pipeline_stack.py").read_text()
    match = re.search(
        r"classifier_input_data = \[(.*?)\n        \]", source, re.DOTALL
    )
    assert match, "classifier_input_data block not found"
    block = match.group(1)
    assert 'channel_name="train"' in block
    assert 'channel_name="validation"' in block
    assert 'channel_name="test"' not in block


def test_pipeline_does_not_mount_test_into_anomaly_training():
    """The anomaly InputData list must contain train and validation only."""
    source = (REPO_ROOT / "sagemaker_cv_pipeline_stack.py").read_text()
    match = re.search(r"anomalib_input_data = \[(.*?)\n        \]", source, re.DOTALL)
    assert match, "anomalib_input_data block not found"
    block = match.group(1)
    assert 'channel_name="train"' in block
    assert 'channel_name="validation"' in block
    assert 'channel_name="test"' not in block
