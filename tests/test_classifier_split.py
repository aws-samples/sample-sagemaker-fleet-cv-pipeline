# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests that classifier training can never consume the test channel.

The quality gate evaluates the complete test split, so its integrity
depends on the classifier training exclusively from the train and
validation channels. These tests pin that contract at three layers:
the data-preparation function, the training script's inputs, and the
pipeline stack's channel wiring.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402

# The training code depends on torch/torchvision, which are available in the
# training container but not necessarily in the repo venv. Data-path tests
# skip when absent (run them via the container); the source-level contract
# tests below run everywhere.
try:
    from model.train_classifier import prepare_classifier_data, stratified_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch/torchvision not installed")


def _write_images(folder: Path, count: int, color):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (16, 16), color=color).save(folder / f"img_{i:03d}.png")


@pytest.fixture()
def channel_dirs(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    _write_images(train_root / "good", 20, (10, 10, 10))
    _write_images(val_root / "good", 10, (20, 20, 20))
    _write_images(val_root / "defect", 4, (200, 20, 20))
    return train_root, val_root


@needs_torch
def test_prepare_uses_only_train_and_validation(channel_dirs):
    train_root, val_root = channel_dirs
    train_x, train_y, hold_x, hold_y = prepare_classifier_data(
        train_root, val_root, image_size=16, seed=7
    )
    # Every image accounted for by the two channels: 20 train-good plus a
    # 70/30 stratified split of 14 validation images.
    assert len(train_y) + len(hold_y) == 20 + 14
    assert len(train_y) == 20 + 7 + 2  # train-good + 70% of 10 good + 70% of 4 defect
    # Both sides carry defects (stratified), holdout has the 30% remainder.
    assert int(train_y.sum()) == 2
    assert int(hold_y.sum()) == 2


@needs_torch
def test_prepare_is_deterministic(channel_dirs):
    train_root, val_root = channel_dirs
    a = prepare_classifier_data(train_root, val_root, image_size=16, seed=7)
    b = prepare_classifier_data(train_root, val_root, image_size=16, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@needs_torch
def test_prepare_rejects_defect_free_validation(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "validation"
    _write_images(train_root / "good", 5, (10, 10, 10))
    _write_images(val_root / "good", 5, (20, 20, 20))
    with pytest.raises(ValueError, match="no defect images"):
        prepare_classifier_data(train_root, val_root, image_size=16, seed=7)


@needs_torch
def test_stratified_split_covers_all_indices():
    labels = np.array([0] * 18 + [1] * 2)
    train_idx, holdout_idx = stratified_split(labels, train_fraction=0.7, seed=3)
    combined = np.sort(np.concatenate([train_idx, holdout_idx]))
    assert np.array_equal(combined, np.arange(len(labels)))
    assert set(train_idx).isdisjoint(holdout_idx)


def test_training_script_has_no_test_channel():
    source = (REPO_ROOT / "src" / "model" / "train_classifier.py").read_text()
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
