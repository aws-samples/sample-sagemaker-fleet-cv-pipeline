# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic tests for the shared AUC implementation.

Run with:  python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.metrics import binary_auc  # noqa: E402


def test_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auc(labels, scores) == 1.0


def test_perfectly_reversed():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auc(labels, scores) == 0.0


def test_all_scores_tied_is_half():
    assert binary_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert binary_auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert binary_auc(np.array([1, 0, 1, 0]), np.full(4, 0.7)) == 0.5


def test_partial_ties_average_ranks():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.5, 0.5, 0.9])
    assert binary_auc(labels, scores) == 0.875


def test_one_class_returns_half():
    'A split with one class carries no ranking information -> 0.5.'
    assert binary_auc(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5
    assert binary_auc(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5


@pytest.mark.parametrize("scores", [
    [0.1, 0.2, np.nan, np.nan],      # NaN on defects -> used to return 1.0
    [np.nan, np.nan, 0.1, 0.2],      # NaN on good    -> used to return 0.0
    [np.nan] * 4,                    # all NaN        -> used to return 1.0
    [0.1, 0.2, np.inf, np.inf],      # +inf           -> used to return 1.0
    [0.1, 0.2, 0.3, -np.inf],
])
def test_non_finite_scores_raise(scores):
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="non-finite"):
        binary_auc(labels, np.array(scores, dtype=float))


def test_nan_no_longer_passes_the_default_gate():
    'The concrete failure: a NaN model clearing the shipped MinAUROC.'
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, np.nan, np.nan])
    with pytest.raises(ValueError, match="non-finite"):
        binary_auc(labels, scores)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="one score per label"):
        binary_auc(np.array([0, 1, 1]), np.array([0.1, 0.2]))


def test_non_one_dimensional_input_raises():
    with pytest.raises(ValueError, match="1-D arrays"):
        binary_auc(np.array([[0, 1], [1, 0]]), np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty input"):
        binary_auc(np.array([]), np.array([]))


def test_non_binary_labels_raise():
    with pytest.raises(ValueError, match=r"labels in \{0, 1\}"):
        binary_auc(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


def test_known_reference_case():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    expected = 8 / 9
    assert abs(binary_auc(labels, scores) - expected) < 1e-12


def test_matches_sklearn_when_available():
    'Cross-check against scikit-learn.'
    roc_auc_score = pytest.importorskip(
        "sklearn.metrics", reason="scikit-learn not installed -- "
        "pip install -r requirements-dev.txt"
    ).roc_auc_score
    rng = np.random.default_rng(42)
    for _ in range(20):
        labels = rng.integers(0, 2, size=50)
        if labels.min() == labels.max():
            continue
        scores = np.round(rng.random(50), 1)
        assert abs(binary_auc(labels, scores) - roc_auc_score(labels, scores)) < 1e-12


def test_input_order_invariance():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=30)
    labels[0], labels[1] = 0, 1  # both classes present
    scores = np.round(rng.random(30), 1)
    base = binary_auc(labels, scores)
    for _ in range(5):
        perm = rng.permutation(30)
        assert binary_auc(labels[perm], scores[perm]) == base
