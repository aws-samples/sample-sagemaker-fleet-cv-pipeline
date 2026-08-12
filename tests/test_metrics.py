# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic tests for the shared AUC implementation.

Run with:  python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np

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
    # Identical scores carry no ranking information: AUC must be 0.5
    # regardless of label order.
    assert binary_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert binary_auc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert binary_auc(np.array([1, 0, 1, 0]), np.full(4, 0.7)) == 0.5


def test_partial_ties_average_ranks():
    # scores: neg=0.2, pos=0.5, neg=0.5, pos=0.9
    # The 0.5 tie contributes half a concordant pair: AUC = (1 + 0.5 + 1 + 1) / 4
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.5, 0.5, 0.9])
    assert binary_auc(labels, scores) == 0.875


def test_one_class_returns_half():
    assert binary_auc(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5
    assert binary_auc(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5


def test_known_reference_case():
    # neg scores {0.1, 0.4, 0.2}, pos scores {0.35, 0.8, 0.9}: of the 9
    # (neg, pos) pairs only (0.4, 0.35) is discordant -> AUC = 8/9.
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    expected = 8 / 9
    assert abs(binary_auc(labels, scores) - expected) < 1e-12


def test_matches_sklearn_when_available():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return  # optional cross-check only
    rng = np.random.default_rng(42)
    for _ in range(20):
        labels = rng.integers(0, 2, size=50)
        if labels.min() == labels.max():
            continue
        # Quantized scores force plenty of ties.
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
