# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Evaluation metrics shared across model scripts."""
import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC via the Mann-Whitney U statistic (no sklearn dependency).

    Tied scores receive the average of the rank positions they span; without
    this, the result for tied scores depends on input order (sequential ranks
    give 1.0 or 0.0 for identical scores where the correct value is 0.5).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # ranks are 1-based; every member of a tie group gets the group mean
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = labels == 1
    n_pos = np.sum(pos)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)