# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Evaluation metrics shared across model scripts."""
import numpy as np


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    'Rank-based AUC via the Mann-Whitney U statistic (no sklearn dependency).'
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            f"binary_auc expects 1-D arrays, got labels.ndim={labels.ndim} "
            f"scores.ndim={scores.ndim}"
        )
    if labels.shape != scores.shape:
        raise ValueError(
            f"binary_auc requires one score per label, got {labels.shape[0]} "
            f"labels and {scores.shape[0]} scores"
        )
    if labels.size == 0:
        raise ValueError("binary_auc received empty input; nothing to score")
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        raise ValueError(
            f"binary_auc received {n_bad} non-finite score(s) out of "
            f"{scores.size} (NaN or infinity). Ranking them is meaningless: NaN "
            f"sorts last and would be treated as the strongest evidence of an "
            f"anomaly, which can report a perfect AUROC for a model producing no "
            f"usable output. Fix the model or the evaluation input rather than "
            f"scoring it."
        )
    valid_labels = np.isin(labels, (0, 1))
    if not valid_labels.all():
        offenders = np.unique(labels[~valid_labels])
        raise ValueError(
            f"binary_auc expects labels in {{0, 1}}, found {offenders.tolist()}"
        )

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
        # No ranking information. 0.5 is below every gate this project ships.
        return 0.5
    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
