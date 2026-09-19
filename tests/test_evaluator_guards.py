# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'The evaluators must refuse unusable model output before deriving a metric.'

import numpy as np
import pytest

from evaluate_anomalib_onnx import check_reconstruction
from evaluate_classifier_onnx import check_logits

SOURCE = np.zeros((4, 3, 8, 8), dtype=np.float32)


def test_a_healthy_reconstruction_is_accepted():
    """The control. A guard that rejects everything is not a guard."""
    check_reconstruction("good", SOURCE.copy(), SOURCE)


def test_a_broadcastable_shape_is_rejected():
    'The case that matters most: numpy would broadcast rather than raise.'
    wrong = np.zeros((4, 3, 8, 1), dtype=np.float32)
    assert (wrong - SOURCE).shape == SOURCE.shape

    with pytest.raises(ValueError, match="broadcast"):
        check_reconstruction("good", wrong, SOURCE)


def test_a_nan_reconstruction_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        check_reconstruction("good", np.full_like(SOURCE, np.nan), SOURCE)


def test_an_infinite_reconstruction_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        check_reconstruction("defect", np.full_like(SOURCE, np.inf), SOURCE)


def test_a_single_non_finite_value_is_enough():
    """One NaN among 768 finite values must still fail. .all() not .any()."""
    nearly = SOURCE.copy()
    nearly[2, 1, 4, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_reconstruction("good", nearly, SOURCE)


def test_the_rejection_names_the_split():
    """An operator reading the log needs to know which split failed."""
    with pytest.raises(ValueError, match="defect"):
        check_reconstruction("defect", np.full_like(SOURCE, np.nan), SOURCE)


def test_healthy_logits_are_accepted():
    check_logits(np.zeros((4, 2), dtype=np.float32), (4, 2))


def test_a_wrong_class_count_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        check_logits(np.zeros((4, 3), dtype=np.float32), (4, 2))


def test_a_wrong_row_count_is_rejected():
    """Fewer rows than test images means the predictions do not correspond to them."""
    with pytest.raises(ValueError, match="expected"):
        check_logits(np.zeros((3, 2), dtype=np.float32), (4, 2))


def test_nan_logits_are_rejected_because_argmax_would_not_complain():
    """Establishes WHY this matters, not just that it raises."""
    nan_logits = np.full((4, 2), np.nan, dtype=np.float32)
    assert np.argmax(nan_logits, axis=1).tolist() == [0, 0, 0, 0]

    with pytest.raises(ValueError, match="non-finite"):
        check_logits(nan_logits, (4, 2))


def test_a_single_nan_logit_is_enough():
    logits = np.zeros((4, 2), dtype=np.float32)
    logits[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_logits(logits, (4, 2))


def test_the_rejection_counts_the_non_finite_values():
    """The message must quantify the damage, not just assert it."""
    logits = np.full((4, 2), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="8 non-finite"):
        check_logits(logits, (4, 2))
