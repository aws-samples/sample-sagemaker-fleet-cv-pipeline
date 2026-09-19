# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Evaluate a classifier ONNX model on the complete test channel. The classifier trains exclusively on the train and validation channels (see train_classifier.py), so every test image is unseen by construction and the quality gate scores the entire split. Computes accuracy and writes quality metrics for the pipeline quality gate.'
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from utils.dataset import resolve_dataset_root, load_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["good", "defect"]


def load_test_channel(test_root: Path, image_size: int):
    """Load every image in the test channel's good/ and defect/ directories."""
    images = []
    labels = []
    for class_name in CLASS_NAMES:
        class_dir = test_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Test channel is missing the '{class_name}' directory under {test_root}; "
                "the quality gate needs both classes to compute balanced accuracy."
            )
        class_images = load_images(class_dir, image_size)
        images.append(class_images)
        labels.extend([CLASS_NAMES.index(class_name)] * len(class_images))
        logger.info("Test channel: %d '%s' images", len(class_images), class_name)
    return np.concatenate(images).astype(np.float32), np.array(labels, dtype=np.int64)


def check_logits(logits, expected_shape):
    'Refuse unusable ONNX logits before any prediction is derived from them.'
    if logits.shape != expected_shape:
        raise ValueError(
            f"ONNX logits have shape {logits.shape}, expected {expected_shape}. "
            f"argmax over an unexpected shape would produce predictions that do "
            f"not correspond to the test images."
        )
    if not np.isfinite(logits).all():
        raise ValueError(
            f"ONNX logits contain {int((~np.isfinite(logits)).sum())} non-finite "
            f"value(s). argmax over NaN silently returns class 0, reporting every "
            f"image as 'good'; refusing to derive predictions from it."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-dir", default="/opt/ml/processing/input/model")
    parser.add_argument("--test-dir", default="/opt/ml/processing/input/test")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/evaluation")
    args = parser.parse_args()

    # Resolve model artifact -- may be a tarball or pre-extracted
    model_base = Path(args.model_dir)
    model_dir = model_base
    tarballs = list(model_base.rglob("*.tar.gz")) + list(model_base.rglob("*.tgz"))
    if tarballs:
        from utils.tarball import extract_tarball
        model_dir = model_base / "extracted"
        extract_tarball(tarballs[0], model_dir)

    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        onnx_files = list(model_dir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        onnx_path = onnx_files[0]

    # Resolve test data
    test_base = Path(args.test_dir)
    test_dir = test_base
    test_tarballs = list(test_base.rglob("*.tar.gz")) + list(test_base.rglob("*.tgz"))
    if test_tarballs:
        from utils.tarball import extract_tarball
        test_dir = test_base / "extracted"
        extract_tarball(test_tarballs[0], test_dir)

    test_root = resolve_dataset_root(test_dir, "test", require_defect=True)
    test_images, test_labels = load_test_channel(test_root, args.image_size)

    # Run ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: test_images.astype(np.float32)})[0]

    # Validate the model's output BEFORE deriving predictions. np.argmax over
    # all-NaN logits returns index 0 without complaint, which silently reports
    # "every image is good" as though it were a real prediction.
    check_logits(logits, (len(test_labels), len(CLASS_NAMES)))

    preds = np.argmax(logits, axis=1)

    accuracy = float(np.mean(preds == test_labels))
    per_class_recall = []
    support = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = test_labels == class_idx
        support[class_name] = int(mask.sum())
        if not mask.any():
            raise ValueError(
                f"Test channel has no '{class_name}' images; per-class recall is "
                f"undefined. Regenerate the dataset with both classes present."
            )
        per_class_recall.append(float(np.mean(preds[mask] == class_idx)))
    balanced_accuracy = float(np.mean(per_class_recall))
    defect_recall = per_class_recall[CLASS_NAMES.index("defect")]

    # Confusion counts make the gated metrics interpretable: balanced accuracy
    # alone does not say whether a passing model floods the line with false
    # positives.
    confusion = {
        f"true_{actual}_pred_{predicted}": int(
            np.sum((test_labels == a_idx) & (preds == p_idx))
        )
        for a_idx, actual in enumerate(CLASS_NAMES)
        for p_idx, predicted in enumerate(CLASS_NAMES)
    }

    logger.info(
        "ONNX evaluation: accuracy=%.4f balanced_accuracy=%.4f defect_recall=%.4f (%d samples)",
        accuracy, balanced_accuracy, defect_recall, len(test_images),
    )
    logger.info("Support: %s", support)
    logger.info("Confusion: %s", confusion)

    # Write metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "defect_recall": defect_recall,
        "good_recall": per_class_recall[CLASS_NAMES.index("good")],
        "num_samples": len(test_images),
        "support": support,
        "confusion": confusion,
    }
    # allow_nan=False: bare NaN/Infinity is not valid JSON and must never reach
    # the pipeline's numeric quality-gate comparison.
    with open(output_dir / "model_quality_statistics.json", "w") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)

    print(f"evaluation:accuracy={accuracy:.4f}")
    print(f"evaluation:balanced_accuracy={balanced_accuracy:.4f}")
    print(f"evaluation:defect_recall={defect_recall:.4f}")


if __name__ == "__main__":
    main()
