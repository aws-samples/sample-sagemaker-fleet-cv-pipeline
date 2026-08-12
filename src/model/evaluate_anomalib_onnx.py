# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from utils.tarball import extract_tarball
from utils.dataset import resolve_dataset_root, load_images
from utils.metrics import binary_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model-dir", default="/opt/ml/processing/input/model")
    parser.add_argument("--test-dir", default="/opt/ml/processing/input/test")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/evaluation")
    args = parser.parse_args()

    # Resolve model artifact -- may be a tarball or pre-extracted
    model_base = Path(args.model_dir)
    model_dir = model_base
    tarballs = list(model_base.rglob("*.tar.gz")) + list(model_base.rglob("*.tgz"))
    if tarballs:
        model_dir = model_base / "extracted"
        extract_tarball(tarballs[0], model_dir)

    onnx_models = list(model_dir.rglob("*.onnx"))
    if not onnx_models:
        raise ValueError("No ONNX model found in model artifact")
    onnx_path = onnx_models[0]

    # Resolve test data -- may be a tarball or pre-extracted
    test_base = Path(args.test_dir)
    test_dir = test_base
    tarballs = list(test_base.rglob("*.tar.gz")) + list(test_base.rglob("*.tgz"))
    if tarballs:
        test_dir = test_base / "extracted"
        extract_tarball(tarballs[0], test_dir)

    root = resolve_dataset_root(test_dir)

    good = load_images(root / "good", args.image_size)
    defect = load_images(root / "defect", args.image_size)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    good_recon = sess.run([output_name], {input_name: good})[0]
    defect_recon = sess.run([output_name], {input_name: defect})[0]
    good_err = np.mean((good_recon - good) ** 2, axis=(1, 2, 3))
    defect_err = np.mean((defect_recon - defect) ** 2, axis=(1, 2, 3))

    labels = np.concatenate([np.zeros_like(good_err), np.ones_like(defect_err)])
    scores = np.concatenate([good_err, defect_err])
    image_auroc = binary_auc(labels, scores)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"image_auroc": image_auroc}, f, indent=2)

    model_quality = {
        "version": 0,
        "dataset_type": "cv-anomaly-onnx",
        "binary_classification_metrics": {
            "auc": {"value": float(image_auroc)},
        },
    }
    with open(out_dir / "model_quality_statistics.json", "w", encoding="utf-8") as f:
        json.dump(model_quality, f, indent=2)

    print(f"onnx_eval:image_auroc={image_auroc:.4f}")


if __name__ == "__main__":
    main()