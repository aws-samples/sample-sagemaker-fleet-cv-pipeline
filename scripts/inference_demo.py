# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Sample ONNX inference script for registered SageMaker models.

Downloads the latest model package from Model Registry, extracts the ONNX
artifact, and runs inference on a sample image (synthetic or user-provided).

Usage:
    # Anomalib (anomaly detection)
    python scripts/inference_demo.py --model-group sm1k-anomaly-model-group --type anomalib

    # Classifier (defect classification)
    python scripts/inference_demo.py --model-group sm1k-classifier-model-group --type classifier

    # With a real image
    python scripts/inference_demo.py --model-group sm1k-anomaly-model-group --type anomalib --image path/to/image.png
"""
import argparse
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
from PIL import Image


def get_latest_model_artifact(model_group: str, region: str, camera_id: str) -> str:
    """Return S3 URI of the latest model artifact for *camera_id*.

    Every camera in a fleet registers into the same package group, so the
    lookup filters on the CameraId lineage metadata; taking the newest
    package unfiltered could return another camera's model. ListModelPackages
    cannot filter on customer metadata, so walk newest-first and describe.
    """
    sm = boto3.client("sagemaker", region_name=region)
    paginator = sm.get_paginator("list_model_packages")
    pages = paginator.paginate(
        ModelPackageGroupName=model_group,
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    for page in pages:
        for summary in page["ModelPackageSummaryList"]:
            desc = sm.describe_model_package(
                ModelPackageName=summary["ModelPackageArn"]
            )
            metadata = desc.get("CustomerMetadataProperties", {})
            if metadata.get("CameraId") == camera_id:
                return desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    raise RuntimeError(
        f"No model package in {model_group} with CameraId={camera_id}. "
        "Pass --camera-id matching the CameraId parameter of the pipeline run."
    )


def download_and_extract(s3_uri: str, dest: Path, region: str) -> Path:
    """Download model.tar.gz from S3 and extract, return path to .onnx file."""
    s3 = boto3.client("s3", region_name=region)
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    tar_path = dest / "model.tar.gz"
    s3.download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    onnx_files = list(dest.rglob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"No .onnx file found in {s3_uri}")
    return onnx_files[0]


def create_sample_image(size: int) -> np.ndarray:
    """Generate a synthetic test image."""
    img = Image.new("RGB", (size, size), color=(180, 180, 185))
    return np.array(img)


def run_anomalib_inference(onnx_path: Path, image: np.ndarray, image_size: int):
    """Run anomalib autoencoder inference -- compute reconstruction error."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]

    img = Image.fromarray(image).resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    batch = arr[np.newaxis, ...]

    recon = sess.run([out.name], {inp.name: batch})[0]
    mse = float(np.mean((recon - batch) ** 2))
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Anomaly score (lower = more normal): {mse:.6f}")
    return mse


def run_classifier_inference(onnx_path: Path, image: np.ndarray, image_size: int):
    """Run defect classifier inference -- softmax over good/defect logits."""
    import onnxruntime as ort

    class_names = ["good", "defect"]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]

    img = Image.fromarray(image).resize((image_size, image_size))
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    batch = arr[np.newaxis, ...]

    logits = sess.run([out.name], {inp.name: batch})[0][0]
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    pred = int(np.argmax(probs))
    print(f"Predicted class: {class_names[pred]}")
    for name, prob in zip(class_names, probs):
        print(f"  {name}: {prob:.4f}")
    return class_names[pred]


def main():
    parser = argparse.ArgumentParser(description="ONNX inference demo")
    parser.add_argument("--model-group", required=True, help="Model package group name")
    parser.add_argument("--type", required=True, choices=["anomalib", "classifier"])
    parser.add_argument("--image", default=None, help="Path to input image (optional)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size (default: 256 for anomalib, 224 for classifier)")
    parser.add_argument("--camera-id", default="default",
                        help="Fleet device identity; selects the newest package "
                             "registered with matching CameraId metadata")
    parser.add_argument("--region", default=boto3.session.Session().region_name)
    args = parser.parse_args()

    if not args.region:
        raise RuntimeError(
            "AWS region could not be resolved from the boto3 session. "
            "Pass --region or configure AWS_REGION/AWS_DEFAULT_REGION."
        )

    if args.image_size is None:
        args.image_size = 256 if args.type == "anomalib" else 224

    print(f"Model group: {args.model_group}")
    print(f"Region:      {args.region}")
    print(f"Type:        {args.type}")
    print(f"Camera:      {args.camera_id}")

    s3_uri = get_latest_model_artifact(args.model_group, args.region, args.camera_id)
    print(f"Model S3:    {s3_uri}")

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = download_and_extract(s3_uri, Path(tmpdir), args.region)
        print(f"ONNX model:  {onnx_path.name}")

        if args.image:
            image = np.array(Image.open(args.image).convert("RGB"))
            print(f"Input image: {args.image} ({image.shape})")
        else:
            image = create_sample_image(args.image_size)
            print(f"Input image: synthetic ({image.shape})")

        print("---")
        if args.type == "anomalib":
            run_anomalib_inference(onnx_path, image, args.image_size)
        else:
            run_classifier_inference(onnx_path, image, args.image_size)


if __name__ == "__main__":
    main()