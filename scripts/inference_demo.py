# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Sample ONNX inference script for registered SageMaker models.'
import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aws_env import AwsEnvError, resolve_region  # noqa: E402
from config import ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP  # noqa: E402


def get_model_artifact(model_group: str, region: str, camera_id: str,
                       version: int = None, package_arn: str = None,
                       allow_unapproved: bool = False) -> str:
    'Return the S3 URI of a model artifact to run inference on.'
    sm = boto3.client("sagemaker", region_name=region)

    def _ineligible(desc: dict):
        'Why *desc* cannot be used, or None if it can.'
        status = desc.get("ModelApprovalStatus")
        if status != "Approved" and not allow_unapproved:
            return (
                f"approval status {status!r}, not 'Approved' (pass "
                f"--allow-unapproved to run it anyway)"
            )
        package_status = desc.get("ModelPackageStatus")
        if package_status != "Completed":
            return (
                f"status {package_status!r}, not 'Completed'; its artifact may be "
                f"absent"
            )
        try:
            desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        except (KeyError, IndexError):
            return "no InferenceSpecification container with a ModelDataUrl"
        return None

    def _artifact(desc: dict) -> str:
        reason = _ineligible(desc)
        if reason:
            raise RuntimeError(f"Model package {desc['ModelPackageArn']} has {reason}.")
        return desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    if package_arn:
        return _artifact(sm.describe_model_package(ModelPackageName=package_arn))
    if version is not None:
        # DescribeModelPackage does not accept "<group>/<version>" as a name; it
        # needs either a bare package name or a full ARN. Build the ARN.
        sts = boto3.client("sts", region_name=region)
        account = sts.get_caller_identity()["Account"]
        arn = (
            f"arn:aws:sagemaker:{region}:{account}:model-package/"
            f"{model_group}/{version}"
        )
        return _artifact(sm.describe_model_package(ModelPackageName=arn))

    paginator = sm.get_paginator("list_model_packages")
    kwargs = {
        "ModelPackageGroupName": model_group,
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
    }
    if not allow_unapproved:
        kwargs["ModelApprovalStatus"] = "Approved"
    skipped = []
    for page in paginator.paginate(**kwargs):
        for summary in page["ModelPackageSummaryList"]:
            desc = sm.describe_model_package(
                ModelPackageName=summary["ModelPackageArn"]
            )
            metadata = desc.get("CustomerMetadataProperties", {})
            if metadata.get("CameraId") != camera_id:
                continue
            reason = _ineligible(desc)
            if reason:
                skipped.append((summary["ModelPackageArn"], reason))
                continue
            if skipped:
                print(f"Skipped {len(skipped)} newer but ineligible package(s):")
                for arn, why in skipped:
                    print(f"  {arn}: {why}")
            return desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    if skipped:
        detail = "; ".join(f"{arn} ({why})" for arn, why in skipped[:5])
        raise RuntimeError(
            f"No usable model package for CameraId={camera_id!r} in {model_group}. "
            f"{len(skipped)} package(s) matched the camera but were ineligible: "
            f"{detail}"
        )

    qualifier = "" if allow_unapproved else "Approved "
    hint = (
        ""
        if allow_unapproved
        else " Pass --allow-unapproved to consider packages that are not Approved."
    )
    raise RuntimeError(
        f"No {qualifier}model package in {model_group} carries "
        f"CameraId={camera_id!r} metadata. Pass --camera-id matching the CameraId "
        f"parameter of the pipeline run that produced the model. Note that "
        f"packages registered before CameraId lineage was recorded carry no "
        f"CameraId at all and cannot be selected this way -- use --model-version "
        f"or --package-arn for those.{hint}"
    )


# Backwards-compatible alias for the previous name.
get_latest_model_artifact = get_model_artifact


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
    if recon.shape != batch.shape:
        raise RuntimeError(
            f"Anomaly ONNX output shape {recon.shape} does not match the input "
            f"{batch.shape}. NumPy would broadcast a mismatched shape and produce "
            f"a reconstruction error that looks reasonable but compares the image "
            f"against something other than its own reconstruction."
        )
    if not np.isfinite(recon).all():
        raise RuntimeError(
            f"Anomaly ONNX output contains "
            f"{int((~np.isfinite(recon)).sum())} non-finite value(s). The "
            f"resulting score cannot be ordered against a threshold, so no "
            f"decision made from it is meaningful."
        )
    mse = float(np.mean((recon - batch) ** 2))
    if not np.isfinite(mse):
        raise RuntimeError(f"Reconstruction error is {mse}, which is not finite.")
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

    raw = sess.run([out.name], {inp.name: batch})[0]
    # The COMPLETE array, before any indexing: a response with a valid first row
    # and non-finite values elsewhere would otherwise pass once indexed.
    if raw.shape != (1, len(class_names)) or not np.isfinite(raw).all():
        raise RuntimeError(
            f"Classifier ONNX output is unusable: shape {raw.shape}, "
            f"{int((~np.isfinite(raw)).sum())} non-finite value(s). argmax over NaN "
            f"silently returns class 0, so this would print 'good' with NaN "
            f"probabilities. Passing evaluation on one dataset does not guarantee "
            f"valid output for every later input."
        )
    logits = raw[0]
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    pred = int(np.argmax(probs))
    print(f"Predicted class: {class_names[pred]}")
    for name, prob in zip(class_names, probs):
        print(f"  {name}: {prob:.4f}")
    return class_names[pred]


def main():
    parser = argparse.ArgumentParser(description="ONNX inference demo")
    parser.add_argument("--model-group", required=True,
                        choices=[ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP],
                        help="Model package group name")
    parser.add_argument("--type", required=True, choices=["anomalib", "classifier"])
    parser.add_argument("--image", default=None, help="Path to input image (optional)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size (default: 256 for anomalib, 224 for classifier)")
    parser.add_argument("--camera-id", default="default",
                        help="Fleet device identity; selects the newest package "
                             "registered with matching CameraId metadata")
    parser.add_argument("--model-version", type=int, default=None,
                        help="Select an exact package version instead of the newest")
    parser.add_argument("--package-arn", default=None,
                        help="Select an exact model package ARN")
    parser.add_argument("--allow-unapproved", action="store_true",
                        help="Permit a package whose approval status is not Approved. "
                             "Off by default: rejecting a model must actually stop it "
                             "being served.")
    parser.add_argument("--region", default=None,
                        help="AWS Region (defaults to the shared resolution in aws_env.py)")
    args = parser.parse_args()

    try:
        args.region = resolve_region(args.region)
    except AwsEnvError as exc:
        raise SystemExit(str(exc))

    if args.image_size is None:
        args.image_size = 256 if args.type == "anomalib" else 224

    print(f"Model group: {args.model_group}")
    print(f"Region:      {args.region}")
    print(f"Type:        {args.type}")
    if args.package_arn or args.model_version is not None:
        selector = ("--package-arn" if args.package_arn else "--model-version")
        print(f"Camera:      (ignored -- {selector} selects the package exactly)")
    else:
        print(f"Camera:      {args.camera_id}")

    s3_uri = get_model_artifact(
        args.model_group, args.region, args.camera_id,
        version=args.model_version, package_arn=args.package_arn,
        allow_unapproved=args.allow_unapproved,
    )
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
