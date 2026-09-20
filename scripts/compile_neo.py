#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Ad-hoc SageMaker Neo compilation for additional edge targets.'
import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aws_env import AwsEnvError, resolve_region  # noqa: E402
from config import (  # noqa: E402
    ANOMALY_MODEL_GROUP,
    CLASSIFIER_MODEL_GROUP,
    OUTPUT_EXECUTION_ROLE_ARN,
    PIPELINE_STACK_NAME,
)

# Input tensor shapes must be concrete for Neo; the exported models use a
# dynamic batch axis, so batch is pinned to 1 at compilation time.
INPUT_SHAPES = {
    "anomalib": {"input": [1, 3, 256, 256]},
    "classifier": {"input": [1, 3, 224, 224]},
}
POLL_INTERVAL_SECONDS = 20
MAX_COMPILE_RUNTIME_SECONDS = 900


def get_latest_model_artifact(model_group: str, region: str, camera_id: str) -> str:
    'Return S3 URI of the latest Approved model artifact for *camera_id*.'
    sm = boto3.client("sagemaker", region_name=region)
    paginator = sm.get_paginator("list_model_packages")
    pages = paginator.paginate(
        ModelPackageGroupName=model_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    for page in pages:
        for summary in page["ModelPackageSummaryList"]:
            desc = sm.describe_model_package(
                ModelPackageName=summary["ModelPackageArn"]
            )
            metadata = desc.get("CustomerMetadataProperties", {})
            if metadata.get("CameraId") != camera_id:
                continue
            # Re-checked on the description: the list filter reflects the moment
            # the page was produced, so a package rejected in between would
            # otherwise be compiled and shipped to a device.
            if (desc.get("ModelApprovalStatus") != "Approved"
                    or desc.get("ModelPackageStatus") != "Completed"):
                print(
                    f"  skipping {summary['ModelPackageArn']}: approval="
                    f"{desc.get('ModelApprovalStatus')!r} status="
                    f"{desc.get('ModelPackageStatus')!r}"
                )
                continue
            return desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    raise RuntimeError(
        f"No Approved model package in {model_group} with CameraId={camera_id} "
        f"in region {region}. Pass --camera-id matching the CameraId parameter "
        "of the pipeline run, and confirm --region matches where the pipeline ran."
    )


def repackage_onnx_only(model_s3_uri: str, dest_s3_uri: str, region: str) -> None:
    'Extract model.onnx from the training artifact and upload it alone.'
    s3 = boto3.client("s3", region_name=region)
    src_bucket, src_key = model_s3_uri.split("/", 3)[2], model_s3_uri.split("/", 3)[3]
    dest_bucket, dest_key = dest_s3_uri.split("/", 3)[2], dest_s3_uri.split("/", 3)[3]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tar_path = tmp / "model.tar.gz"
        s3.download_file(src_bucket, src_key, str(tar_path))
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extract("model.onnx", tmp, filter="data")
        onnx_tar = tmp / "model-onnx.tar.gz"
        with tarfile.open(onnx_tar, "w:gz") as tar:
            tar.add(tmp / "model.onnx", arcname="model.onnx")
        s3.upload_file(str(onnx_tar), dest_bucket, dest_key)


def run_compilation_job(
    job_name: str,
    input_s3_uri: str,
    output_s3_prefix: str,
    data_input_config: str,
    role_arn: str,
    region: str,
    target_device: str,
    target_os: str,
    target_arch: str,
) -> str:
    """Create a Neo compilation job, poll to completion, return artifact URI."""
    sm = boto3.client("sagemaker", region_name=region)
    # TargetDevice and TargetPlatform are mutually exclusive in the API.
    output_config = {"S3OutputLocation": output_s3_prefix}
    if target_device and target_device != "none":
        output_config["TargetDevice"] = target_device
    else:
        output_config["TargetPlatform"] = {"Os": target_os, "Arch": target_arch}

    sm.create_compilation_job(
        CompilationJobName=job_name,
        RoleArn=role_arn,
        InputConfig={
            "S3Uri": input_s3_uri,
            "DataInputConfig": data_input_config,
            "Framework": "ONNX",
        },
        OutputConfig=output_config,
        StoppingCondition={"MaxRuntimeInSeconds": MAX_COMPILE_RUNTIME_SECONDS},
    )
    print(f"Compilation job started: {job_name}")

    # Bounded, as the in-pipeline compiler already is: an unbounded loop would
    # hang indefinitely if a job never reached a terminal state.
    deadline = time.monotonic() + MAX_COMPILE_RUNTIME_SECONDS + 300
    while True:
        desc = sm.describe_compilation_job(CompilationJobName=job_name)
        status = desc["CompilationJobStatus"]
        print(f"  status: {status}")
        if status == "COMPLETED":
            return desc["ModelArtifacts"]["S3ModelArtifacts"]
        if status in ("FAILED", "STOPPED"):
            raise RuntimeError(
                f"Compilation job {job_name} ended with status {status}: "
                f"{desc.get('FailureReason', 'no failure reason reported')}"
            )
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Timed out waiting for compilation job {job_name} (last status "
                f"{status}). The job may still be running; check it with "
                f"'aws sagemaker describe-compilation-job --compilation-job-name "
                f"{job_name}'."
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def resolve_execution_role(region: str) -> str:
    """Read the pipeline execution role ARN from the deployed stack output."""
    cfn = boto3.client("cloudformation", region_name=region)
    stacks = cfn.describe_stacks(StackName=PIPELINE_STACK_NAME)["Stacks"]
    outputs = stacks[0].get("Outputs", [])
    for output in outputs:
        if output["OutputKey"] == OUTPUT_EXECUTION_ROLE_ARN:
            return output["OutputValue"]
    raise RuntimeError(
        f"{OUTPUT_EXECUTION_ROLE_ARN} output not found on {PIPELINE_STACK_NAME} in "
        f"{region}; found outputs: {[o['OutputKey'] for o in outputs]}. "
        "Deploy the stack first (see README)."
    )


def main():
    parser = argparse.ArgumentParser(description="Compile a registered model with SageMaker Neo")
    parser.add_argument("--model-group", required=True,
                        choices=[ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP],
                        help="Model package group name")
    parser.add_argument("--type", required=True, choices=["anomalib", "classifier"])
    parser.add_argument("--camera-id", default="default",
                        help="Fleet device identity; selects the newest Approved package "
                             "registered with matching CameraId metadata")
    parser.add_argument("--target-device", default="none",
                        help="Neo TargetDevice value. 'none' (default) compiles for "
                             "--target-os/--target-arch instead of a named device.")
    parser.add_argument("--target-os", default="LINUX",
                        help="Neo TargetPlatform OS, used when --target-device is 'none'")
    parser.add_argument("--target-arch", default="X86_64",
                        help="Neo TargetPlatform architecture, used when --target-device is 'none'")
    parser.add_argument("--region", default=None,
                        help="AWS Region (defaults to the shared resolution in aws_env.py)")
    args = parser.parse_args()

    try:
        args.region = resolve_region(args.region)
    except AwsEnvError as exc:
        raise SystemExit(str(exc))

    model_s3_uri = get_latest_model_artifact(args.model_group, args.region, args.camera_id)
    print(f"Model artifact: {model_s3_uri}")

    bucket = model_s3_uri.split("/")[2]
    attempt = hashlib.sha256(
        f"{model_s3_uri}|{args.type}|{args.camera_id}|{args.target_device}|"
        f"{args.target_os}|{args.target_arch}|{time.time_ns()}".encode()
    ).hexdigest()[:12]
    safe_camera = re.sub(r"[^a-zA-Z0-9-]", "-", args.camera_id)[:20].strip("-") or "cam"
    job_name = f"neo-{args.type}-{safe_camera}-{attempt}"[:63].rstrip("-")
    prefix = f"neo/{args.camera_id}/{args.type}/{attempt}"
    onnx_input_uri = f"s3://{bucket}/{prefix}/input/model-onnx.tar.gz"
    output_prefix = f"s3://{bucket}/{prefix}/compiled/"

    repackage_onnx_only(model_s3_uri, onnx_input_uri, args.region)
    print(f"ONNX-only input: {onnx_input_uri}")

    artifact = run_compilation_job(
        job_name=job_name,
        input_s3_uri=onnx_input_uri,
        output_s3_prefix=output_prefix,
        data_input_config=json.dumps(INPUT_SHAPES[args.type]),
        role_arn=resolve_execution_role(args.region),
        region=args.region,
        target_device=args.target_device,
        target_os=args.target_os,
        target_arch=args.target_arch,
    )
    print(f"Compiled artifact: {artifact}")


if __name__ == "__main__":
    main()
