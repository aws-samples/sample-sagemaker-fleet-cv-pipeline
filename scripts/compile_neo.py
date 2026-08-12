#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Ad-hoc SageMaker Neo compilation for additional edge targets.

The pipeline itself compiles every quality-gated model automatically (see
src/model/compile_neo_step.py, controlled by the NeoTargetDevice /
NeoTargetOs / NeoTargetArch pipeline parameters). Use this script to compile
an ALREADY-REGISTERED model for further device targets after the fact,
without retraining -- for example when a new device type joins the fleet.

Pulls the latest Approved model package for a camera from the Model Registry,
repackages the artifact so it contains only the ONNX model (Neo rejects
tarballs with extra members such as model.pt or metrics.json), uploads it,
runs a Neo compilation job, and prints the compiled artifact S3 URI.

Usage:
    # Compile the anomaly detector for x86_64 Linux (runs on any x86 host)
    python scripts/compile_neo.py --model-group sm1k-anomaly-model-group --type anomalib

    # Compile the classifier for a Jetson Xavier
    python scripts/compile_neo.py --model-group sm1k-classifier-model-group \\
        --type classifier --target-device jetson_xavier

    # Fleet usage: compile a specific camera's model
    python scripts/compile_neo.py --model-group sm1k-classifier-model-group \\
        --type classifier --camera-id cam-014
"""
import argparse
import json
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

# Input tensor shapes must be concrete for Neo; the exported models use a
# dynamic batch axis, so batch is pinned to 1 at compilation time.
INPUT_SHAPES = {
    "anomalib": {"input": [1, 3, 256, 256]},
    "classifier": {"input": [1, 3, 224, 224]},
}
POLL_INTERVAL_SECONDS = 20
MAX_COMPILE_RUNTIME_SECONDS = 900


def get_latest_model_artifact(model_group: str, region: str, camera_id: str) -> str:
    """Return S3 URI of the latest Approved model artifact for *camera_id*.

    ListModelPackages cannot filter on customer metadata, so walk
    newest-first and describe each package until the CameraId matches.
    """
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
            if metadata.get("CameraId") == camera_id:
                return desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    raise RuntimeError(
        f"No Approved model package in {model_group} with CameraId={camera_id} "
        f"in region {region}. Pass --camera-id matching the CameraId parameter "
        "of the pipeline run, and confirm --region matches where the pipeline ran."
    )


def repackage_onnx_only(model_s3_uri: str, dest_s3_uri: str, region: str) -> None:
    """Extract model.onnx from the training artifact and upload it alone.

    Training jobs emit model.tar.gz containing model.pt, model.onnx and
    metrics.json together; Neo requires the input tarball to contain only
    the model file.
    """
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
) -> str:
    """Create a Neo compilation job, poll to completion, return artifact URI."""
    sm = boto3.client("sagemaker", region_name=region)
    output_config = {"S3OutputLocation": output_s3_prefix}
    if target_device == "x86_64_linux":
        output_config["TargetPlatform"] = {"Os": "LINUX", "Arch": "X86_64"}
    else:
        output_config["TargetDevice"] = target_device

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
        time.sleep(POLL_INTERVAL_SECONDS)


def resolve_execution_role(region: str) -> str:
    """Read the pipeline execution role ARN from the deployed stack output."""
    cfn = boto3.client("cloudformation", region_name=region)
    stacks = cfn.describe_stacks(StackName="SageMakerCVPipelineStack")["Stacks"]
    outputs = stacks[0].get("Outputs", [])
    for output in outputs:
        if output["OutputKey"] == "CVExecutionRoleArn":
            return output["OutputValue"]
    raise RuntimeError(
        "CVExecutionRoleArn output not found on SageMakerCVPipelineStack in "
        f"{region}; found outputs: {[o['OutputKey'] for o in outputs]}. "
        "Deploy the stack first (see README)."
    )


def main():
    parser = argparse.ArgumentParser(description="Compile a registered model with SageMaker Neo")
    parser.add_argument("--model-group", required=True, help="Model package group name")
    parser.add_argument("--type", required=True, choices=["anomalib", "classifier"])
    parser.add_argument("--camera-id", default="default",
                        help="Fleet device identity; selects the newest Approved package "
                             "registered with matching CameraId metadata")
    parser.add_argument("--target-device", default="x86_64_linux",
                        help="Neo target: 'x86_64_linux' (default, runs on any x86 host) "
                             "or a TargetDevice value such as jetson_xavier, rasp4b")
    parser.add_argument("--region", default=boto3.session.Session().region_name)
    args = parser.parse_args()

    if not args.region:
        raise RuntimeError(
            "AWS region could not be resolved from the boto3 session. "
            "Pass --region or configure AWS_REGION/AWS_DEFAULT_REGION."
        )

    model_s3_uri = get_latest_model_artifact(args.model_group, args.region, args.camera_id)
    print(f"Model artifact: {model_s3_uri}")

    bucket = model_s3_uri.split("/")[2]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"neo-{args.type}-{args.camera_id}-{timestamp}"[:63]
    onnx_input_uri = f"s3://{bucket}/neo/{args.camera_id}/{args.type}/input/model-onnx.tar.gz"
    output_prefix = f"s3://{bucket}/neo/{args.camera_id}/{args.type}/compiled/"

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
    )
    print(f"Compiled artifact: {artifact}")


if __name__ == "__main__":
    main()
