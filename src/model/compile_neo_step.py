# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Pipeline step: compile the quality-gated ONNX model with SageMaker Neo.

Runs as a ProcessingStep nested under each quality gate, ordered before
registration (gate -> compile -> register), so only models that both pass
their quality gate AND compile for the edge target become Approved packages.
The training artifact is mounted as a processing input; the script extracts
the ONNX model (Neo rejects tarballs with extra members such as model.pt or
metrics.json), uploads an ONNX-only tarball, submits the compilation job,
polls to a terminal state, and writes the compiled artifact URI to the step
output. A FAILED compilation fails this step, blocking registration, loudly.

Retries are idempotent: the compilation job name is derived from the pipeline
execution ID and model type only, so a retried step resumes polling the job
it already started instead of orphaning it.
"""
import argparse
import json
import logging
import re
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_MODEL_DIR = Path("/opt/ml/processing/input/model")
OUTPUT_DIR = Path("/opt/ml/processing/output/compiled")
POLL_INTERVAL_SECONDS = 20
MAX_COMPILE_RUNTIME_SECONDS = 900


def repackage_onnx_only(work_dir: Path) -> Path:
    """Extract model.onnx from the mounted training artifact, tar it alone."""
    tarballs = list(INPUT_MODEL_DIR.rglob("*.tar.gz"))
    if not tarballs:
        raise RuntimeError(f"No model tarball found under {INPUT_MODEL_DIR}")
    with tarfile.open(tarballs[0], "r:gz") as tar:
        tar.extract("model.onnx", work_dir, filter="data")
    onnx_tar = work_dir / "model-onnx.tar.gz"
    with tarfile.open(onnx_tar, "w:gz") as tar:
        tar.add(work_dir / "model.onnx", arcname="model.onnx")
    return onnx_tar


def deterministic_job_name(execution_id: str, model_type: str) -> str:
    """Compilation job name stable across retries of the same step.

    Derived only from the pipeline execution ID and model type, so a retried
    processing step computes the same name and resumes the job it already
    started. Sanitized to the [a-zA-Z0-9-]{1,63} constraint of the API.
    """
    raw = f"neo-{model_type}-{execution_id}"
    return re.sub(r"[^a-zA-Z0-9-]", "-", raw)[:63].rstrip("-")


def build_output_config(bucket: str, output_prefix: str, target_device: str,
                        target_os: str, target_arch: str) -> dict:
    """OutputConfig for either a named device target or an OS/arch platform.

    TargetDevice and TargetPlatform are mutually exclusive in the API;
    a non-'none' device wins.
    """
    config = {"S3OutputLocation": f"s3://{bucket}/{output_prefix}"}
    if target_device and target_device != "none":
        config["TargetDevice"] = target_device
    else:
        config["TargetPlatform"] = {"Os": target_os, "Arch": target_arch}
    return config


def submit_or_resume(sm, job_name: str, create_kwargs: dict) -> None:
    """Create the compilation job, or resume if a retry already created it."""
    try:
        sm.create_compilation_job(CompilationJobName=job_name, **create_kwargs)
        logger.info("Compilation job started: %s", job_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", "")
        if code == "ResourceInUse" or "already" in message.lower():
            logger.info("Compilation job %s already exists; resuming poll", job_name)
            return
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Neo compilation pipeline step")
    parser.add_argument("--region", required=True)
    parser.add_argument("--role-arn", required=True,
                        help="Execution role Neo assumes for S3 input/output access")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--execution-id", required=True,
                        help="Pipeline execution ID; scopes S3 prefixes and job name")
    parser.add_argument("--model-type", required=True, choices=["anomalib", "classifier"])
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--target-device", default="none",
                        help="Neo TargetDevice (e.g. jetson_xavier, rasp4b); "
                             "'none' compiles for --target-os/--target-arch instead")
    parser.add_argument("--target-os", default="LINUX")
    parser.add_argument("--target-arch", default="X86_64")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_tar = repackage_onnx_only(OUTPUT_DIR)

    input_key = (
        f"cv-pipeline/runs/{args.execution_id}/neo/{args.model_type}/input/model-onnx.tar.gz"
    )
    output_prefix = f"cv-pipeline/runs/{args.execution_id}/neo/{args.model_type}/compiled/"
    s3.upload_file(str(onnx_tar), args.bucket, input_key)
    logger.info("Uploaded ONNX-only input: s3://%s/%s", args.bucket, input_key)

    job_name = deterministic_job_name(args.execution_id, args.model_type)
    # Neo requires concrete input shapes; the export uses a dynamic batch
    # axis, so batch is pinned to 1 at compilation time.
    data_input_config = json.dumps({"input": [1, 3, args.image_size, args.image_size]})
    output_config = build_output_config(
        args.bucket, output_prefix, args.target_device, args.target_os, args.target_arch
    )

    submit_or_resume(
        sm,
        job_name,
        {
            "RoleArn": args.role_arn,
            "InputConfig": {
                "S3Uri": f"s3://{args.bucket}/{input_key}",
                "DataInputConfig": data_input_config,
                "Framework": "ONNX",
            },
            "OutputConfig": output_config,
            "StoppingCondition": {"MaxRuntimeInSeconds": MAX_COMPILE_RUNTIME_SECONDS},
        },
    )

    deadline = time.time() + MAX_COMPILE_RUNTIME_SECONDS + 300
    while True:
        desc = sm.describe_compilation_job(CompilationJobName=job_name)
        status = desc["CompilationJobStatus"]
        logger.info("Compilation status: %s", status)
        if status == "COMPLETED":
            artifact_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
            break
        if status in ("FAILED", "STOPPED"):
            raise RuntimeError(
                f"Compilation job {job_name} ended with status {status}: "
                f"{desc.get('FailureReason', 'no failure reason reported')}"
            )
        if time.time() > deadline:
            raise RuntimeError(f"Timed out waiting for compilation job {job_name}")
        time.sleep(POLL_INTERVAL_SECONDS)

    result = {
        "compilation_job_name": job_name,
        "compiled_artifact_s3_uri": artifact_uri,
        "camera_id": args.camera_id,
        "model_type": args.model_type,
        "output_config": {k: v for k, v in output_config.items() if k != "S3OutputLocation"},
    }
    result_path = OUTPUT_DIR / "compilation_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    # The raw ONNX working copies must not upload as step output artifacts.
    (OUTPUT_DIR / "model.onnx").unlink()
    onnx_tar.unlink()
    logger.info("Compiled artifact: %s", artifact_uri)


if __name__ == "__main__":
    main()
