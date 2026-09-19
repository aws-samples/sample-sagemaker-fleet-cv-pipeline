# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Pipeline step: compile the quality-gated ONNX model with SageMaker Neo.'
import argparse
import hashlib
import json
import os
import logging
import tarfile
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# Compilation job identity is shared with the lease manager, which has to reconstruct
# these names to find a child job the driver may have orphaned. The module is
# stdlib-only and is copied into this container by docker/training/Dockerfile.
from neo_jobs import (  # noqa: E402
    MAX_ATTEMPTS,
    deterministic_job_name,
    submission_key,
    submission_record,
)

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


def resolve_attempt(sm, execution_id: str, model_type: str,
                    max_attempts: int = MAX_ATTEMPTS) -> str:
    'The job name this step should use now.'
    for attempt in range(1, max_attempts + 1):
        name = deterministic_job_name(execution_id, model_type, attempt)
        try:
            status = sm.describe_compilation_job(
                CompilationJobName=name
            )["CompilationJobStatus"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in (
                "ResourceNotFound", "ValidationException"
            ):
                logger.info("Using compilation job name %s (attempt %d)",
                            name, attempt)
                return name
            raise
        if status in ("FAILED", "STOPPED"):
            logger.warning(
                "Compilation job %s is %s; advancing to the next attempt rather "
                "than resuming a terminal job", name, status,
            )
            continue
        logger.info("Compilation job %s exists with status %s; resuming it",
                    name, status)
        return name

    raise RuntimeError(
        f"All {max_attempts} compilation attempts for {model_type} in execution "
        f"{execution_id} have failed. The model export or the target configuration "
        f"is the likely cause; inspect the failure reason of "
        f"{deterministic_job_name(execution_id, model_type, max_attempts)} rather "
        f"than retrying again."
    )


def build_output_config(bucket: str, output_prefix: str, target_device: str,
                        target_os: str, target_arch: str) -> dict:
    'OutputConfig for either a named device target or an OS/arch platform.'
    config = {"S3OutputLocation": f"s3://{bucket}/{output_prefix}"}
    if target_device and target_device != "none":
        config["TargetDevice"] = target_device
    else:
        config["TargetPlatform"] = {"Os": target_os, "Arch": target_arch}
    return config


def _request_digest(job_name: str, create_kwargs: dict) -> str:
    'A stable digest of the exact create request this submission represents.'
    canonical = json.dumps({"CompilationJobName": job_name, **create_kwargs},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def submit_or_resume(sm, job_name: str, create_kwargs: dict) -> None:
    'Create the compilation job, or resume if a retry already created it.'
    try:
        sm.create_compilation_job(CompilationJobName=job_name, **create_kwargs)
        logger.info("Compilation job started: %s", job_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", "")
        if code == "ResourceInUse" or "already" in message.lower():
            logger.info("Compilation job %s already exists; resuming poll", job_name)
            return
        logger.error(
            "Creating compilation job %s failed (%s). The submission intent stands and the "
            "lease is retained: this failure does NOT establish that no job was created, "
            "because the SDK may have already delivered an earlier attempt. Recover with the "
            "documented compilation-submission adjudication.", job_name, code or "unknown",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Neo compilation pipeline step")
    parser.add_argument("--region", required=True)
    parser.add_argument("--role-arn", required=True,
                        help="Execution role Neo assumes for S3 input/output access")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--execution-id", required=True,
                        help="Pipeline execution ID; scopes S3 prefixes and job name")
    parser.add_argument("--runs-prefix", required=True,
                        help="Shared S3 run prefix (config.RUNS_PREFIX)")
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

    run_prefix = f"{args.runs_prefix.strip('/')}/{args.execution_id}/neo/{args.model_type}"
    input_key = f"{run_prefix}/input/model-onnx.tar.gz"
    output_prefix = f"{run_prefix}/compiled/"
    s3.upload_file(str(onnx_tar), args.bucket, input_key)
    logger.info("Uploaded ONNX-only input: s3://%s/%s", args.bucket, input_key)

    # Resolve the attempt against real job state rather than assuming attempt 1:
    # a terminal failure must not be resumed forever.
    job_name = resolve_attempt(sm, args.execution_id, args.model_type)

    data_input_config = json.dumps({"input": [1, 3, args.image_size, args.image_size]})
    output_config = build_output_config(
        args.bucket, output_prefix, args.target_device, args.target_os, args.target_arch
    )
    create_kwargs = {
        "RoleArn": args.role_arn,
        "InputConfig": {
            "S3Uri": f"s3://{args.bucket}/{input_key}",
            "DataInputConfig": data_input_config,
            "Framework": "ONNX",
        },
        "OutputConfig": output_config,
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_COMPILE_RUNTIME_SECONDS},
    }

    submission_id = str(uuid.uuid4())
    key = submission_key(args.runs_prefix, args.execution_id, args.model_type, submission_id)
    record = submission_record(
        submission_id=submission_id,
        execution_id=args.execution_id,
        model_type=args.model_type,
        job_name=job_name,
        request_digest=_request_digest(job_name, create_kwargs),
        recorded_at=int(time.time()),
        producer=os.environ.get("TRAINING_JOB_NAME") or os.environ.get("HOSTNAME"),
    )
    s3.put_object(Bucket=args.bucket, Key=key,
                  Body=json.dumps(record, indent=2, sort_keys=True).encode())
    logger.info("Recorded submission intent %s for %s at s3://%s/%s",
                submission_id, job_name, args.bucket, key)

    submit_or_resume(sm, job_name, create_kwargs)

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
