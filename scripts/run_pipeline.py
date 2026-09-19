#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Start or resume monitoring an execution and verify its registered and compiled models."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aws_env import resolve_region  # noqa: E402
from config import ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP, PIPELINE_NAME  # noqa: E402


def read_s3_json(s3, uri):
    location = urlparse(uri)
    return json.loads(s3.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))["Body"].read())


def verify_execution(sm, s3, arn):
    """Bind evidence to this execution, then ask AWS about every resulting artifact."""
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    if execution["PipelineExecutionStatus"] != "Succeeded":
        raise RuntimeError(f"Execution did not succeed: {execution}")
    steps = [
        step
        for page in sm.get_paginator("list_pipeline_execution_steps").paginate(PipelineExecutionArn=arn)
        for step in page["PipelineExecutionSteps"]
    ]
    if not steps or any(step["StepStatus"] != "Succeeded" for step in steps):
        raise RuntimeError("Not every execution step succeeded.")
    packages, compilations, training_jobs = [], [], []
    for step in steps:
        metadata = step.get("Metadata", {})
        if "TrainingJob" in metadata:
            job_name = metadata["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
            job = sm.describe_training_job(TrainingJobName=job_name)
            if job["TrainingJobStatus"] != "Completed" or job["SecondaryStatus"] != "Completed":
                raise RuntimeError(f"Training did not complete normally: {job_name}")
            training_jobs.append({
                "name": job_name, "status": job["TrainingJobStatus"],
                "secondary_status": job["SecondaryStatus"],
                "billable_seconds": job.get("BillableTimeInSeconds"),
            })
        if "RegisterModel" in metadata:
            package_arn = metadata["RegisterModel"]["Arn"]
            package = sm.describe_model_package(ModelPackageName=package_arn)
            if (package["ModelApprovalStatus"] != "Approved"
                    or package["ModelPackageStatus"] != "Completed"):
                raise RuntimeError(f"Package is not Approved and Completed: {package_arn}")
            packages.append({
                "arn": package_arn,
                "group": package["ModelPackageGroupName"],
                "version": package["ModelPackageVersion"],
                "approval": package["ModelApprovalStatus"],
            })
        if step["StepName"] in ("CompileAnomalibNeo", "CompileClassifierNeo"):
            job_name = metadata["ProcessingJob"]["Arn"].rsplit("/", 1)[-1]
            processing = sm.describe_processing_job(ProcessingJobName=job_name)
            outputs = processing["ProcessingOutputConfig"]["Outputs"]
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one compilation output from {job_name}")
            prefix = outputs[0]["S3Output"]["S3Uri"].rstrip("/")
            result = read_s3_json(s3, prefix + "/compilation_result.json")
            compiled = sm.describe_compilation_job(CompilationJobName=result["compilation_job_name"])
            if compiled["CompilationJobStatus"] != "COMPLETED":
                raise RuntimeError(f"Neo compilation is not complete: {compiled['CompilationJobName']}")
            uri = compiled["ModelArtifacts"]["S3ModelArtifacts"]
            if uri != result["compiled_artifact_s3_uri"]:
                raise RuntimeError("Compilation output disagrees with the pipeline result.")
            location = urlparse(uri)
            artifact = s3.head_object(Bucket=location.netloc, Key=location.path.lstrip("/"))
            if artifact["ContentLength"] <= 0:
                raise RuntimeError(f"Empty compiled artifact: {uri}")
            compilations.append({
                "job": compiled["CompilationJobName"],
                "model_type": result["model_type"],
                "artifact": uri,
                "bytes": artifact["ContentLength"],
                "target": result["output_config"],
            })
    if len(packages) != 2 or {p["group"] for p in packages} != {ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP}:
        raise RuntimeError("This execution did not register exactly one package in each model group.")
    if len(compilations) != 2 or {c["model_type"] for c in compilations} != {"anomalib", "classifier"}:
        raise RuntimeError("This execution did not produce both Neo artifacts.")
    return {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "execution_arn": arn,
        "status": execution["PipelineExecutionStatus"],
        "steps": [{"name": s["StepName"], "status": s["StepStatus"]} for s in steps],
        "packages": packages,
        "compilations": compilations,
        "training_jobs": training_jobs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region")
    parser.add_argument("--camera-id", default="default")
    parser.add_argument("--data-s3-uri", help="Published dataset prefix; omit for synthetic data")
    parser.add_argument("--execution-arn", help="Monitor an existing execution without starting another")
    parser.add_argument("--output", type=Path, default=Path("execution.json"))
    parser.add_argument("--timeout-minutes", type=float, default=90)
    args = parser.parse_args()
    if args.timeout_minutes <= 0:
        parser.error("--timeout-minutes must be positive")
    region = resolve_region(args.region)
    sm, s3 = (boto3.client(service, region_name=region) for service in ("sagemaker", "s3"))
    arn = args.execution_arn
    if arn is None:
        parameters = [{"Name": "CameraId", "Value": args.camera_id}]
        if args.data_s3_uri:
            parameters.append({"Name": "TrainingDataS3Uri", "Value": args.data_s3_uri})
        arn = sm.start_pipeline_execution(
            PipelineName=PIPELINE_NAME, PipelineParameters=parameters,
        )["PipelineExecutionArn"]
    # Save identity immediately so a terminal disconnect never requires another start.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"execution_arn": arn}, indent=2) + "\n")
    print(f"Execution: {arn}", flush=True)
    deadline = time.monotonic() + args.timeout_minutes * 60
    while True:
        execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
        status = execution["PipelineExecutionStatus"]
        print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {status}", flush=True)
        if status in ("Succeeded", "Failed", "Stopped"):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Still running. Resume with --execution-arn {arn}; no new execution is needed.")
        time.sleep(30)
    evidence = verify_execution(sm, s3, arn)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    print(f"Verified execution evidence saved to {args.output}")


if __name__ == "__main__":
    main()
