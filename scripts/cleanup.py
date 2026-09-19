#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Delete the model packages this sample registered, in the right order.'
import argparse
import shlex
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestration"))

from aws_env import AwsEnvError, resolve_region  # noqa: E402
from neo_jobs import (  # noqa: E402
    NEO_JOB_PREFIX,
    TERMINAL_COMPILATION_STATUSES,
)
from config import (  # noqa: E402
    ANOMALY_MODEL_GROUP,
    CLASSIFIER_MODEL_GROUP,
    ORCHESTRATION_STACK_NAME,
    OUTPUT_LEASE_TABLE_NAME,
    OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME,
    OUTPUT_QUEUE_URL,
    OUTPUT_STATE_MACHINE_ARN,
    PIPELINE_NAME,
    PIPELINE_STACK_NAME,
    TRAINING_REPO_NAME,
    bucket_name,
)

ADMISSION_CLOSED_ID = "control#admission-closed"

MODEL_GROUPS = (ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP)


def list_packages(sm, group: str) -> list:
    """Every package ARN in *group*, via the paginator (not a one-page read)."""
    arns = []
    paginator = sm.get_paginator("list_model_packages")
    try:
        for page in paginator.paginate(ModelPackageGroupName=group):
            arns += [s["ModelPackageArn"] for s in page["ModelPackageSummaryList"]]
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ValidationException":
            print(f"  {group}: not found in this Region, skipping")
            return []
        raise
    return arns


def report_inflight(sm, region: str) -> int:
    """Report non-terminal pipeline executions. Returns how many are live."""
    try:
        # Paginated: reading only the first page reported zero live executions
        # while a later page held running ones, which is the opposite of what a
        # safety check must do.
        summaries = []
        paginator = sm.get_paginator("list_pipeline_executions")
        for page in paginator.paginate(PipelineName=PIPELINE_NAME):
            summaries += page["PipelineExecutionSummaries"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("ResourceNotFound", "ValidationException"):
            print(f"Pipeline {PIPELINE_NAME} not found in {region}; nothing in flight.")
            return 0
        raise
    live = [s for s in summaries if s["PipelineExecutionStatus"] not in ("Succeeded", "Failed", "Stopped")]
    if live:
        print(f"\n{len(live)} pipeline execution(s) are still running:")
        for s in live:
            print(f"  {s['PipelineExecutionStatus']:10} {s['PipelineExecutionArn']}")
        print(
            "\nThese can register new packages after this script lists them, and may\n"
            "reference artifacts it deletes. Let them finish, or stop them with\n"
            "  aws sagemaker stop-pipeline-execution --pipeline-execution-arn <arn>\n"
            "and re-run cleanup after they finish."
        )
    return len(live)


def report_admitted_work(region: str) -> int:
    'Report work that is admitted but has not started a pipeline yet.'
    total = 0
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        outputs = {
            o["OutputKey"]: o["OutputValue"]
            for o in cfn.describe_stacks(StackName=ORCHESTRATION_STACK_NAME)["Stacks"][0]
            .get("Outputs", [])
        }
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("ValidationError", "ResourceNotFoundException"):
            print(f"{ORCHESTRATION_STACK_NAME} not deployed in {region}; "
                  f"no queued or waiting work.")
            return 0
        raise

    queue_url = outputs.get(OUTPUT_QUEUE_URL)
    if queue_url:
        sqs = boto3.client("sqs", region_name=region)
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible"],
        )["Attributes"]
        visible = int(attrs.get("ApproximateNumberOfMessages", 0))
        hidden = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
        if visible or hidden:
            total += visible + hidden
            print(f"\n{visible + hidden} message(s) still on the training queue "
                  f"({visible} visible, {hidden} in flight).")
            print("  Each can start a pipeline execution. Disable the queue's event "
                  "source\n  mapping and let them drain, or purge the queue.")

    sm_arn = outputs.get(OUTPUT_STATE_MACHINE_ARN)
    if sm_arn:
        sfn = boto3.client("stepfunctions", region_name=region)
        running = []
        paginator = sfn.get_paginator("list_executions")
        for page in paginator.paginate(stateMachineArn=sm_arn, statusFilter="RUNNING"):
            running += page["executions"]
        if running:
            total += len(running)
            print(f"\n{len(running)} orchestration execution(s) are still running:")
            for execution in running[:10]:
                print(f"  {execution['name']}")
            if len(running) > 10:
                print(f"  ... and {len(running) - 10} more")
            print("  An execution waiting for a slot has not started its pipeline "
                  "yet, so it\n  does not appear in the list above -- but it will "
                  "start one when a slot frees.")
    return total


class DeploymentUnresolved(RuntimeError):
    'The deployment could not be resolved, so nothing about it is established.'


def resolve_deployment(region: str) -> dict:
    "Read the orchestration stack's outputs ONCE."
    stack = stack_description(region, ORCHESTRATION_STACK_NAME)
    if stack is None:
        return {}

    outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}
    missing = [
        name for name in (
            OUTPUT_LEASE_TABLE_NAME,
            OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME,
            OUTPUT_QUEUE_URL,
        )
        if not outputs.get(name)
    ]
    if missing:
        raise DeploymentUnresolved(
            f"{ORCHESTRATION_STACK_NAME} in {region} does not publish {missing}. A "
            f"deployment without lease outputs cannot be safely drained. See docs/fleet.md."
        )
    return outputs


def close_admission(region: str, table: str) -> None:
    'Raise the admission barrier so no NEW slot can be acquired.'
    dynamodb = boto3.client("dynamodb", region_name=region)
    dynamodb.put_item(
        TableName=table,
        Item={
            "LeaseId": {"S": ADMISSION_CLOSED_ID},
            "CreatedAt": {"N": str(int(time.time()))},
            "ReassertedAt": {"N": str(int(time.time()))},
            "Reason": {"S": "scripts/cleanup.py"},
        },
    )
    print("\nAdmission CLOSED. No new pipeline can be admitted.")
    print("  This is NOT reopened automatically, including if this script fails.")
    print("  To reopen:  see docs/fleet.md")


def disable_queue_consumption(region: str, outputs: dict, wait_seconds: int = 120) -> None:
    'Disable the event source mapping for THIS queue and this function, and wait.'
    sqs = boto3.client("sqs", region_name=region)
    queue_arn = sqs.get_queue_attributes(
        QueueUrl=outputs[OUTPUT_QUEUE_URL], AttributeNames=["QueueArn"]
    )["Attributes"]["QueueArn"]

    function_name = outputs[OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME]
    lam = boto3.client("lambda", region_name=region)
    mappings = [
        m for m in lam.list_event_source_mappings(FunctionName=function_name)
        .get("EventSourceMappings", [])
        if m.get("EventSourceArn") == queue_arn
    ]
    if not mappings:
        raise DeploymentUnresolved(
            f"No event source mapping links {function_name} to {queue_arn}. Without "
            f"one this script cannot stop consumption, so it cannot establish that "
            f"the deployment will stay idle."
        )
    if len(mappings) > 1:
        raise DeploymentUnresolved(
            f"{len(mappings)} event source mappings link {function_name} to "
            f"{queue_arn}: {[m['UUID'] for m in mappings]}. Disabling one of several "
            f"would leave the queue being consumed. Resolve this by hand."
        )

    uuid = mappings[0]["UUID"]
    lam.update_event_source_mapping(UUID=uuid, Enabled=False)
    print(f"\nDisabling queue consumption (mapping {uuid})...")

    deadline = time.time() + wait_seconds
    while True:
        state = lam.get_event_source_mapping(UUID=uuid)["State"]
        if state == "Disabled":
            print(f"  mapping {uuid} is Disabled")
            return
        if time.time() >= deadline:
            raise DeploymentUnresolved(
                f"Mapping {uuid} is still {state!r} after {wait_seconds}s. It has "
                f"not stopped consuming, so the deployment is not quiescent."
            )
        time.sleep(5)


def retained_leases(region: str, table: str) -> list:
    'Every ownership record, read CONSISTENTLY and to the last page.'
    dynamodb = boto3.client("dynamodb", region_name=region)
    held, start_key = [], None
    while True:
        kwargs = {
            "TableName": table,
            "FilterExpression": "begins_with(LeaseId, :p)",
            "ExpressionAttributeValues": {":p": {"S": "owner#"}},
            "ConsistentRead": True,
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        try:
            page = dynamodb.scan(**kwargs)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ResourceNotFoundException":
                raise DeploymentUnresolved(
                    f"Lease table {table} does not exist in {region}, so no "
                    f"statement can be made about held leases."
                ) from exc
            raise
        held.extend(page.get("Items", []))
        start_key = page.get("LastEvaluatedKey")
        if not start_key:
            break

    if held:
        print(f"\n{len(held)} lease(s) are still held:")
        for item in held[:10]:
            print(f"  {item['LeaseId']['S']} slot={item.get('Slot', {}).get('N')}")
        if len(held) > 10:
            print(f"  ... and {len(held) - 10} more")
        print("  A held lease can correspond to a running pipeline whose outcome "
              "could not be\n  established. Run the reconcile action (see "
              "docs/fleet.md) first.")
    return held


def active_compilation_jobs(region: str) -> list:
    'Non-terminal Neo compilation jobs this sample could have created.'
    sm = boto3.client("sagemaker", region_name=region)
    active = []
    paginator = sm.get_paginator("list_compilation_jobs")
    for page in paginator.paginate(NameContains=NEO_JOB_PREFIX):
        for summary in page.get("CompilationJobSummaries", []):
            name = summary["CompilationJobName"]
            status = summary["CompilationJobStatus"]
            if not name.startswith(NEO_JOB_PREFIX):
                continue
            if status not in TERMINAL_COMPILATION_STATUSES:
                active.append((name, status))
    if active:
        print(f"\n{len(active)} Neo compilation job(s) are still running:")
        for name, status in active[:10]:
            print(f"  {name} ({status})")
        if len(active) > 10:
            print(f"  ... and {len(active) - 10} more")
        print("  These write to the bucket a destroy would empty. They include the ad-hoc\n"
              "  compiler (scripts/compile_neo.py), which takes no lease, so an empty lease\n"
              "  scan does not account for them. Wait for them, or stop them deliberately.")
    return active


def stack_description(region, name):
    """An absent optional stack is allowed; access failures still stop cleanup."""
    try:
        return boto3.client("cloudformation", region_name=region).describe_stacks(StackName=name)["Stacks"][0]
    except ClientError as exc:
        error = exc.response["Error"]
        if error["Code"] == "ValidationError" and "does not exist" in error.get("Message", ""):
            return None
        raise


def verify_deleted(region):
    """Check stack, bucket, repository, and model group deletion directly with AWS."""
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    remaining = [
        name for name in (ORCHESTRATION_STACK_NAME, PIPELINE_STACK_NAME)
        if stack_description(region, name) is not None
    ]
    probes = [
        ("s3", "head_bucket", {"Bucket": bucket_name(account, region)}, {"404", "NoSuchBucket"}),
        ("ecr", "describe_repositories", {"repositoryNames": [TRAINING_REPO_NAME]}, {"RepositoryNotFoundException"}),
    ]
    for group in MODEL_GROUPS:
        probes.append(("sagemaker", "describe_model_package_group", {"ModelPackageGroupName": group}, {"ValidationException"}))
    for service, operation, arguments, absent_codes in probes:
        try:
            getattr(boto3.client(service, region_name=region), operation)(**arguments)
        except ClientError as exc:
            error = exc.response["Error"]
            missing = error["Code"] in absent_codes
            if error["Code"] == "ValidationException":
                missing = missing and "does not exist" in error.get("Message", "").lower()
            if not missing:
                raise
        else:
            remaining.append(f"{service}: {arguments}")
    if remaining:
        print("Still present:\n  " + "\n  ".join(remaining))
        return 1
    print("Verified: both sample stacks, S3 bucket, ECR repository, and model groups are absent.")
    print("The shared CDK bootstrap stack and SageMaker job history/logs are not deleted.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Read-only inventory of model packages and active work")
    modes.add_argument("--verify-deleted", action="store_true", help="Read-only verification after cdk destroy")
    modes.add_argument("--list-active-compilations", action="store_true")
    modes.add_argument("--close-for-capacity-change", action="store_true", help="Close fleet admission without deleting models")
    modes.add_argument("--preflight-capacity-change", action="store_true", help="Check that closed fleet admission has drained")
    args = parser.parse_args()
    try:
        region = resolve_region(args.region)
        print(f"Region: {region}")
        if args.verify_deleted:
            return verify_deleted(region)
        if args.list_active_compilations:
            return int(bool(active_compilation_jobs(region)))
        # Require the core stack even when the optional fleet extension is absent.
        if stack_description(region, PIPELINE_STACK_NAME) is None:
            raise DeploymentUnresolved(f"{PIPELINE_STACK_NAME} is absent; check the profile and Region, or use --verify-deleted.")
        outputs = resolve_deployment(region)
        if args.close_for_capacity_change or args.preflight_capacity_change:
            if not outputs:
                raise DeploymentUnresolved("The optional fleet stack is not deployed.")
            if args.close_for_capacity_change:
                close_admission(region, outputs[OUTPUT_LEASE_TABLE_NAME])
                disable_queue_consumption(region, outputs)
                print("Fleet admission closed. No model packages deleted.")
                return 0
            held = retained_leases(region, outputs[OUTPUT_LEASE_TABLE_NAME])
            print("This check requires admission to remain closed throughout the capacity change.")
            return int(bool(held))
        sm = boto3.client("sagemaker", region_name=region)
        if args.dry_run:
            if outputs:
                report_admitted_work(region)
            report_inflight(sm, region)
            active_compilation_jobs(region)
            for group in MODEL_GROUPS:
                for arn in list_packages(sm, group):
                    print(f"[dry-run] would delete {arn}")
            print("Dry run: no changes made.")
            return 0
        # Fence queued requests before reading ownership. Direct callers must stop submitting.
        if outputs:
            close_admission(region, outputs[OUTPUT_LEASE_TABLE_NAME])
            disable_queue_consumption(region, outputs)
            if retained_leases(region, outputs[OUTPUT_LEASE_TABLE_NAME]):
                print("Fleet leases remain. Let jobs finish or reconcile them; see docs/fleet.md.")
                return 1
            if report_admitted_work(region):
                print("Queued or waiting work remains. See docs/fleet.md before discarding queued requests.")
                return 1
        if report_inflight(sm, region) or active_compilation_jobs(region):
            print("Work is still active. Wait for it to finish, then rerun cleanup.")
            return 1
        for group in MODEL_GROUPS:
            for arn in list_packages(sm, group):
                sm.delete_model_package(ModelPackageName=arn)
                print(f"Deleted {arn}")
        if any(list_packages(sm, group) for group in MODEL_GROUPS):
            print("Some packages remain; rerun cleanup before destroying the stacks.")
            return 1
        # fleet=true includes the optional stack in destruction even if it was never deployed.
        assignments = " ".join(
            f"{key}={shlex.quote(region)}"
            for key in ("AWS_REGION", "AWS_DEFAULT_REGION", "CDK_DEFAULT_REGION")
        )
        print(f"Model groups are empty. Now run:\n  {assignments} "
              "cdk destroy --all -c bootstrap=true -c fleet=true --force")
        return 0
    except (AwsEnvError, DeploymentUnresolved) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
