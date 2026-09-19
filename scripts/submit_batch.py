# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Submit batch training requests to the orchestration queue.'
import argparse
import hashlib
import json
import re
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aws_env import AwsEnvError, resolve_region  # noqa: E402
from config import (  # noqa: E402
    ORCHESTRATION_STACK_NAME,
    OUTPUT_QUEUE_URL,
    PUBLICATION_INDEX_NAME,
)


# SQS MessageDeduplicationId: up to 128 characters of alphanumerics and punctuation.
SQS_DEDUP_ID_MAX = 128
_SQS_DEDUP_SAFE = re.compile(r"^[A-Za-z0-9!-~]{1,128}$")
_SQS_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def camera_request_id(batch_id: str, camera_id: str) -> str:
    'A stable, unambiguous per-camera request id.'
    payload = f"{len(batch_id)}:{batch_id}:{len(camera_id)}:{camera_id}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    candidate = f"{batch_id}-{camera_id}-{digest}"

    if not _SQS_DEDUP_SAFE.match(candidate) or len(candidate) > SQS_DEDUP_ID_MAX:
        keep = SQS_DEDUP_ID_MAX - len(digest) - 1
        readable = _SQS_UNSAFE.sub("-", f"{batch_id}-{camera_id}")[:keep].strip("-")
        candidate = f"{readable}-{digest}"
    if not _SQS_DEDUP_SAFE.match(candidate) or len(candidate) > SQS_DEDUP_ID_MAX:
        raise SystemExit(
            f"Cannot derive a valid SQS deduplication id for batch {batch_id!r} and "
            f"camera {camera_id!r}: got {candidate!r}. Use a batch id of "
            f"alphanumerics and hyphens."
        )
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Submit batch training requests")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cameras", type=int, help="Number of cameras to train")
    group.add_argument("--camera-ids", nargs="+", help="Specific camera IDs")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    # 400 matches the pipeline default: at 100 the classifier eval split is
    # ~7 images and the balanced-accuracy gate fails on data starvation.
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--publication-index", default=None,
                        help=f"Path to the {PUBLICATION_INDEX_NAME} written by "
                             f"scripts/download_bccd.py. Supplies each camera's exact "
                             f"published dataset URI. Omit to train on synthetic data.")
    parser.add_argument("--queue-url", default=None, help="SQS queue URL (auto-detected if omitted)")
    parser.add_argument("--region", default=None,
                        help="AWS Region (defaults to the shared resolution in aws_env.py)")
    parser.add_argument("--batch-id", default=None,
                        help="Reuse a batch id to RETRY a failed submission without "
                             "creating duplicate training runs. Each camera gets its "
                             "own request id derived from (batch id, camera id), so a "
                             "retry reproduces exactly the same per-camera ids. Omit "
                             "for a new retrain; the generated batch id is printed.")
    parser.add_argument("--request-id", default=None,
                        help=argparse.SUPPRESS)  # removed; see the check below
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.request_id:
        raise SystemExit(
            "--request-id has been replaced by --batch-id. It assigned the SAME id "
            "to every camera, and that id is the SQS FIFO deduplication id: with "
            "the queue's queue-wide deduplication scope, SQS accepted all sends "
            "but suppressed delivery of every message after the first for five "
            "minutes, while this script still reported them all as sent. "
            f"Use --batch-id {args.request_id!r} to reproduce the intended retry; "
            "each camera then gets its own id derived from (batch id, camera id)."
        )

    try:
        args.region = resolve_region(args.region)
    except AwsEnvError as exc:
        raise SystemExit(str(exc))

    camera_ids = args.camera_ids or [f"cam-{i:03d}" for i in range(1, args.cameras + 1)]

    # Resolve every camera's dataset URI up front and fail before enqueueing
    # anything. Discovering a missing mapping halfway through a 50-camera
    # submission leaves the fleet in a partially-dispatched state.
    dataset_uris = {}
    if args.publication_index:
        index_path = Path(args.publication_index)
        if not index_path.is_file():
            raise SystemExit(f"Publication index not found: {index_path}")
        index = json.loads(index_path.read_text())
        cameras = index.get("cameras", {})
        missing = [c for c in camera_ids if c not in cameras]
        if missing:
            raise SystemExit(
                f"{len(missing)} camera(s) have no published dataset in "
                f"{index_path}: {', '.join(missing[:10])}"
                f"{'...' if len(missing) > 10 else ''}. Published cameras: "
                f"{', '.join(sorted(cameras)) or '(none)'}. Publish them with "
                f"'python scripts/download_bccd.py --num-cameras N --upload-s3'."
            )
        for camera in camera_ids:
            uri = cameras[camera].get("s3_uri")
            if not uri:
                raise SystemExit(
                    f"Camera {camera} is in the index but was never uploaded to S3 "
                    f"(s3_uri is null). Re-run download_bccd.py with --upload-s3."
                )
            dataset_uris[camera] = uri

    batch_id = args.batch_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:8]
    )

    if not args.queue_url and not args.dry_run:
        cfn = boto3.client("cloudformation", region_name=args.region)
        outputs = cfn.describe_stacks(StackName=ORCHESTRATION_STACK_NAME)["Stacks"][0]["Outputs"]
        args.queue_url = next(
            o["OutputValue"] for o in outputs if o["OutputKey"] == OUTPUT_QUEUE_URL
        )

    print(f"Batch id: {batch_id}")
    print(f"  If this run fails part-way, retry it with:  --batch-id {batch_id}\n")

    sqs = boto3.client("sqs", region_name=args.region) if not args.dry_run else None

    print(f"Submitting {len(camera_ids)} training requests")
    if not args.dry_run:
        print(f"Queue: {args.queue_url}")

    for cam_id in camera_ids:
        request_id = camera_request_id(batch_id, cam_id)
        msg = {
            "request_id": request_id,
            "camera_id": cam_id,
            "parameters": {
                "TrainingInstanceType": args.instance_type,
                "NumSamples": str(args.num_samples),
            },
        }
        if cam_id in dataset_uris:
            msg["parameters"]["TrainingDataS3Uri"] = dataset_uris[cam_id]
        if args.dry_run:
            print(f"  [dry-run] {cam_id}: {json.dumps(msg)}")
        else:
            sqs.send_message(
                QueueUrl=args.queue_url,
                MessageBody=json.dumps(msg, sort_keys=True),
                # Per-camera group, not one group for the whole fleet. A single
                # group serialises every camera behind the head of the queue, so
                # one poison message blocks the entire batch.
                MessageGroupId=cam_id,
                MessageDeduplicationId=request_id,
            )
            print(f"  sent {cam_id} (request_id={request_id})")

    print(f"\nDone. {len(camera_ids)} messages {'would be ' if args.dry_run else ''}sent.")
    print(f"Batch id: {batch_id}")
    print(f"  To RETRY this exact submission without creating duplicate training "
          f"runs:\n    --batch-id {batch_id}")


if __name__ == "__main__":
    main()
