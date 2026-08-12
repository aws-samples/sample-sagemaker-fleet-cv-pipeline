# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Submit batch training requests to the orchestration queue.

Usage:
    # Train 50 camera models (5 at a time via orchestration)
    python scripts/submit_batch.py --cameras 50

    # Train specific cameras with custom parameters
    python scripts/submit_batch.py --camera-ids cam-001 cam-002 cam-003 --instance-type ml.g4dn.xlarge

    # Dry run -- show messages without sending
    python scripts/submit_batch.py --cameras 10 --dry-run
"""
import argparse
import json
import boto3


def main():
    parser = argparse.ArgumentParser(description="Submit batch training requests")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cameras", type=int, help="Number of cameras to train")
    group.add_argument("--camera-ids", nargs="+", help="Specific camera IDs")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    # 400 matches the pipeline default: at 100 the classifier eval split is
    # ~7 images and the balanced-accuracy gate fails on data starvation.
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--data-s3-prefix", default=None,
                        help="S3 prefix containing per-camera data (e.g. s3://bucket/cameras). "
                             "Each camera gets {prefix}/{camera_id}/anomalib")
    parser.add_argument("--queue-url", default=None, help="SQS queue URL (auto-detected if omitted)")
    parser.add_argument("--region", default=boto3.session.Session().region_name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.region:
        raise RuntimeError(
            "AWS region could not be resolved from the boto3 session. "
            "Pass --region or configure AWS_REGION/AWS_DEFAULT_REGION."
        )

    camera_ids = args.camera_ids or [f"cam-{i:03d}" for i in range(1, args.cameras + 1)]

    if not args.queue_url and not args.dry_run:
        cfn = boto3.client("cloudformation", region_name=args.region)
        outputs = cfn.describe_stacks(StackName="OrchestrationStack")["Stacks"][0]["Outputs"]
        args.queue_url = next(o["OutputValue"] for o in outputs if o["OutputKey"] == "QueueUrl")

    sqs = boto3.client("sqs", region_name=args.region) if not args.dry_run else None

    print(f"Submitting {len(camera_ids)} training requests")
    if not args.dry_run:
        print(f"Queue: {args.queue_url}")

    for cam_id in camera_ids:
        msg = {
            "camera_id": cam_id,
            "parameters": {
                "TrainingInstanceType": args.instance_type,
                "NumSamples": str(args.num_samples),
            },
        }
        if args.data_s3_prefix:
            prefix = args.data_s3_prefix.rstrip("/")
            msg["parameters"]["TrainingDataS3Uri"] = f"{prefix}/{cam_id}"
        if args.dry_run:
            print(f"  [dry-run] {cam_id}: {json.dumps(msg)}")
        else:
            sqs.send_message(
                QueueUrl=args.queue_url,
                MessageBody=json.dumps(msg),
                MessageGroupId="training-batch",
            )
            print(f"  sent {cam_id}")

    print(f"\nDone. {len(camera_ids)} messages {'would be ' if args.dry_run else ''}sent.")


if __name__ == "__main__":
    main()