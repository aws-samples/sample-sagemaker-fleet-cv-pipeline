#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Fetch latest approved model from SageMaker Model Registry for retraining pipeline."""
import argparse
import json
from pathlib import Path

import boto3


def find_latest_for_camera(sm, group_name: str, camera_id: str):
    """Return (arn, description) of the newest approved package whose
    CameraId metadata matches, or (None, None).

    ListModelPackages cannot filter on customer metadata, so walk the
    newest-first pages and describe candidates until one matches. Without
    this filter every camera would warm-start from whichever camera
    registered most recently.
    """
    paginator = sm.get_paginator("list_model_packages")
    pages = paginator.paginate(
        ModelPackageGroupName=group_name,
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
                return summary["ModelPackageArn"], desc
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-group-name", required=True)
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/pretrained_model")
    parser.add_argument("--region", default=boto3.session.Session().region_name)
    parser.add_argument(
        "--camera-id",
        default="default",
        help="Fleet device identity; warm-start only considers packages "
        "registered with the same CameraId metadata.",
    )
    args = parser.parse_args()

    if not args.region:
        raise RuntimeError(
            "AWS region could not be resolved from the boto3 session. "
            "Pass --region or configure AWS_REGION/AWS_DEFAULT_REGION."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sm = boto3.client("sagemaker", region_name=args.region)

    arn, desc = find_latest_for_camera(
        sm, args.model_package_group_name, args.camera_id
    )

    if arn is None:
        print(
            f"No approved model for camera '{args.camera_id}' - "
            "creating marker file for first-time training"
        )
        (output_dir / "no_pretrained_model.marker").touch()
        return

    s3_uri = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    
    print(f"Found approved model: {arn}")
    print(f"Downloading from: {s3_uri}")
    
    # Download model tarball
    s3 = boto3.client("s3", region_name=args.region)
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    
    model_path = output_dir / "model.tar.gz"
    s3.download_file(bucket, key, str(model_path))
    
    # Write metadata
    metadata = {
        "model_package_arn": arn,
        "model_data_url": s3_uri,
        "camera_id": args.camera_id,
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model downloaded to: {model_path}")
    print("Metadata written for lineage tracking")


if __name__ == "__main__":
    main()