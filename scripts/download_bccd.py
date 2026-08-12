#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Download and prepare BCCD dataset for anomaly detection and classifier training.

Dataset provenance: the BCCD blood cell images were originally open-sourced on
GitHub by cosmicad (https://github.com/cosmicad/dataset) and akshaylamba
(https://github.com/akshaylamba/all_CELL_data). This script clones the
Shenggan/BCCD_Dataset repository (MIT license), which reorganizes those images
into Pascal VOC format with XML annotations. The same underlying dataset is
also distributed by Roboflow (https://public.roboflow.com/object-detection/bccd)
as public domain.
"""

import argparse
import logging
import random
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
# The stdlib xml parser is vulnerable to XXE / entity-expansion attacks;
# defusedxml disables those constructs. The annotations come from a
# third-party GitHub repo, i.e. untrusted input.
import defusedxml.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_MAPPING = {"RBC": 0, "WBC": 1, "Platelets": 2}
GITHUB_REPO = "https://github.com/Shenggan/BCCD_Dataset.git"


def clone_dataset(temp_dir: Path) -> Path:
    """Clone BCCD dataset from GitHub."""
    dataset_dir = temp_dir / "BCCD_Dataset"
    if dataset_dir.exists():
        logger.info("Dataset already exists, skipping clone")
        return dataset_dir
    
    logger.info("Cloning BCCD dataset from GitHub...")
    subprocess.run(["git", "clone", GITHUB_REPO, str(dataset_dir)], check=True)
    return dataset_dir


def parse_voc_annotation(xml_path: Path) -> List[Dict]:
    """Parse VOC XML annotation file and return the object classes present."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAPPING:
            continue

        objects.append({
            'class': name,
            'class_id': CLASS_MAPPING[name],
        })

    return objects


def split_data(image_files: List[Path], train_ratio=0.7, val_ratio=0.15, seed=1337) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split data into train/val/test sets (seeded, on a sorted copy, so the
    same seed always yields the same split regardless of filesystem order)."""
    image_files = sorted(image_files)
    random.Random(seed).shuffle(image_files)
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    return train_files, val_files, test_files


def create_anomalib_format(dataset_dir: Path, output_dir: Path, image_files: List[Path], train_mode: bool = False):
    """Create Anomalib format (good/defect directories).
    
    In train_mode, only include 'good' images (no Platelets) -- anomalib trains on normal only.
    """
    good_dir = output_dir / "good"
    defect_dir = output_dir / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    if not train_mode:
        defect_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in image_files:
        xml_file = dataset_dir / "BCCD" / "Annotations" / f"{img_file.stem}.xml"
        if not xml_file.exists():
            continue
            
        objects = parse_voc_annotation(xml_file)
        has_platelets = any(obj['class'] == 'Platelets' for obj in objects)
        
        if train_mode and has_platelets:
            continue  # Skip anomalous images in training set
        
        # Copy image to appropriate directory
        target_dir = defect_dir if has_platelets else good_dir
        shutil.copy2(dataset_dir / "BCCD" / "JPEGImages" / img_file.name, target_dir / f"{img_file.stem}.png")


def create_tarball(source_dir: Path, output_path: Path, split_name: str):
    """Create tarball with split_name/ prefix."""
    with tarfile.open(output_path, "w:gz") as tar:
        for item in source_dir.iterdir():
            tar.add(item, arcname=f"{split_name}/{item.name}")


def get_bucket_from_stack() -> str:
    """Get S3 bucket name from CloudFormation stack output."""
    cfn = boto3.client("cloudformation")
    outputs = cfn.describe_stacks(StackName="SageMakerCVPipelineStack")["Stacks"][0]["Outputs"]
    return next(o["OutputValue"] for o in outputs if o["OutputKey"] == "CVBucketName")


def upload_to_s3(local_path: Path, bucket: str, key: str):
    """Upload file to S3."""
    s3 = boto3.client("s3")
    logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare BCCD dataset")
    parser.add_argument("--output-dir", default="./data/bccd", help="Output directory")
    parser.add_argument("--num-cameras", type=int, help="Number of camera folders to create")
    parser.add_argument("--upload-s3", action="store_true", help="Upload to S3 bucket")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for the train/validation/test shuffle")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Clone dataset
        dataset_dir = clone_dataset(temp_path)
        
        # Get all image files
        images_dir = dataset_dir / "BCCD" / "JPEGImages"
        image_files = sorted(images_dir.glob("*.jpg"))
        logger.info(f"Found {len(image_files)} images")
        
        if args.num_cameras:
            # Partition into camera folders
            images_per_camera = len(image_files) // args.num_cameras
            
            for cam_idx in range(args.num_cameras):
                cam_id = f"cam-{cam_idx+1:03d}"
                cam_dir = output_dir / cam_id
                
                # Get images for this camera
                start_idx = cam_idx * images_per_camera
                end_idx = start_idx + images_per_camera if cam_idx < args.num_cameras - 1 else len(image_files)
                cam_images = image_files[start_idx:end_idx]
                
                logger.info(f"Processing camera {cam_id} with {len(cam_images)} images")
                
                # Split data for this camera
                train_files, val_files, test_files = split_data(cam_images, seed=args.seed)
                
                for split_name, split_files in [("train", train_files), ("validation", val_files), ("test", test_files)]:
                    # Create Anomalib format
                    anomalib_split_dir = cam_dir / "anomalib" / split_name
                    create_anomalib_format(dataset_dir, anomalib_split_dir, split_files, train_mode=(split_name == "train"))

                    # Create tarball
                    anomalib_tar = cam_dir / "anomalib" / f"{split_name}.tar.gz"
                    create_tarball(anomalib_split_dir, anomalib_tar, split_name)

                    logger.info(f"Created {anomalib_tar}")

                    # Upload to S3 if requested
                    if args.upload_s3:
                        bucket = get_bucket_from_stack()

                        anomalib_key = f"cameras/{cam_id}/anomalib/{split_name}/{split_name}.tar.gz"
                        upload_to_s3(anomalib_tar, bucket, anomalib_key)
        else:
            # Single dataset (no camera partitioning)
            train_files, val_files, test_files = split_data(image_files, seed=args.seed)
            
            for split_name, split_files in [("train", train_files), ("validation", val_files), ("test", test_files)]:
                # Create Anomalib format
                anomalib_split_dir = output_dir / "anomalib" / split_name
                create_anomalib_format(dataset_dir, anomalib_split_dir, split_files, train_mode=(split_name == "train"))

                # Create tarball
                anomalib_tar = output_dir / "anomalib" / f"{split_name}.tar.gz"
                create_tarball(anomalib_split_dir, anomalib_tar, split_name)

                logger.info(f"Created {anomalib_tar}")

                # Upload to S3 if requested
                if args.upload_s3:
                    bucket = get_bucket_from_stack()

                    anomalib_key = f"data/anomalib/{split_name}/{split_name}.tar.gz"
                    upload_to_s3(anomalib_tar, bucket, anomalib_key)
    
    logger.info("Dataset preparation complete")


if __name__ == "__main__":
    main()