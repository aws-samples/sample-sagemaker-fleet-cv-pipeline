# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Computer Vision Data Preprocessing for SageMaker SDK v3
Generates Anomalib (anomaly detection) datasets in MVTec-style format,
which also feed the defect classifier training branch.
"""
import argparse
import os
import logging
from pathlib import Path
import tarfile
import shutil
from PIL import Image, ImageDraw
import random
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_normal_image(size):
    """Create a normal image (clean production line)"""
    img = Image.new('RGB', size, color=(180, 180, 185))
    draw = ImageDraw.Draw(img)
    
    # Add subtle texture variations (reflections)
    for _ in range(50):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        radius = random.randint(5, 20)
        color = (
            random.randint(170, 190),
            random.randint(170, 190),
            random.randint(175, 195)
        )
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    return img


def create_anomalous_image(size):
    """Create an anomalous image (debris, vials, tools on line)"""
    img = create_normal_image(size)
    draw = ImageDraw.Draw(img)
    
    # Add random anomalies
    num_anomalies = random.randint(1, 3)
    
    for _ in range(num_anomalies):
        anomaly_type = random.choice(['debris', 'vial', 'tool'])
        
        if anomaly_type == 'debris':
            x = random.randint(20, size[0]-20)
            y = random.randint(20, size[1]-20)
            radius = random.randint(3, 10)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=(50, 50, 50))
        
        elif anomaly_type == 'vial':
            x = random.randint(30, size[0]-30)
            y = random.randint(30, size[1]-30)
            width = random.randint(15, 25)
            height = random.randint(40, 60)
            draw.rectangle([x, y, x+width, y+height], 
                          fill=(220, 220, 230), 
                          outline=(100, 100, 110), 
                          width=2)
        
        elif anomaly_type == 'tool':
            x = random.randint(30, size[0]-50)
            y = random.randint(30, size[1]-30)
            width = random.randint(30, 60)
            height = random.randint(10, 20)
            draw.rectangle([x, y, x+width, y+height], 
                          fill=(80, 80, 90), 
                          outline=(50, 50, 60), 
                          width=1)
    
    return img


def create_anomalib_dataset(output_dir, num_samples, split_name, anomaly_ratio=0.1, image_size=256):
    """Create dataset in MVTec-style format for Anomalib"""
    output_path = Path(output_dir)
    size = (image_size, image_size)
    
    normal_dir = output_path / split_name / 'good'
    anomaly_dir = output_path / split_name / 'defect'
    
    normal_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    
    num_anomalies = int(num_samples * anomaly_ratio)
    num_normal = num_samples - num_anomalies
    
    logger.info(f"Creating {split_name} Anomalib dataset: {num_normal} normal, {num_anomalies} anomalous")
    
    for i in range(num_normal):
        img = create_normal_image(size)
        img.save(normal_dir / f"normal_{i:04d}.png")

    for i in range(num_anomalies):
        img = create_anomalous_image(size)
        img.save(anomaly_dir / f"anomaly_{i:04d}.png")
    
    logger.info(f"Created {split_name} Anomalib dataset at {output_path}")
    
    return output_path / split_name


def create_tarball(source_dir, output_file):
    """Create tarball from directory"""
    with tarfile.open(output_file, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    logger.info(f"Created tarball: {output_file}")


def download_s3_training_data(s3_uri, base_output):
    """Download pre-existing training data from S3"""
    s3 = boto3.client('s3')
    
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    s3_path = s3_uri[5:]  # Remove 's3://'
    bucket = s3_path.split('/')[0]
    prefix = s3_path[len(bucket):].lstrip('/')
    
    logger.info(f"Downloading training data from s3://{bucket}/{prefix}")
    
    # Download each split's tarball
    for split in ['train', 'validation', 'test']:
        key = f"{prefix}/{split}/{split}.tar.gz"
        output_dir = base_output / split
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{split}.tar.gz"
        
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3.download_file(bucket, key, str(output_file))
        logger.info(f"Downloaded {split}.tar.gz ({output_file.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic CV dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples per split')
    parser.add_argument('--format', type=str, default='anomalib',
                       choices=['anomalib'],
                       help='Dataset format: anomalib (anomaly detection, also used by the classifier)')
    parser.add_argument('--anomaly-ratio', type=float, default=0.1,
                       help='Ratio of anomalous samples for Anomalib (0.0-1.0)')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size (square)')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Seed for synthetic image generation')
    parser.add_argument('--training-data-s3-uri', type=str, default='',
                       help='S3 URI containing pre-existing training data (bypasses synthetic generation)')

    args = parser.parse_args()

    random.seed(args.seed)
    logger.info(f"Starting preprocessing with format: {args.format}, seed: {args.seed}")
    logger.info(f"Samples per split: {args.num_samples}")
    
    # Output directories
    base_output = Path("/opt/ml/processing/output")
    
    # Check if real training data S3 URI is provided
    if args.training_data_s3_uri and args.training_data_s3_uri.lower() != "none":
        s3_uri = args.training_data_s3_uri.rstrip("/")
        # Append format to S3 path: {base}/anomalib
        s3_uri = f"{s3_uri}/{args.format}"
        logger.info(f"Using pre-existing training data from: {s3_uri}")
        download_s3_training_data(s3_uri, base_output)
        logger.info("Real data passthrough completed successfully!")
        return
    
    # Fall back to synthetic data generation
    logger.info("No training data S3 URI provided, generating synthetic data")
    temp_dir = Path("/opt/ml/processing/temp")
    # Ensure no stale files leak across runs/formats.
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets for each split. Validation and test are sized at a
    # quarter of the training split (minimum 40) so each carries enough
    # defect images (10 at the default NumSamples=400, anomaly ratio 0.1)
    # for the classifier gate: with fewer defects a single prediction flip
    # moves balanced accuracy by whole tenths.
    splits = {
        'train': args.num_samples,
        'validation': max(40, args.num_samples // 4),
        'test': max(40, args.num_samples // 4),
    }
    
    for split_name, num_samples in splits.items():
        logger.info(f"Creating {split_name} dataset...")
        split_temp_dir = temp_dir / split_name
        if split_temp_dir.exists():
            shutil.rmtree(split_temp_dir)
        
        # Anomalib: train only on normal, validation/test have anomalies
        anomaly_ratio = args.anomaly_ratio if split_name != 'train' else 0.0
        dataset_dir = create_anomalib_dataset(
            temp_dir,
            num_samples,
            split_name,
            anomaly_ratio=anomaly_ratio,
            image_size=args.image_size,
        )
        
        # Create tarball
        output_tar = base_output / split_name / f"{split_name}.tar.gz"
        output_tar.parent.mkdir(parents=True, exist_ok=True)
        if output_tar.exists():
            output_tar.unlink()
        create_tarball(dataset_dir, output_tar)
    
    logger.info("Preprocessing completed successfully!")
    
    # List outputs
    logger.info("Generated files:")
    for split in splits.keys():
        tar_file = base_output / split / f"{split}.tar.gz"
        if tar_file.exists():
            size = tar_file.stat().st_size / 1024
            logger.info(f"  {split}.tar.gz: {size:.1f} KB")


if __name__ == "__main__":
    main()