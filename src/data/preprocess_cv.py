# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Computer Vision Data Preprocessing for SageMaker SDK v3 Generates Anomalib (anomaly detection) datasets in MVTec-style format, which also feed the defect classifier training branch.'
import argparse
import hashlib
import json
import os
import logging
from math import ceil
from pathlib import Path
import tarfile
import shutil
from PIL import Image, ImageDraw
import random
import boto3
from botocore.exceptions import ClientError

from utils.dataset import (
    SUPPORTED_SPLIT_SCHEME_VERSIONS,
    assert_splits_disjoint,
    assign_split,
    derive_dataset_id,
    split_content_hash as content_hash,
)
from utils.lineage import DATASET_IDENTITY_FIELDS, parse_split_scheme_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Every split must carry at least this many images of BOTH classes. Below two,
# per-class recall is undefined or a single coin flip, and the pipeline quality
# gates key on per-class recall.
MIN_IMAGES_PER_CLASS = 2
# Floor for the validation and test splits, which are otherwise a quarter of
# train.
MIN_SPLIT_SAMPLES = 40

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Must match config.DATASET_MANIFEST_NAME. Defined locally because this script
# runs inside the training image, which does not contain the repository's
# config.py.
MANIFEST_NAME = "dataset_manifest.json"

SPLIT_SCHEME = "generated-disjoint"
SPLIT_SCHEME_VERSION = 2

VALIDATION_OF_TRAIN = 0.25
TEST_OF_TRAIN = 0.25


def split_fractions(splits: dict) -> dict:
    """The fractions a concrete split plan actually realises."""
    total = sum(splits.values())
    return {name: round(count / total, 6) for name, count in splits.items()}
LABEL_MAPPING = "synthetic: images with injected artefacts -> defect, else good"
# Must match config.DATASET_READY_MARKER.
READY_MARKER = "_READY"


def class_counts(split_dir: Path) -> dict:
    """Per-class image counts for a split directory."""
    counts = {}
    for class_name in ("good", "defect"):
        class_dir = split_dir / class_name
        counts[class_name] = (
            sum(1 for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if class_dir.is_dir() else 0
        )
    return counts


def create_normal_image(size):
    'Create a normal image (clean production line).'
    img = Image.new('RGB', size, color=(180, 180, 185))
    draw = ImageDraw.Draw(img)
    width, height = size

    # Add subtle texture variations (reflections)
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(max(1, int(width * 0.02)), max(2, int(width * 0.08)))
        color = (
            random.randint(170, 190),
            random.randint(170, 190),
            random.randint(175, 195)
        )
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

    return img


def create_anomalous_image(size):
    'Create an anomalous image (debris, vials, tools on line).'
    img = create_normal_image(size)
    draw = ImageDraw.Draw(img)
    width, height = size

    def span(lo_frac, hi_frac, extent):
        """Random pixel length between two fractions of *extent*, at least 1px."""
        lo = max(1, int(extent * lo_frac))
        hi = max(lo, int(extent * hi_frac))
        return random.randint(lo, hi)

    def origin(shape_w, shape_h):
        """Random top-left placing a shape_w x shape_h box fully inside the image."""
        return (
            random.randint(0, max(0, width - shape_w)),
            random.randint(0, max(0, height - shape_h)),
        )

    # Add random anomalies
    num_anomalies = random.randint(1, 3)

    for _ in range(num_anomalies):
        anomaly_type = random.choice(['debris', 'vial', 'tool'])

        if anomaly_type == 'debris':
            radius = span(0.012, 0.04, width)
            x, y = origin(2 * radius, 2 * radius)
            draw.ellipse([x, y, x + 2 * radius, y + 2 * radius], fill=(50, 50, 50))

        elif anomaly_type == 'vial':
            w = span(0.06, 0.10, width)
            h = span(0.16, 0.23, height)
            x, y = origin(w, h)
            draw.rectangle([x, y, x + w, y + h],
                           fill=(220, 220, 230),
                           outline=(100, 100, 110),
                           width=max(1, width // 128))

        elif anomaly_type == 'tool':
            w = span(0.12, 0.23, width)
            h = span(0.04, 0.08, height)
            x, y = origin(w, h)
            draw.rectangle([x, y, x + w, y + h],
                           fill=(80, 80, 90),
                           outline=(50, 50, 60),
                           width=max(1, width // 256))

    return img


def create_anomalib_dataset(output_dir, num_samples, split_name, anomaly_ratio=0.1, image_size=256):
    'Create dataset in MVTec-style format for Anomalib.'
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


def plan_splits(num_samples, anomaly_ratio):
    'Return {split: count}, or raise naming what is too small.'
    if num_samples < 1:
        raise ValueError(f"--num-samples must be >= 1, got {num_samples}")
    if not 0.0 < anomaly_ratio < 1.0:
        raise ValueError(
            f"--anomaly-ratio must be strictly between 0 and 1, got {anomaly_ratio}. "
            "Every split needs both classes."
        )

    splits = {
        'train': num_samples,
        'validation': max(MIN_SPLIT_SAMPLES, num_samples // 4),
        'test': max(MIN_SPLIT_SAMPLES, num_samples // 4),
    }

    problems = []
    for split_name, count in splits.items():
        defects = int(count * anomaly_ratio)
        good = count - defects
        if defects < MIN_IMAGES_PER_CLASS or good < MIN_IMAGES_PER_CLASS:
            problems.append(
                f"  {split_name}: {count} images -> {good} good, {defects} defect "
                f"(need >= {MIN_IMAGES_PER_CLASS} of each)"
            )
    if problems:
        scarcer_fraction = min(anomaly_ratio, 1.0 - anomaly_ratio)
        # smallest split >= MIN/scarcer_fraction, and that split is train//4
        needed_per_split = ceil(MIN_IMAGES_PER_CLASS / scarcer_fraction)
        min_samples = max(needed_per_split, 4 * needed_per_split
                          if needed_per_split > MIN_SPLIT_SAMPLES else needed_per_split)
        raise ValueError(
            f"--num-samples {num_samples} with --anomaly-ratio {anomaly_ratio} "
            f"produces splits that cannot support a per-class metric:\n"
            + "\n".join(problems)
            + f"\nAt this ratio the scarcer class is "
            f"{'defect' if anomaly_ratio <= 0.5 else 'good'} "
            f"({scarcer_fraction:.3g} of each split), and validation/test are a "
            f"quarter of train (floor {MIN_SPLIT_SAMPLES}). Use --num-samples >= "
            f"{min_samples}, or move --anomaly-ratio closer to 0.5."
        )
    return splits


def create_tarball(source_dir, output_file):
    """Create tarball from directory"""
    with tarfile.open(output_file, 'w:gz') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    logger.info(f"Created tarball: {output_file}")


def download_s3_training_data(base_uri, data_format, base_output, region=None):
    'Download a published dataset from S3 into the processing output.'
    s3 = boto3.client('s3', region_name=region)

    if not base_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {base_uri}")
    path = base_uri[5:]
    bucket = path.split('/')[0]
    base_prefix = path[len(bucket):].strip('/')

    marker_key = f"{base_prefix}/{READY_MARKER}" if base_prefix else READY_MARKER
    try:
        s3.head_object(Bucket=bucket, Key=marker_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            raise RuntimeError(
                f"Dataset at {base_uri} has no {READY_MARKER} marker "
                f"(s3://{bucket}/{marker_key}). The publisher writes that marker "
                f"last, so its absence means the dataset is incomplete, still "
                f"uploading, or was not published by scripts/download_bccd.py. "
                f"Refusing to train on it: a partial dataset produces metrics that "
                f"look normal but are computed on splits that do not belong "
                f"together. Publish with "
                f"'python scripts/download_bccd.py --upload-s3' and use the URI it "
                f"prints."
            ) from exc
        raise

    # REQUIRED, not advisory: it identifies the dataset and the split contents are checked
    # against it. An absent manifest raises below rather than proceeding.
    manifest_key = f"{base_prefix}/{MANIFEST_NAME}" if base_prefix else MANIFEST_NAME
    try:
        body = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        manifest = json.loads(body)
        missing = [s for s in ("train", "validation", "test")
                   if s not in manifest.get("splits", {})]
        if missing:
            raise RuntimeError(
                f"Dataset manifest at s3://{bucket}/{manifest_key} does not "
                f"describe split(s): {', '.join(missing)}."
            )
        logger.info("Dataset manifest: dataset_id=%s source=%s",
                    manifest.get("dataset_id"), manifest.get("source"))
        for split_name, info in manifest["splits"].items():
            logger.info("  %s: %s", split_name, info.get("counts"))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchKey", "NotFound"):
            raise
        # NOT advisory any more. Downstream validation now REQUIRES the manifest to check the
        # downloaded splits, so "proceed without it" left the run to fail later with a less
        # useful message -- or to consume a dataset whose identity nothing can establish.
        raise RuntimeError(
            f"No {MANIFEST_NAME} at s3://{bucket}/{manifest_key}. This dataset cannot be "
            f"consumed: the manifest is what identifies it and what the split contents are "
            f"checked against. Publish it with "
            f"'python scripts/download_bccd.py --output-dir <dir> --upload-s3', which writes "
            f"the manifest alongside the splits."
        )

    format_prefix = f"{base_prefix}/{data_format}".strip('/')
    logger.info("Downloading training data from s3://%s/%s", bucket, format_prefix)

    staging = base_output / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'validation', 'test'):
        key = f"{format_prefix}/{split}/{split}.tar.gz"
        staged = staging / f"{split}.tar.gz"
        logger.info("Downloading s3://%s/%s", bucket, key)
        try:
            s3.download_file(bucket, key, str(staged))
        except ClientError as exc:
            raise RuntimeError(
                f"Could not download s3://{bucket}/{key}: {exc}. The dataset is "
                f"marked ready, so this split is missing or unreadable. Note that "
                f"TrainingDataS3Uri must be the prefix ABOVE the format directory "
                f"-- this script appends /{data_format}/<split>/<split>.tar.gz."
            ) from exc
        logger.info("Downloaded %s.tar.gz (%.1f KB)",
                    split, staged.stat().st_size / 1024)

    _validate_downloaded_dataset(staging, manifest)

    # Only now publish to the processing outputs SageMaker will upload.
    for split in ('train', 'validation', 'test'):
        output_dir = base_output / split
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / f"{split}.tar.gz"),
                    str(output_dir / f"{split}.tar.gz"))

    # Keep the manifest alongside the splits so downstream steps can read it. Unconditional --
    # reaching here without one is impossible now that an absent manifest raises.
    (base_output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    shutil.rmtree(staging, ignore_errors=True)


# The split directory names a consumer's resolver recognises. Kept here so the
# validator can reject a layout the resolver would read differently.
KNOWN_SPLIT_NAMES = ("train", "validation", "test")


def _validate_downloaded_dataset(staging: Path, manifest: dict) -> None:
    'Check staged archives against the manifest before they are used.'
    if manifest is None:
        raise RuntimeError(
            f"Refusing to train on a dataset with no {MANIFEST_NAME}: its contents "
            f"cannot be checked, so a split that overlaps another or is smaller "
            f"than its metric needs would go unnoticed. Republish with "
            f"'python scripts/download_bccd.py --upload-s3'."
        )

    extracted = staging / "extracted"
    split_dirs = {}
    for split in ('train', 'validation', 'test'):
        target = extracted / split
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(staging / f"{split}.tar.gz", "r:gz") as tar:
            tar.extractall(target, filter="data")
        # Each archive must contain the split it is named for. A train archive
        # published under the test key would otherwise be evaluated as test.
        root = target / split
        if not root.is_dir():
            raise RuntimeError(
                f"{split}.tar.gz does not contain a top-level '{split}/' directory. "
                f"Found: {sorted(p.name for p in target.iterdir())}. The archive "
                f"does not hold the split it is published as."
            )

        nested = [d.name for d in root.iterdir()
                  if d.is_dir() and d.name in KNOWN_SPLIT_NAMES]
        if nested:
            raise RuntimeError(
                f"{split}.tar.gz has an ambiguous nested layout: '{split}/' contains "
                f"split-named subdirectories {sorted(nested)}. Validation would "
                f"inspect '{split}/' while training would select "
                f"'{split}/{nested[0]}/', so the validated data and the trained data "
                f"need not be the same. Republish with "
                f"'python scripts/download_bccd.py --upload-s3'."
            )
        split_dirs[split] = root

    for split, root in split_dirs.items():
        embedded_path = root / MANIFEST_NAME
        if not embedded_path.is_file():
            raise RuntimeError(
                f"{split}.tar.gz has no {MANIFEST_NAME} at its split root. "
                f"The trainers read that embedded copy to record what the model was "
                f"fitted on, so an archive without it cannot establish its own "
                f"identity. Republish with "
                f"'python scripts/download_bccd.py --upload-s3'."
            )
        try:
            embedded = json.loads(embedded_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{split}.tar.gz contains an unreadable {MANIFEST_NAME} "
                f"({exc}); refusing to guess the dataset identity."
            ) from exc
        mismatched = [
            f"    {field}: embedded {embedded.get(field)!r} vs published "
            f"{manifest.get(field)!r}"
            for field in DATASET_IDENTITY_FIELDS
            if embedded.get(field) != manifest.get(field)
        ]
        if mismatched:
            raise RuntimeError(
                f"The {MANIFEST_NAME} inside {split}.tar.gz disagrees with "
                f"the published one:\n" + "\n".join(mismatched) + "\n  The embedded "
                "copy is what the trainers read, so this dataset would train "
                "under an identity that is not the one it was validated as."
            )

    # Recompute rather than trust: the manifest is metadata and can be wrong or
    # forged, while these numbers come from the bytes actually downloaded.
    problems = []
    for split, root in split_dirs.items():
        declared = manifest["splits"][split]
        actual_counts = class_counts(root)
        if actual_counts != declared["counts"]:
            problems.append(
                f"  {split}: manifest says {declared['counts']}, archive contains "
                f"{actual_counts}"
            )
        actual_hash = content_hash(root)
        if actual_hash != declared["content_sha256"]:
            problems.append(
                f"  {split}: content hash {actual_hash[:16]}... does not match the "
                f"manifest's {declared['content_sha256'][:16]}..."
            )
        for class_name, count in actual_counts.items():
            if count < MIN_IMAGES_PER_CLASS:
                problems.append(
                    f"  {split}: {count} '{class_name}' image(s), need >= "
                    f"{MIN_IMAGES_PER_CLASS} for a per-class metric"
                )
    recomputed_hashes = {
        split: content_hash(root) for split, root in sorted(split_dirs.items())
    }
    implied_id = derive_dataset_id(recomputed_hashes)
    declared_id = manifest.get("dataset_id")
    if declared_id != implied_id:
        problems.append(
            f"  dataset_id: manifest declares {declared_id!r} but this content "
            f"implies {implied_id!r}. The id is derived from the per-split content "
            f"hashes, so a mismatch means the metadata does not describe these "
            f"images -- and warm-start uses that id to decide whether two datasets "
            f"are the same version."
        )

    declared_scheme = manifest.get("split_scheme")
    if declared_scheme == "content-addressed-sha256":
        # No implicit defaults. Verifying the assignment against ASSUMED ratios
        # would confirm a partitioning the dataset never declared, which is the
        # opposite of what this check is for.
        ratios = manifest.get("split_ratios") or {}
        missing_ratios = [k for k in ("train", "validation") if k not in ratios]
        if missing_ratios:
            problems.append(
                f"  split_ratios: a manifest declaring {declared_scheme!r} must "
                f"record its assignment ratios, but {', '.join(missing_ratios)} "
                f"{'is' if len(missing_ratios) == 1 else 'are'} absent. The split "
                f"assignment cannot be verified against ratios that were never "
                f"stated."
            )
            ratios = None
        train_ratio = ratios.get("train") if ratios else None
        val_ratio = ratios.get("validation") if ratios else None
        declared_version = manifest.get("split_scheme_version")
        # Strict: a fractional version must not truncate into a supported one.
        declared_version = parse_split_scheme_version(declared_version)
        if declared_version is None:
            problems.append(
                f"  split_scheme_version: a manifest declaring {declared_scheme!r} "
                f"must record an integer scheme version, got "
                f"{manifest.get('split_scheme_version')!r}. The assignment cannot be "
                f"verified without knowing which digest it was derived from."
            )
            ratios = None
            declared_version = None
        else:
            if declared_version not in SUPPORTED_SPLIT_SCHEME_VERSIONS:
                problems.append(
                    f"  split_scheme_version: {declared_version} is not a version "
                    f"this code can verify (supported: "
                    f"{list(SUPPORTED_SPLIT_SCHEME_VERSIONS)}). Refusing to check the "
                    f"assignment under a different version than the one it was made "
                    f"with."
                )
                ratios = None
                declared_version = None
        misplaced = []
        for split, root in sorted(split_dirs.items()) if ratios else []:
            for class_name in ("good", "defect"):
                class_dir = root / class_name
                if not class_dir.is_dir():
                    continue
                for image in sorted(class_dir.iterdir()):
                    if not image.is_file():
                        continue
                    expected = assign_split(image, train_ratio, val_ratio,
                                            scheme_version=declared_version)
                    if expected != split:
                        misplaced.append((f"{split}/{class_name}/{image.name}",
                                          expected))
                        if len(misplaced) >= 5:
                            break
                if len(misplaced) >= 5:
                    break
            if len(misplaced) >= 5:
                break
        if misplaced:
            detail = "; ".join(f"{where} belongs in {should}"
                               for where, should in misplaced)
            problems.append(
                f"  split_scheme: the manifest declares "
                f"{declared_scheme!r}, but images do not follow that assignment "
                f"({detail}). That scheme is what permits warm-starting across "
                f"dataset versions, so a false declaration would let a model "
                f"inherit weights fitted on its own holdout."
            )

    if problems:
        raise RuntimeError(
            "Downloaded dataset does not match its manifest:\n" + "\n".join(problems)
        )

    # The invariant every reported number depends on.
    assert_splits_disjoint(split_dirs)

    logger.info("Dataset validated: counts, content hashes and split disjointness "
                "all match the manifest")
    shutil.rmtree(extracted, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic CV dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples in the training split')
    parser.add_argument('--format', type=str, default='anomalib',
                       choices=['anomalib'],
                       help='Dataset format: anomalib (anomaly detection, also used by the classifier)')
    parser.add_argument('--anomaly-ratio', type=float, default=0.1,
                       help='Fraction of anomalous samples in every split (0 < r < 1)')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size (square)')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Seed for synthetic image generation')
    parser.add_argument('--training-data-s3-uri', type=str, default='',
                       help='S3 URI containing pre-existing training data (bypasses synthetic '
                            'generation). Give the prefix ABOVE the format directory: this '
                            'script appends /{format}/{split}/{split}.tar.gz')
    # Parameterized so the job can run outside a SageMaker container -- the
    # previous hardcoded /opt/ml/processing paths made local reproduction of a
    # preprocessing bug impossible without a container.
    parser.add_argument('--output-dir', type=str,
                       default=os.environ.get('SM_OUTPUT_DIR', '/opt/ml/processing/output'),
                       help='Directory to write per-split tarballs into')
    # Processing containers do not receive AWS_REGION automatically, so the S3
    # client would otherwise resolve no Region on the real-data path.
    parser.add_argument('--region', type=str,
                       default=os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION'),
                       help='AWS Region for S3 access on the real-data path')
    parser.add_argument('--temp-dir', type=str,
                       default=os.environ.get('SM_TEMP_DIR', '/opt/ml/processing/temp'),
                       help='Scratch directory for staging generated images')

    args = parser.parse_args()

    random.seed(args.seed)
    logger.info(f"Starting preprocessing with format: {args.format}, seed: {args.seed}")
    logger.info(f"Samples in training split: {args.num_samples}")

    base_output = Path(args.output_dir)

    # Check if real training data S3 URI is provided
    if args.training_data_s3_uri and args.training_data_s3_uri.lower() != "none":
        # The URI is the dataset ROOT, above the format directory: this script
        # appends /{format}/{split}/{split}.tar.gz itself.
        base_uri = args.training_data_s3_uri.rstrip("/")
        logger.info("Using published training data from: %s", base_uri)
        download_s3_training_data(base_uri, args.format, base_output, region=args.region)
        logger.info("Real data passthrough completed successfully!")
        return

    # Fall back to synthetic data generation
    logger.info("No training data S3 URI provided, generating synthetic data")

    # Validate the split plan BEFORE writing any image or billing any job.
    splits = plan_splits(args.num_samples, args.anomaly_ratio)

    temp_dir = Path(args.temp_dir)
    # Ensure no stale files leak across runs/formats.
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    manifest_splits = {}
    split_dirs = {}
    for split_name, num_samples in splits.items():
        logger.info(f"Creating {split_name} dataset...")
        split_temp_dir = temp_dir / split_name
        if split_temp_dir.exists():
            shutil.rmtree(split_temp_dir)

        # Every split carries both classes; the autoencoder selects train/good.
        dataset_dir = create_anomalib_dataset(
            temp_dir,
            num_samples,
            split_name,
            anomaly_ratio=args.anomaly_ratio,
            image_size=args.image_size,
        )
        split_dirs[split_name] = dataset_dir

        manifest_splits[split_name] = {
            "counts": class_counts(dataset_dir),
            "content_sha256": content_hash(dataset_dir),
        }

    assert_splits_disjoint(split_dirs)

    manifest = {
        "schema_version": 1,
        "format": args.format,
        "source": "synthetic",
        "generator": "src/data/preprocess_cv.py",
        "split_scheme": SPLIT_SCHEME,
        "split_scheme_version": SPLIT_SCHEME_VERSION,
        # The fractions this plan ACTUALLY realises, not the nominal rule. Part of
        # the assignment contract: a consumer compares them before warm-starting,
        # and a manifest that omits them is refused rather than assumed to match.
        "split_ratios": split_fractions(splits),
        "label_mapping": LABEL_MAPPING,
        "seed": args.seed,
        "image_size": args.image_size,
        "anomaly_ratio": args.anomaly_ratio,
        "num_samples": args.num_samples,
        "splits": manifest_splits,
    }
    # dataset_id identifies the DATA, derived from the split content hashes, so
    # a warm-start consumer can tell whether a parent model was trained on the
    # same data without needing the data itself.
    manifest["dataset_id"] = hashlib.sha256(
        json.dumps(
            {k: v["content_sha256"] for k, v in manifest_splits.items()}, sort_keys=True
        ).encode()
    ).hexdigest()[:32]
    manifest_json = json.dumps(manifest, indent=2, allow_nan=False)

    for split_name, dataset_dir in split_dirs.items():
        (dataset_dir / MANIFEST_NAME).write_text(manifest_json, encoding="utf-8")

        output_tar = base_output / split_name / f"{split_name}.tar.gz"
        output_tar.parent.mkdir(parents=True, exist_ok=True)
        if output_tar.exists():
            output_tar.unlink()
        create_tarball(dataset_dir, output_tar)

    # Also at the output root, for local use and for the S3 publication path.
    manifest_path = base_output / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_json, encoding="utf-8")
    logger.info("Dataset manifest: dataset_id=%s", manifest["dataset_id"])

    logger.info("Preprocessing completed successfully!")

    # List outputs
    logger.info("Generated files:")
    for split in splits.keys():
        tar_file = base_output / split / f"{split}.tar.gz"
        if tar_file.exists():
            size = tar_file.stat().st_size / 1024
            counts = manifest_splits[split]["counts"]
            logger.info(f"  {split}.tar.gz: {size:.1f} KB "
                        f"({counts['good']} good, {counts['defect']} defect)")


if __name__ == "__main__":
    main()
