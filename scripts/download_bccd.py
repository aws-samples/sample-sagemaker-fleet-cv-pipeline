#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Download and prepare the BCCD dataset for anomaly detection and classifier training.'

import argparse
import gzip
import json
import logging
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError
# The stdlib xml parser is vulnerable to XXE / entity-expansion attacks;
# defusedxml disables those constructs. The annotations come from a
# third-party GitHub repo, i.e. untrusted input.
import defusedxml.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from aws_env import AwsEnvError, resolve_region  # noqa: E402
from utils.dataset import (  # noqa: E402
    assert_splits_disjoint,
    assign_split,
    derive_dataset_id,
    # One implementation, shared with the validator. Two copies of this existed and both
    # concatenated class + filename + raw bytes with no framing, which let one layout's
    # image content absorb another's filename and produce equal hashes for different data.
    split_content_hash as content_hash,
)
from config import (  # noqa: E402
    DATASET_MANIFEST_NAME,
    PUBLICATION_INDEX_NAME,
    DATASET_READY_MARKER,
    DATASETS_PREFIX,
    OUTPUT_BUCKET_NAME,
    PIPELINE_NAME,
    PIPELINE_STACK_NAME,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_MAPPING = {"RBC": 0, "WBC": 1, "Platelets": 2}
GITHUB_REPO = "https://github.com/Shenggan/BCCD_Dataset.git"
SPLITS = ("train", "validation", "test")
# The upstream licence, copied verbatim next to the images. Named distinctly so it
# is never confused with this repository's own MIT-0 LICENSE.
UPSTREAM_LICENSE_NAME = "LICENSE.upstream-bccd"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SPLIT_SCHEME = "content-addressed-sha256"
SPLIT_SCHEME_VERSION = 2
# Every split needs at least this many of BOTH classes, matching
# src/data/preprocess_cv.py: below two, per-class recall is undefined or a coin
# flip, and the pipeline quality gates key on per-class recall.
MIN_IMAGES_PER_CLASS = 2

UPSTREAM_NOTICE = """\
This dataset was generated from the BCCD blood cell dataset.

Upstream repository : https://github.com/Shenggan/BCCD_Dataset
Upstream commit     : {commit}
Upstream license    : MIT
Original sources    : https://github.com/cosmicad/dataset
                      https://github.com/akshaylamba/all_CELL_data
Also distributed by : https://public.roboflow.com/object-detection/bccd

Label mapping applied by scripts/download_bccd.py:
  an image containing at least one "Platelets" annotation -> class "defect"
  any other image                                         -> class "good"
This mapping is an illustrative stand-in for a manufacturing defect
distribution; it is not a medically meaningful label.

The MIT-0 license of the sample-sagemaker-fleet-cv-pipeline repository applies
to that repository's code, NOT to these dataset images. The upstream MIT license
above governs the images.
"""


def clone_dataset(temp_dir: Path, ref: str = None) -> Tuple[Path, str]:
    'Clone the BCCD dataset and return (path, resolved commit SHA).'
    dataset_dir = temp_dir / "BCCD_Dataset"
    if not dataset_dir.exists():
        if ref:
            logger.info("Cloning BCCD dataset at pinned ref %s...", ref)
            dataset_dir.mkdir(parents=True)
            subprocess.run(["git", "-C", str(dataset_dir), "init", "--quiet"], check=True)
            subprocess.run(
                ["git", "-C", str(dataset_dir), "remote", "add", "origin", GITHUB_REPO],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dataset_dir), "fetch", "--depth", "1", "origin", ref],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(dataset_dir), "checkout", "--quiet", "FETCH_HEAD"],
                check=True,
            )
        else:
            logger.info("Cloning BCCD dataset from GitHub (branch tip, NOT pinned)...")
            subprocess.run(
                ["git", "clone", "--depth", "1", GITHUB_REPO, str(dataset_dir)], check=True
            )
    commit = subprocess.run(
        ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if ref:
        logger.info("BCCD upstream commit: %s (pinned to %s)", commit, ref)
    else:
        logger.warning(
            "BCCD upstream commit: %s -- resolved from the MOVING branch tip. Pass "
            "--upstream-ref %s to reproduce this dataset later.", commit, commit,
        )
    return dataset_dir, commit


def _find_upstream_license(dataset_dir: Path):
    """Locate the cloned repository's licence file, whatever it is called."""
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        candidate = dataset_dir / name
        if candidate.is_file():
            return candidate
    return None


def parse_voc_annotation(xml_path: Path) -> List[Dict]:
    'Parse a VOC XML annotation file and return the object classes present.'
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for index, obj in enumerate(root.findall('object')):
        name_el = obj.find('name')
        name = name_el.text.strip() if name_el is not None and name_el.text else None
        if not name:
            raise ValueError(
                f"{xml_path}: object {index} has no <name>. Refusing to label this "
                f"image: an unnamed object cannot be shown to be absent of "
                f"platelets, so treating it as 'good' would publish manufactured "
                f"normal ground truth. Fix or remove the annotation."
            )
        if name not in CLASS_MAPPING:
            raise ValueError(
                f"{xml_path}: object {index} has class {name!r}, which is not in "
                f"the known BCCD vocabulary {sorted(CLASS_MAPPING)}. Skipping it "
                f"would let an unrecognised class -- a typo like 'Platelet', or a "
                f"different dataset entirely -- be published as a confirmed-normal "
                f"image."
            )
        objects.append({'class': name, 'class_id': CLASS_MAPPING[name]})
    return objects


def label_images(dataset_dir: Path) -> List[Tuple[Path, str]]:
    'Return [(image_path, "good"|"defect")] for every ANNOTATED image.'
    images_dir = dataset_dir / "BCCD" / "JPEGImages"
    annotations_dir = dataset_dir / "BCCD" / "Annotations"
    labeled, missing = [], []
    for img_file in sorted(images_dir.glob("*.jpg")):
        xml_file = annotations_dir / f"{img_file.stem}.xml"
        if not xml_file.exists():
            missing.append(img_file.name)
            continue
        try:
            objects = parse_voc_annotation(xml_file)
        except (ET.ParseError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot label {img_file.name}: {exc}. Publishing this dataset "
                f"would record labels the annotations do not support."
            ) from exc
        has_platelets = any(obj['class'] == 'Platelets' for obj in objects)
        labeled.append((img_file, "defect" if has_platelets else "good"))
    if missing:
        logger.warning(
            "Skipping %d image(s) with no annotation file: %s",
            len(missing), ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""),
        )
    counts = Counter(label for _, label in labeled)
    logger.info("Labelled %d images: %d good, %d defect",
                len(labeled), counts["good"], counts["defect"])
    if not labeled:
        raise RuntimeError(f"No annotated images found under {images_dir}")
    return labeled


def content_addressed_split(
    labeled: List[Tuple[Path, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, List[Tuple[Path, str]]]:
    'Group labelled images into splits by their own content hashes.'
    splits: Dict[str, List[Tuple[Path, str]]] = {name: [] for name in SPLITS}
    for path, label in sorted(labeled):
        splits[assign_split(path, train_ratio, val_ratio,
                            scheme_version=SPLIT_SCHEME_VERSION)].append((path, label))
    return splits


def check_split_feasibility(splits: Dict[str, List[Tuple[Path, str]]], context: str) -> None:
    """Raise if any split lacks MIN_IMAGES_PER_CLASS of either class."""
    problems = []
    for name in SPLITS:
        counts = Counter(label for _, label in splits[name])
        if counts["good"] < MIN_IMAGES_PER_CLASS or counts["defect"] < MIN_IMAGES_PER_CLASS:
            problems.append(
                f"  {name}: {counts['good']} good, {counts['defect']} defect "
                f"(need >= {MIN_IMAGES_PER_CLASS} of each)"
            )
    if problems:
        raise ValueError(
            f"{context} cannot support a per-class metric:\n" + "\n".join(problems)
        )


def camera_count_is_feasible(labeled: List[Tuple[Path, str]], num_cameras: int,
                             seed: int) -> bool:
    "True if EVERY camera's every split keeps MIN_IMAGES_PER_CLASS of both classes."
    try:
        partitions = _partition_cameras_unchecked(labeled, num_cameras, seed)
    except ValueError:
        return False
    for cam_labeled in partitions.values():
        splits = content_addressed_split(cam_labeled)
        for name in SPLITS:
            counts = Counter(label for _, label in splits[name])
            if (counts["good"] < MIN_IMAGES_PER_CLASS
                    or counts["defect"] < MIN_IMAGES_PER_CLASS):
                return False
    return True


def max_feasible_cameras(labeled: List[Tuple[Path, str]], seed: int) -> int:
    'Largest camera count that is actually feasible, found by trying each.'
    counts = Counter(label for _, label in labeled)
    scarcest = min(counts["good"], counts["defect"])
    # Each camera needs at least MIN per class per split, i.e. 3 * MIN of the
    # scarcer class at an absolute minimum. Nothing above this can work.
    upper = max(0, scarcest // (3 * MIN_IMAGES_PER_CLASS))
    for candidate in range(upper, 0, -1):
        if camera_count_is_feasible(labeled, candidate, seed):
            return candidate
    return 0


def _partition_cameras_unchecked(
    labeled: List[Tuple[Path, str]], num_cameras: int, seed: int
) -> Dict[str, List[Tuple[Path, str]]]:
    'Deal labelled images round-robin per class into num_cameras partitions.'
    if num_cameras < 1:
        raise ValueError(f"--num-cameras must be >= 1, got {num_cameras}")

    rng = random.Random(seed)
    by_class: Dict[str, List[Path]] = {}
    for path, label in labeled:
        by_class.setdefault(label, []).append(path)

    partitions: Dict[str, List[Tuple[Path, str]]] = {
        f"cam-{i + 1:03d}": [] for i in range(num_cameras)
    }
    names = list(partitions)
    for label, paths in sorted(by_class.items()):
        ordered = sorted(paths)
        rng.shuffle(ordered)
        for idx, path in enumerate(ordered):
            partitions[names[idx % num_cameras]].append((path, label))
    return partitions


def partition_cameras(
    labeled: List[Tuple[Path, str]], num_cameras: int, seed: int
) -> Dict[str, List[Tuple[Path, str]]]:
    """Partition into per-camera datasets, refusing an infeasible count."""
    if num_cameras < 1:
        raise ValueError(f"--num-cameras must be >= 1, got {num_cameras}")
    if not camera_count_is_feasible(labeled, num_cameras, seed):
        counts = Counter(label for _, label in labeled)
        feasible = max_feasible_cameras(labeled, seed)
        raise ValueError(
            f"--num-cameras {num_cameras} is not feasible for this dataset: "
            f"{len(labeled)} labelled images ({counts['good']} good, "
            f"{counts['defect']} defect). Every camera needs at least "
            f"{MIN_IMAGES_PER_CLASS} of BOTH classes in each of its three splits, "
            f"and splits are assigned from image content rather than by ratio, so "
            f"a small partition can land with none of a class. The maximum that "
            f"actually works here is {feasible}."
        )
    return _partition_cameras_unchecked(labeled, num_cameras, seed)


def stage_split(dataset_dir: Path, staging_split_dir: Path,
                entries: List[Tuple[Path, str]]) -> None:
    "Copy a split's images into a FRESH staging directory, preserving .jpg."
    if staging_split_dir.exists():
        shutil.rmtree(staging_split_dir)
    for class_name in ("good", "defect"):
        (staging_split_dir / class_name).mkdir(parents=True, exist_ok=True)
    for img_path, label in entries:
        source = dataset_dir / "BCCD" / "JPEGImages" / img_path.name
        shutil.copy2(source, staging_split_dir / label / img_path.name)


def create_tarball(source_dir: Path, output_path: Path, split_name: str) -> None:
    'Create a REPRODUCIBLE tarball whose members are all under split_name/.'
    def _normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    # gzip mtime=0 and a fixed compression level, so the container is stable too.
    with output_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           compresslevel=9, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                for item in sorted(source_dir.rglob("*")):
                    relative = item.relative_to(source_dir).as_posix()
                    tar.add(item, arcname=f"{split_name}/{relative}",
                            recursive=False, filter=_normalise)


def build_dataset(dataset_dir: Path, commit: str, staging_dir: Path,
                  splits: Dict[str, List[Tuple[Path, str]]], seed: int,
                  camera_id: str = None, train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO) -> dict:
    """Stage every split, verify disjointness, tar them, and return the manifest."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    manifest_splits = {}
    for split_name in SPLITS:
        split_dir = staging_dir / split_name
        stage_split(dataset_dir, split_dir, splits[split_name])
        counts = Counter(label for _, label in splits[split_name])
        manifest_splits[split_name] = {
            "counts": {"good": counts["good"], "defect": counts["defect"]},
            "content_sha256": content_hash(split_dir),
        }

    assert_splits_disjoint({name: staging_dir / name for name in SPLITS})

    (staging_dir / "PROVENANCE.txt").write_text(UPSTREAM_NOTICE.format(commit=commit))

    upstream_license = _find_upstream_license(dataset_dir)
    if upstream_license is None:
        raise RuntimeError(
            f"No LICENSE file found in the cloned BCCD repository at {dataset_dir}. "
            f"Refusing to publish images without the licence they are distributed "
            f"under. Check the upstream repository layout."
        )
    (staging_dir / UPSTREAM_LICENSE_NAME).write_text(
        upstream_license.read_text(encoding="utf-8"), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "format": "anomalib",
        "source": "bccd",
        "generator": "scripts/download_bccd.py",
        "upstream_repo": GITHUB_REPO,
        "upstream_commit": commit,
        "upstream_license": "MIT",
        "label_mapping": "image with >=1 Platelets annotation -> defect, else good",
        "split_scheme": SPLIT_SCHEME,
        "split_scheme_version": SPLIT_SCHEME_VERSION,
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 6),
        },
        "seed": seed,
        "camera_id": camera_id,
        "splits": manifest_splits,
    }
    # Derived through the SHARED helper, so the validator that recomputes this on
    # download cannot drift from the producer that stamps it.
    manifest["dataset_id"] = derive_dataset_id(
        {k: v["content_sha256"] for k, v in manifest_splits.items()}
    )
    manifest_json = json.dumps(manifest, indent=2, allow_nan=False)
    (staging_dir / DATASET_MANIFEST_NAME).write_text(manifest_json)

    for split_name in SPLITS:
        (staging_dir / split_name / DATASET_MANIFEST_NAME).write_text(manifest_json)
        # A split archive is distributed on its own -- it is what a training job
        # receives -- so the licence and provenance travel inside it too.
        for name in (UPSTREAM_LICENSE_NAME, "PROVENANCE.txt"):
            (staging_dir / split_name / name).write_text(
                (staging_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        create_tarball(
            staging_dir / split_name,
            staging_dir / f"{split_name}.tar.gz",
            split_name,
        )
    return manifest


def publish_local(staging_dir: Path, dest_dir: Path, force: bool) -> None:
    'Move a staged dataset to its local destination.'
    if dest_dir.exists() or dest_dir.is_symlink():
        if any(dest_dir.iterdir()) if dest_dir.is_dir() else True:
            if not force:
                raise RuntimeError(
                    f"Destination {dest_dir} is not empty. Pass --force to REPLACE it "
                    "(contents are deleted, not merged -- merging would leave images "
                    "from a previous run in their former splits)."
                )
        # Empty needs no --force -- there is nothing to lose -- but it must still go, or the
        # move nests inside it.
        if dest_dir.is_symlink() or not dest_dir.is_dir():
            dest_dir.unlink()
        else:
            shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_dir), str(dest_dir))
    if not (dest_dir / "train.tar.gz").exists():
        raise RuntimeError(
            f"Published {dest_dir} but {dest_dir / 'train.tar.gz'} is absent. The move did not "
            f"land where later steps read from; contents are: "
            f"{sorted(p.name for p in dest_dir.iterdir())}"
        )


def resolve_bucket(region: str) -> str:
    'Read the pipeline bucket name from the deployed stack output.'
    cfn = boto3.client("cloudformation", region_name=region)
    outputs = cfn.describe_stacks(StackName=PIPELINE_STACK_NAME)["Stacks"][0]["Outputs"]
    for output in outputs:
        if output["OutputKey"] == OUTPUT_BUCKET_NAME:
            return output["OutputValue"]
    raise RuntimeError(
        f"{OUTPUT_BUCKET_NAME} output not found on {PIPELINE_STACK_NAME}; deploy the "
        "stack first (see README Deployment)."
    )


def _put_create_only(s3, bucket: str, key: str, path: Path) -> bool:
    'Upload *path* to *key* only if that key does not already exist.'
    body = path.read_bytes()
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
            raise
        existing = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if existing == body:
            logger.info("s3://%s/%s already published with identical content",
                        bucket, key)
            return False
        raise RuntimeError(
            f"s3://{bucket}/{key} already exists with DIFFERENT content "
            f"({len(existing)} bytes published, {len(body)} bytes local). The "
            f"dataset prefix is supposed to identify its contents, so this means "
            f"either the prefix derivation changed or an unrelated dataset was "
            f"published here. Refusing to overwrite: consumers pinned to this URI "
            f"would silently change datasets."
        ) from exc


def publish_s3(local_dir: Path, bucket: str, prefix: str, region: str) -> str:
    'Upload a dataset under an immutable prefix, writing _READY last.'
    s3 = boto3.client("s3", region_name=region)
    marker_key = f"{prefix}/{DATASET_READY_MARKER}"
    try:
        s3.head_object(Bucket=bucket, Key=marker_key)
        differing = []
        for name in (DATASET_MANIFEST_NAME, "PROVENANCE.txt", UPSTREAM_LICENSE_NAME):
            local = local_dir / name
            if not local.is_file():
                continue
            published = s3.get_object(Bucket=bucket, Key=f"{prefix}/{name}")["Body"].read()
            if published != local.read_bytes():
                differing.append(name)
        for split_name in SPLITS:
            key = f"{prefix}/anomalib/{split_name}/{split_name}.tar.gz"
            published = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            if published != (local_dir / f"{split_name}.tar.gz").read_bytes():
                differing.append(f"{split_name}.tar.gz")
        if differing:
            raise RuntimeError(
                f"s3://{bucket}/{prefix} is already published and marked ready, but "
                f"{len(differing)} object(s) differ from what was just built: "
                f"{', '.join(differing)}. The dataset id covers image content only, "
                f"so the same images built with a different seed, camera id or "
                f"upstream commit share this prefix while carrying different "
                f"metadata. Refusing to report success for content that is not "
                f"published here: consumers pinned to this URI would read the other "
                f"build's metadata. This CLI has no destination override, so recover by "
                f"rebuilding into a FRESH --output-dir without --upload-s3 to compare, "
                f"then either accept the published version or delete this prefix "
                f"deliberately and republish. See \"Republishing the same images\" in "
                f"README.md."
            )
        logger.info("Dataset already published at s3://%s/%s; verified all objects "
                    "match", bucket, prefix)
        return f"s3://{bucket}/{prefix}"
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("403", "AccessDenied", "Forbidden"):
            raise RuntimeError(
                f"Cannot determine whether s3://{bucket}/{marker_key} exists: "
                f"access denied. Refusing to publish, because a denied read is not "
                f"evidence that the dataset is absent, and overwriting a published "
                f"immutable dataset would break every consumer pinned to it. Grant "
                f"s3:GetObject on the dataset prefix and retry."
            ) from exc
        if code not in ("404", "NoSuchKey", "NotFound"):
            raise

    created, existed = 0, 0
    for split_name in SPLITS:
        key = f"{prefix}/anomalib/{split_name}/{split_name}.tar.gz"
        logger.info("Uploading s3://%s/%s", bucket, key)
        if _put_create_only(s3, bucket, key, local_dir / f"{split_name}.tar.gz"):
            created += 1
        else:
            existed += 1
    for name in (DATASET_MANIFEST_NAME, "PROVENANCE.txt", UPSTREAM_LICENSE_NAME):
        if _put_create_only(s3, bucket, f"{prefix}/{name}", local_dir / name):
            created += 1
        else:
            existed += 1

    try:
        s3.put_object(Bucket=bucket, Key=marker_key, Body=b"ready\n",
                      IfNoneMatch="*")
        logger.info("Published s3://%s/%s (%d objects created, %d already present)",
                    bucket, prefix, created, existed)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
            raise
        logger.info("Another publisher completed s3://%s/%s first; content is "
                    "identical, so both agree on one publication", bucket, prefix)
    return f"s3://{bucket}/{prefix}"


def main():
    parser = argparse.ArgumentParser(description="Download and prepare BCCD dataset")
    parser.add_argument("--output-dir", default="./data/bccd", help="Output directory")
    parser.add_argument("--num-cameras", type=int,
                        help="Number of per-camera datasets to create")
    parser.add_argument("--upload-s3", action="store_true", help="Publish to the pipeline bucket")
    parser.add_argument("--region", default=None,
                        help="AWS Region (defaults to the configured profile Region)")
    parser.add_argument("--upstream-ref", default=None,
                        help="Pin the upstream BCCD commit SHA or tag. Without it the "
                             "MOVING branch tip is cloned, so the same command can "
                             "produce different datasets on different days. The resolved "
                             "commit is recorded either way.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for CAMERA PARTITIONING only. Split assignment is "
                             "content-addressed and takes no seed: a seed there would "
                             "move images between splits across dataset versions, "
                             "which is what contaminates a warm-started model's holdout.")
    parser.add_argument("--force", action="store_true",
                        help="Replace a non-empty output directory (contents deleted, not merged)")
    args = parser.parse_args()

    try:
        region = resolve_region(args.region)
    except AwsEnvError as exc:
        if args.upload_s3:
            raise SystemExit(str(exc))
        region = None  # only needed for publication

    output_dir = Path(args.output_dir)
    bucket = resolve_bucket(region) if args.upload_s3 else None
    published = []
    index_entries = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir, commit = clone_dataset(temp_path, args.upstream_ref)
        labeled = label_images(dataset_dir)

        if args.num_cameras:
            logger.info("Maximum feasible cameras for this dataset: %d",
                        max_feasible_cameras(labeled, args.seed))
            partitions = partition_cameras(labeled, args.num_cameras, args.seed)
            targets = []
            for cam_id, cam_labeled in partitions.items():
                splits = content_addressed_split(cam_labeled)
                check_split_feasibility(splits, f"camera {cam_id}")
                targets.append((cam_id, splits))
        else:
            splits = content_addressed_split(labeled)
            check_split_feasibility(splits, "dataset")
            targets = [(None, splits)]

        for cam_id, splits in targets:
            label = cam_id or "single"
            staging = temp_path / f"staging-{label}"
            manifest = build_dataset(
                dataset_dir, commit, staging, splits, args.seed, camera_id=cam_id
            )
            dataset_id = manifest["dataset_id"]
            dest = output_dir / (cam_id or "dataset") / dataset_id
            publish_local(staging, dest, force=args.force)
            logger.info("Built %s dataset_id=%s at %s", label, dataset_id, dest)

            uri = None
            if args.upload_s3:
                prefix = f"{DATASETS_PREFIX}/{dataset_id}"
                uri = publish_s3(dest, bucket, prefix, region)
                published.append((label, uri))
            index_entries.append((cam_id or "default", dataset_id, dest, uri))

    index = {
        "schema_version": 1,
        "format": "anomalib",
        "source": "bccd",
        "cameras": {
            label: {
                "dataset_id": dataset_id,
                "local_path": str(local_path.resolve()),
                "s3_uri": uri,
            }
            for label, dataset_id, local_path, uri in index_entries
        },
    }
    index_path = output_dir / PUBLICATION_INDEX_NAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("Publication index: %s (%d camera(s))", index_path, len(index["cameras"]))
    if bucket:
        boto3.client("s3", region_name=region).put_object(
            Bucket=bucket,
            Key=f"{DATASETS_PREFIX}/{PUBLICATION_INDEX_NAME}",
            Body=index_path.read_bytes(),
        )
        logger.info("Publication index uploaded to s3://%s/%s/%s",
                    bucket, DATASETS_PREFIX, PUBLICATION_INDEX_NAME)

    logger.info("Dataset preparation complete")

    if published:
        print("\nPublished datasets. TrainingDataS3Uri must be the prefix ABOVE the")
        print("format directory -- preprocess_cv.py appends /anomalib/<split>/<split>.tar.gz:\n")
        for label, uri in published:
            print(f"  {label}: {uri}")
        first_label, first_uri = published[0]
        print("\nRun the pipeline against it with:\n")
        print("  aws sagemaker start-pipeline-execution \\")
        print(f"    --pipeline-name {PIPELINE_NAME} \\")
        print(f"    --region {region} \\")
        if first_label != "single":
            print(f"    --pipeline-parameters "
                  f"'[{{\"Name\":\"TrainingDataS3Uri\",\"Value\":\"{first_uri}\"}},"
                  f"{{\"Name\":\"CameraId\",\"Value\":\"{first_label}\"}}]'")
        else:
            print(f"    --pipeline-parameters "
                  f"'[{{\"Name\":\"TrainingDataS3Uri\",\"Value\":\"{first_uri}\"}}]'")


if __name__ == "__main__":
    main()
