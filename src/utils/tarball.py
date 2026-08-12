# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tarball extraction helpers shared across model scripts."""
import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_tarball(tar_path: Path, output_dir: Path):
    """Extract a single .tar.gz / .tgz archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir, filter="data")


def extract_tarballs(channel_dir: Path, split_name: str) -> Path:
    """Extract SageMaker channel tarballs and return the resolved split directory."""
    channel_dir.mkdir(parents=True, exist_ok=True)

    split_dir_in_channel = channel_dir / split_name
    if split_dir_in_channel.exists():
        return split_dir_in_channel

    tar_candidates = list(channel_dir.rglob("*.tar.gz")) + list(channel_dir.rglob("*.tgz"))
    for tar_path in tar_candidates:
        extract_target = channel_dir / f"extracted_{split_name}"
        extract_target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_target, filter="data")
        logger.info("Extracted %s to %s", tar_path, extract_target)
        split_dir = extract_target / split_name
        if split_dir.exists():
            return split_dir
        return extract_target
    return channel_dir