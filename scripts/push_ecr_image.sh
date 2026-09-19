#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

# Build and push the custom training image to ECR using docker buildx.
#
# The SageMaker pipeline references the repository by DIGEST when one is pinned:
#   cdk deploy --all -c training_image_digest=sha256:<64 hex>
# and falls back to <account>.dkr.ecr.<region>.amazonaws.com/fleet-cv-custom-training:latest
# when it is not. This script prints the digest to pin after every push.
#
# A tagged push does NOT move :latest unless you ask for it. Moving it changes the
# image every unpinned deployment pulls, which is a publishing decision, not a
# side effect of building.
#
# Usage:
#   AWS_REGION=us-west-2 ./scripts/push_ecr_image.sh
#   ./scripts/push_ecr_image.sh --region us-west-2 --tag 20260209-123000
#   ./scripts/push_ecr_image.sh --tag 20260209-123000 --move-latest

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Region resolution is delegated to aws_env.py so this script cannot disagree
# with the Python entry points. It previously preferred AWS_REGION over
# AWS_DEFAULT_REGION and silently accepted a conflict that aws_env rejects, while
# also refusing a profile-only configuration that the Python paths accept.
# An explicit --region still wins; it is passed through as a genuine override.
REGION=""
TAG="latest"
MOVE_LATEST="0"
# Optional path to write THIS build's digest to, so a caller can pin it without a tag lookup.
# The runbook's migration block consumed the digest by resolving the tag afterwards, which is
# the same mutable-tag race this script was changed to avoid -- another publisher moving that
# tag between the push and the lookup made the reader deploy a different build.
DIGEST_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="${2:-}"; shift 2 ;;
    --tag)
      TAG="${2:-}"; shift 2 ;;
    --move-latest)
      MOVE_LATEST="1"; shift ;;
    --digest-file)
      DIGEST_FILE="${2:-}"; shift 2 ;;
    -*)
      echo "Unknown option: $1" >&2; exit 2 ;;
    *)
      echo "Unexpected argument: $1" >&2; exit 2 ;;
  esac
done

if ! REGION="$(python3 -c '
import sys
sys.path.insert(0, "'"${ROOT}"'")
from aws_env import AwsEnvError, resolve_region
try:
    print(resolve_region(sys.argv[1] or None))
except AwsEnvError as exc:
    sys.exit(str(exc))
' "${REGION}")"; then
  echo "Could not resolve the AWS Region (see aws_env.py)." >&2
  exit 2
fi
echo "[ecr] region=${REGION} (resolved via aws_env.py)"

# Print a cdk command that CARRIES the Region we just resolved. Without this the script can
# publish to one Region and then tell the reader to deploy in whatever their ambient
# configuration selects -- which either updates a different deployment or pins a digest that
# does not exist in that Region's repository. %q keeps argument boundaries intact.
print_cdk_command() {
  printf '  AWS_REGION=%q AWS_DEFAULT_REGION=%q CDK_DEFAULT_REGION=%q cdk' \
    "${REGION}" "${REGION}" "${REGION}"
  printf ' %q' "$@"
  printf '\n'
}

command -v aws >/dev/null 2>&1 || { echo "aws CLI not found in PATH" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "docker not found in PATH" >&2; exit 2; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
# Read the repository name from config.py so it cannot drift from the name the
# CDK stack deploys and the pipeline references.
ECR_REPO="$(python3 -c 'import sys; sys.path.insert(0, "'"${ROOT}"'"); import config; print(config.TRAINING_REPO_NAME)')"
if [[ -z "${ECR_REPO}" ]]; then
  echo "Could not read TRAINING_REPO_NAME from config.py" >&2
  exit 2
fi
ECR_URI="${ECR_REGISTRY}/${ECR_REPO}"

echo "[ecr] account=${ACCOUNT_ID} region=${REGION} repo=${ECR_REPO}"

# The ECR repo is created by the CDK stack. Fail fast if it does not exist —
# deploy the stack first rather than creating a repo that collides with it.
if ! aws ecr describe-repositories --region "${REGION}" --repository-names "${ECR_REPO}" >/dev/null 2>&1; then
  echo "ECR repository '${ECR_REPO}' not found in ${REGION}." >&2
  echo "Deploy the CDK stacks first, then re-run this script. On a FIRST deployment the" >&2
  echo "pipeline definition cannot be built yet, so bootstrap mode is required:" >&2
  print_cdk_command deploy --all -c bootstrap=true >&2
  echo "Subsequent deployments pin the digest this script prints:" >&2
  print_cdk_command deploy --all -c training_image_digest=REPLACE_WITH_DIGEST >&2
  exit 2
fi

echo "[ecr] docker login"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}" >/dev/null

echo "[buildx] ensure builder"
docker buildx create --use --name fleet-cv-builder >/dev/null 2>&1 || docker buildx use fleet-cv-builder
docker buildx inspect --bootstrap >/dev/null

echo "[buildx] build+push ${ECR_URI}:${TAG}"
BUILD_TAGS=(-t "${ECR_URI}:${TAG}")
if [[ "${TAG}" != "latest" ]] && [[ "${MOVE_LATEST}" == "1" ]]; then
  # Moving `latest` changes the image every unpinned deployment will pull, so it is
  # an explicit publishing choice rather than a side effect of building. It used to
  # happen on every tagged push: a later build silently changed the code a
  # subsequent training job ran, with no change to the pipeline definition, and
  # reverting the definition did not restore the previous image because the tag had
  # already moved.
  echo "[buildx] ALSO moving :latest (--move-latest)"
  BUILD_TAGS+=(-t "${ECR_URI}:latest")
fi

# Where buildx records what this invocation actually produced.
#
# The template carries explicit X's because GNU mktemp REQUIRES them -- `mktemp -t name` is a
# BSD/macOS extension, and coreutils fails with "too few X's in template". Under `set -e` that
# aborted the script at this assignment, before docker buildx ran at all, which broke the
# README's deployment step on an ordinary Linux host.
BUILDX_METADATA="$(mktemp -t buildx-metadata.XXXXXXXX)"
trap 'rm -f "${BUILDX_METADATA}"' EXIT

/usr/bin/time -p docker buildx build \
  --platform linux/amd64 \
  -f docker/training/Dockerfile \
  "${BUILD_TAGS[@]}" \
  --metadata-file "${BUILDX_METADATA}" \
  --push \
  .

# Take the digest from THIS build's metadata, not by looking the tag up afterwards.
#
# `describe-images --image-ids imageTag=...` resolves a MUTABLE tag, so a concurrent publisher
# that moved the same tag between the push and the lookup made this script print THEIR digest
# and then hand the reader a deploy command describing it as "this exact image". Digest pinning
# still worked -- it pinned the wrong build, and could pair this checkout's uploaded processing
# scripts with a container holding another revision's shared modules.
#
# containerimage.digest is what the push actually produced.
DIGEST="$(python3 -c '
import json, sys
with open(sys.argv[1]) as handle:
    meta = json.load(handle)
digest = meta.get("containerimage.digest")
if not digest:
    sys.exit("buildx metadata has no containerimage.digest: " + ", ".join(sorted(meta)))
print(digest)
' "${BUILDX_METADATA}")"

echo "[ecr] verify ${DIGEST}"
# Verified BY DIGEST. Confirms this build is present in the registry without reintroducing a
# tag lookup, so a moved tag cannot change the answer.
aws ecr describe-images --region "${REGION}" --repository-name "${ECR_REPO}" \
  --image-ids "imageDigest=${DIGEST}" --query 'imageDetails[0].imageDigest' --output text \
  > /dev/null || {
  echo "ERROR: ${DIGEST} is not present in ${ECR_REPO}; the push did not land." >&2
  exit 1
}
echo "${DIGEST}"
if [[ -n "${DIGEST_FILE}" ]]; then
  printf '%s' "${DIGEST}" > "${DIGEST_FILE}"
  echo "[digest] wrote ${DIGEST_FILE}"
fi
echo
echo "Pin the pipeline to this exact image -- a tag can move, a digest cannot:"
print_cdk_command deploy --all -c "training_image_digest=${DIGEST}"
if [[ "${TAG}" == "latest" ]] || [[ "${MOVE_LATEST}" == "1" ]]; then
  echo
  echo "NOTE: :latest now points at ${DIGEST}. Any deployment WITHOUT"
  echo "      -c training_image_digest will pull this image from now on."
fi
