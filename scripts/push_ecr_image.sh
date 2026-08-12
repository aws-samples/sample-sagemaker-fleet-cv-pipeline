#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -euo pipefail

# Build and push the custom training image to ECR using docker buildx.
#
# The SageMaker pipeline references:
#   <account>.dkr.ecr.<region>.amazonaws.com/sm1k-custom-training:latest
#
# Usage:
#   AWS_REGION=us-west-2 ./scripts/push_ecr_image.sh
#   ./scripts/push_ecr_image.sh --region us-west-2 --tag 20260209-123000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
TAG="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="${2:-}"; shift 2 ;;
    --tag)
      TAG="${2:-}"; shift 2 ;;
    -*)
      echo "Unknown option: $1" >&2; exit 2 ;;
    *)
      echo "Unexpected argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${REGION}" ]]; then
  echo "AWS region not set. Set AWS_REGION or pass --region." >&2
  exit 2
fi

command -v aws >/dev/null 2>&1 || { echo "aws CLI not found in PATH" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "docker not found in PATH" >&2; exit 2; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
ECR_REPO="sm1k-custom-training"
ECR_URI="${ECR_REGISTRY}/${ECR_REPO}"

echo "[ecr] account=${ACCOUNT_ID} region=${REGION} repo=${ECR_REPO}"

# The ECR repo is created by the CDK stack. Fail fast if it does not exist —
# deploy the stack first rather than creating a repo that collides with it.
if ! aws ecr describe-repositories --region "${REGION}" --repository-names "${ECR_REPO}" >/dev/null 2>&1; then
  echo "ECR repository '${ECR_REPO}' not found in ${REGION}." >&2
  echo "Deploy the CDK stacks first (cdk deploy --all), then re-run this script." >&2
  exit 2
fi

echo "[ecr] docker login"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}" >/dev/null

echo "[buildx] ensure builder"
docker buildx create --use --name sm1k-builder >/dev/null 2>&1 || docker buildx use sm1k-builder
docker buildx inspect --bootstrap >/dev/null

echo "[buildx] build+push ${ECR_URI}:${TAG} (also tags :latest if TAG != latest)"
if [[ "${TAG}" == "latest" ]]; then
  /usr/bin/time -p docker buildx build \
    --platform linux/amd64 \
    -f docker/training/Dockerfile \
    -t "${ECR_URI}:latest" \
    --push \
    .
else
  /usr/bin/time -p docker buildx build \
    --platform linux/amd64 \
    -f docker/training/Dockerfile \
    -t "${ECR_URI}:${TAG}" \
    -t "${ECR_URI}:latest" \
    --push \
    .
fi

echo "[ecr] verify"
aws ecr describe-images --region "${REGION}" --repository-name "${ECR_REPO}" --image-ids imageTag=latest \
  --query 'imageDetails[0].imageDigest' --output text

