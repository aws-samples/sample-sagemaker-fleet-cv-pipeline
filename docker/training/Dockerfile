# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# ECR Public mirror of the Docker Official python image — avoids Docker Hub
# rate limits and pulls from an AWS-operated registry.
FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /opt/ml/code

RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Tested CPU dependency set. Keep torch and torchvision paired; revalidate both
# ONNX exports and Neo compilation after changing these versions.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 && \
    pip install --no-cache-dir numpy==1.26.4 Pillow==12.3.0 boto3==1.34.162 onnx==1.16.2 onnxruntime==1.18.1

# Pre-cache ImageNet weights so training jobs need no internet egress.
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"

COPY src/model/ /opt/ml/code/
COPY src/utils/ /opt/ml/code/utils/
# Compilation job identity is shared with the lease manager, which recovers a child
# compilation job by reconstructing its name. The Lambda bundle is src/orchestration, so
# the module lives there and is copied in here rather than duplicated.
COPY src/orchestration/neo_jobs.py /opt/ml/code/neo_jobs.py
COPY docker/training/entrypoint.py /opt/ml/code/entrypoint.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/ml/code

ENTRYPOINT ["python", "/opt/ml/code/entrypoint.py"]
