#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import aws_cdk as cdk
from sagemaker_cv_pipeline_stack import SageMakerCVPipelineStack
from orchestration_stack import OrchestrationStack

app = cdk.App()

# CV pipeline: anomaly detection + defect classification with ONNX evaluation
SageMakerCVPipelineStack(
    app,
    "SageMakerCVPipelineStack",
    description="SageMaker SDK v3 CV Pipeline: Anomalib + classifier training, ONNX evaluation, and Model Registry"
)

# Orchestration: SQS → Step Functions with concurrency control
OrchestrationStack(
    app,
    "OrchestrationStack",
    description="Pipeline orchestration: SQS queue, Step Functions concurrency control, SNS notifications",
    max_concurrency=5,
)

app.synth()