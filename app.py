#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'CDK app entry point.'

import sys

import aws_cdk as cdk

from aws_env import AwsEnvError, resolve_env
from config import ORCHESTRATION_STACK_NAME, PIPELINE_NAME, PIPELINE_STACK_NAME
from sagemaker_cv_pipeline_stack import SageMakerCVPipelineStack

def build_app(account, region, *, context=None, outdir=None):
    """Build the core stack and optional fleet extension in the same environment."""
    app = cdk.App(context=context, outdir=outdir)
    env = cdk.Environment(account=account, region=region)
    fleet = str(app.node.try_get_context("fleet") or "false").lower()
    if fleet not in ("true", "false"):
        raise ValueError("Use -c fleet=true or -c fleet=false.")
    max_concurrency = int(app.node.try_get_context("max_concurrency") or "1")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")
    pipeline_stack = SageMakerCVPipelineStack(
        app,
        PIPELINE_STACK_NAME,
        env=env,
        description="SageMaker SDK v3 CV pipeline: training, evaluation, Neo compilation, and registration",
    )
    # Add fleet scheduling after trying the core ML lifecycle. See docs/fleet.md.
    if fleet == "true":
        from orchestration_stack import OrchestrationStack

        orchestration_stack = OrchestrationStack(
            app,
            ORCHESTRATION_STACK_NAME,
            env=env,
            description="Optional fleet scheduling with bounded concurrency and SNS notifications",
            max_concurrency=max_concurrency,
            pipeline_name=PIPELINE_NAME,
        )
        orchestration_stack.add_stack_dependency(pipeline_stack)
    return app


if __name__ == "__main__":
    try:
        account, region = resolve_env()
    except AwsEnvError as exc:
        sys.exit(str(exc))
    build_app(account, region).synth()
