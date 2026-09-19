#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import json
import boto3
from aws_cdk import (
    Stack,
    aws_sagemaker as sm,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ecr as ecr,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import InputData, Compute, OutputDataConfig
from sagemaker.core.training.configs import MetricDefinition
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.model_metrics import ModelMetrics, MetricsSource
from sagemaker.core.shapes import ProcessingOutput, ProcessingS3Output, ProcessingInput, ProcessingS3Input
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.core.workflow.pipeline_context import PipelineSession

import config
from config import (
    ANOMALY_MODEL_GROUP,
    CLASSIFIER_MODEL_GROUP,
    OUTPUT_BUCKET_NAME,
    OUTPUT_EXECUTION_ROLE_ARN,
    OUTPUT_PIPELINE_NAME,
    OUTPUT_TRAINING_REPO_URI,
    PIPELINE_NAME,
    RUNS_PREFIX,
    TRAINING_REPO_NAME,
)

class SageMakerCVPipelineStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ECR repository for the custom training image referenced by the pipeline.
        # This allows a brand-new account to deploy without manually creating ECR.
        self.training_repo = ecr.Repository(
            self,
            "CustomTrainingRepo",
            repository_name=TRAINING_REPO_NAME,
            image_scan_on_push=True,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
        )

        # Create Model Registry groups up-front. RegisterModel steps require these to exist at execution time.
        self.anomaly_model_group = sm.CfnModelPackageGroup(
            self,
            "AnomalyModelPackageGroup",
            model_package_group_name=ANOMALY_MODEL_GROUP,
            model_package_group_description="fleet-cv anomaly detection model packages (ONNX)",
        )

        self.classifier_model_group = sm.CfnModelPackageGroup(
            self,
            "ClassifierModelPackageGroup",
            model_package_group_name=CLASSIFIER_MODEL_GROUP,
            model_package_group_description="fleet-cv defect classifier model packages (ONNX)",
        )

        # Create S3 bucket for pipeline artifacts. See config.bucket_name for why
        # the name is account- and region-qualified (uniqueness, not secrecy).
        self.pipeline_bucket = s3.Bucket(
            self,
            "CVPipelineBucket",
            bucket_name=config.bucket_name(self.account, self.region),
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )

        stack = Stack.of(self)
        # Region-qualified: IAM role names are account-global, so a fixed name
        # blocks deploying this stack to a second region in the same account.
        role_name = config.execution_role_name(stack.region)
        model_group_arns = [
            f"arn:aws:sagemaker:{stack.region}:{stack.account}:model-package-group/{ANOMALY_MODEL_GROUP}",
            f"arn:aws:sagemaker:{stack.region}:{stack.account}:model-package-group/{CLASSIFIER_MODEL_GROUP}",
        ]
        model_package_arns = [
            f"arn:aws:sagemaker:{stack.region}:{stack.account}:model-package/{ANOMALY_MODEL_GROUP}/*",
            f"arn:aws:sagemaker:{stack.region}:{stack.account}:model-package/{CLASSIFIER_MODEL_GROUP}/*",
        ]
        self.sagemaker_role = iam.Role(
            self,
            "SageMakerCVExecutionRole",
            role_name=role_name,
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            inline_policies={
                "PipelineBucketAccess": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:ListBucket",
                            ],
                            resources=[
                                self.pipeline_bucket.bucket_arn,
                                f"{self.pipeline_bucket.bucket_arn}/*",
                            ],
                        ),
                    ]
                ),
                "TrainingImagePull": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            # GetAuthorizationToken does not support resource-level permissions
                            actions=["ecr:GetAuthorizationToken"],
                            resources=["*"],
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "ecr:BatchGetImage",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchCheckLayerAvailability",
                            ],
                            resources=[self.training_repo.repository_arn],
                        ),
                    ]
                ),
                "SageMakerTrainingRuntime": iam.PolicyDocument(
                    statements=[
                        # Image pull IS resource-scopeable: this role only ever pulls
                        # the one image this stack builds.
                        iam.PolicyStatement(
                            actions=[
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:BatchGetImage",
                                "ecr:GetDownloadUrlForLayer",
                            ],
                            resources=[self.training_repo.repository_arn],
                        ),
                        iam.PolicyStatement(
                            actions=["cloudwatch:PutMetricData"],
                            resources=["*"],
                            conditions={
                                "StringEqualsIfExists": {
                                    "cloudwatch:namespace": [
                                        "/aws/sagemaker/TrainingJobs",
                                        "/aws/sagemaker/ProcessingJobs",
                                    ]
                                }
                            },
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "ec2:CreateNetworkInterface",
                                "ec2:CreateNetworkInterfacePermission",
                                "ec2:DeleteNetworkInterface",
                                "ec2:DeleteNetworkInterfacePermission",
                                "ec2:DescribeDhcpOptions",
                                "ec2:DescribeNetworkInterfaces",
                                "ec2:DescribeSecurityGroups",
                                "ec2:DescribeSubnets",
                                "ec2:DescribeVpcs",
                            ],
                            resources=["*"],
                        ),
                    ]
                ),
                "SageMakerJobLogs": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                            ],
                            resources=[
                                f"arn:aws:logs:{stack.region}:{stack.account}:log-group:/aws/sagemaker/*",
                                f"arn:aws:logs:{stack.region}:{stack.account}:log-group:/aws/sagemaker/*:log-stream:*",
                            ],
                        ),
                    ]
                ),
                "SageMakerPipelineSteps": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "sagemaker:CreateTrainingJob",
                                "sagemaker:DescribeTrainingJob",
                                "sagemaker:StopTrainingJob",
                                "sagemaker:AddTags",
                            ],
                            resources=[
                                f"arn:aws:sagemaker:{stack.region}:{stack.account}:training-job/*",
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "sagemaker:CreateProcessingJob",
                                "sagemaker:DescribeProcessingJob",
                                "sagemaker:StopProcessingJob",
                                "sagemaker:AddTags",
                            ],
                            resources=[
                                f"arn:aws:sagemaker:{stack.region}:{stack.account}:processing-job/*",
                            ],
                        ),
                        iam.PolicyStatement(
                            # RegisterModel steps call CreateModelPackageGroup
                            # (idempotent when the group exists) and AddTags on
                            # the group before registering the package.
                            actions=[
                                "sagemaker:CreateModelPackage",
                                "sagemaker:DescribeModelPackage",
                                "sagemaker:ListModelPackages",
                                "sagemaker:UpdateModelPackage",
                                "sagemaker:CreateModelPackageGroup",
                                "sagemaker:DescribeModelPackageGroup",
                                "sagemaker:AddTags",
                            ],
                            resources=model_group_arns + model_package_arns,
                        ),
                        iam.PolicyStatement(
                            # Neo compilation steps (CompileAnomalibNeo /
                            # CompileClassifierNeo) submit and poll
                            # compilation jobs from inside processing jobs.
                            actions=[
                                "sagemaker:CreateCompilationJob",
                                "sagemaker:DescribeCompilationJob",
                                "sagemaker:StopCompilationJob",
                            ],
                            resources=[
                                f"arn:aws:sagemaker:{stack.region}:{stack.account}:compilation-job/*",
                            ],
                        ),
                    ]
                ),
                "PassRoleToSageMaker": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["iam:PassRole"],
                            resources=[
                                f"arn:aws:iam::{stack.account}:role/{role_name}",
                            ],
                            conditions={
                                "StringEquals": {
                                    "iam:PassedToService": "sagemaker.amazonaws.com",
                                },
                            },
                        ),
                    ]
                ),
            },
        )

        if not config.bootstrap_only(self.node.try_get_context("bootstrap")):
            pipeline_definition = self._create_cv_pipeline_definition(
                account=self.account, region=self.region
            )

            self.pipeline = sm.CfnPipeline(
                self,
                "SageMakerCVPipeline",
                pipeline_name=PIPELINE_NAME,
                pipeline_definition={"PipelineDefinitionBody": pipeline_definition},
                role_arn=self.sagemaker_role.role_arn,
            )
            CfnOutput(self, OUTPUT_PIPELINE_NAME, value=self.pipeline.pipeline_name)

        # Outputs
        CfnOutput(self, OUTPUT_BUCKET_NAME, value=self.pipeline_bucket.bucket_name)
        CfnOutput(self, OUTPUT_EXECUTION_ROLE_ARN, value=self.sagemaker_role.role_arn)
        CfnOutput(self, OUTPUT_TRAINING_REPO_URI, value=self.training_repo.repository_uri)

    def _create_cv_pipeline_definition(self, account: str, region: str) -> str:
        # Deliberately local: see "NOTE ON IMPORT SAFETY" at the top of this file.
        # These two modules build boto3 clients at import time, so they must not be
        # reached by the bootstrap deploy or by any offline test.
        from sagemaker.mlops.workflow.condition_step import ConditionStep  # noqa: PLC0415
        from sagemaker.mlops.workflow.fail_step import FailStep  # noqa: PLC0415
        from sagemaker.mlops.workflow.model_step import ModelStep  # noqa: PLC0415
        from sagemaker.mlops.workflow.pipeline import Pipeline  # noqa: PLC0415
        from sagemaker.mlops.workflow.steps import (  # noqa: PLC0415
            CacheConfig,
            ProcessingStep,
            TrainingStep,
        )
        from sagemaker.serve.model_builder import ModelBuilder  # noqa: PLC0415


        for label, value in (("account", account), ("region", region)):
            if not isinstance(value, str) or "${Token" in value or not value:
                raise ValueError(
                    f"Unresolved {label} ({value!r}) while building the pipeline "
                    "definition. Construct the stack with an explicit environment, "
                    "e.g. SageMakerCVPipelineStack(app, name, "
                    "env=cdk.Environment(account=..., region=...)); app.py does "
                    "this by resolving both once. Without it the definition would "
                    "embed a CDK token in the execution role ARN."
                )

        boto_session = boto3.session.Session(region_name=region)
        concrete_region = region
        # IMPORTANT: Use a concrete role ARN string in the pipeline definition.
        # If we pass a CDK token here (from a Role construct), SageMaker SDK may attempt to
        # resolve it via IAM APIs during synthesis and fail.
        role_arn = f"arn:aws:iam::{account}:role/{config.execution_role_name(region)}"

        # Concrete bucket name, identical to the CDK bucket above.
        bucket_name = config.bucket_name(account, region)

        pipeline_session = PipelineSession(
            boto_session=boto_session, default_bucket=bucket_name
        )

        # Pipeline parameters
        training_instance_type = ParameterString(
            name="TrainingInstanceType", default_value="ml.m5.xlarge"
        )
        processing_instance_type = ParameterString(
            name="ProcessingInstanceType", default_value="ml.m5.xlarge"
        )
        min_auroc = ParameterFloat(
            name="MinAUROC", default_value=config.DEFAULT_MIN_AUROC
        )
        min_accuracy = ParameterFloat(name="MinBalancedAccuracy", default_value=0.6)
        min_defect_recall = ParameterFloat(name="MinDefectRecall", default_value=0.5)
        # Fleet lineage: identifies which camera/device this execution trains.
        # Attached to registered model packages and used by warm-start lookup,
        # so each camera only ever warm-starts from its own model history.
        camera_id = ParameterString(name="CameraId", default_value="default")
        pinned_digest = config.training_image_digest(
            self.node.try_get_context("training_image_digest")
        )
        repo_uri = (
            f"{account}.dkr.ecr.{concrete_region}.amazonaws.com/{TRAINING_REPO_NAME}"
        )
        custom_training_image_uri = (
            f"{repo_uri}@{pinned_digest}" if pinned_digest else f"{repo_uri}:latest"
        )
        num_samples = ParameterInteger(name="NumSamples", default_value=400)
        training_data_s3_uri = ParameterString(name="TrainingDataS3Uri", default_value="none")

        # Cache configuration
        cache_config = CacheConfig(enable_caching=False, expire_after="30d")

        # Execution-scoped S3 prefix. Static prefixes let concurrent fleet
        # executions overwrite or consume each other's processing outputs;
        # embedding the execution ID isolates every run.
        def run_prefix(*suffix: str):
            return Join(
                on="/",
                values=[
                    f"s3://{bucket_name}/{RUNS_PREFIX}",
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    *suffix,
                ],
            )

        cv_processor = ScriptProcessor(
            image_uri=custom_training_image_uri,
            command=["python3"],
            instance_type=processing_instance_type,
            instance_count=1,  # Use fixed value instead of parameter
            base_job_name="cv-pipeline-preprocess",
            sagemaker_session=pipeline_session,
            role=role_arn,
        )

        cv_processor_anomalib_args = cv_processor.run(
            inputs=[],  # No input data needed for synthetic dataset generation
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("anomalib", "train"),
                        local_path="/opt/ml/processing/output/train",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
                ProcessingOutput(
                    output_name="validation",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("anomalib", "validation"),
                        local_path="/opt/ml/processing/output/validation",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
                ProcessingOutput(
                    output_name="test",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("anomalib", "test"),
                        local_path="/opt/ml/processing/output/test",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
            code="src/data/preprocess_cv.py",
            arguments=[
                "--num-samples", num_samples.to_string(),
                "--format", "anomalib",
                "--training-data-s3-uri", training_data_s3_uri,
                # Processing containers do not receive AWS_REGION automatically;
                # the real-data path needs it to reach S3.
                "--region", concrete_region,
            ],
        )

        step_cv_process_anomalib = ProcessingStep(
            name="PreprocessAnomalibData",
            step_args=cv_processor_anomalib_args,
            cache_config=cache_config,
        )

        fetch_processor = ScriptProcessor(
            image_uri=custom_training_image_uri,
            command=["python3"],
            instance_type=processing_instance_type,
            instance_count=1,
            base_job_name="cv-pipeline-fetch-model",
            sagemaker_session=pipeline_session,
            role=role_arn,
        )

        fetch_anomalib_args = fetch_processor.run(
            inputs=[],
            outputs=[
                ProcessingOutput(
                    output_name="pretrained_model",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("pretrained", "anomalib"),
                        local_path="/opt/ml/processing/output/pretrained_model",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
            code="src/model/fetch_latest_model.py",
            # Processing containers do not receive AWS_REGION automatically;
            # pass the pipeline's region explicitly rather than relying on
            # in-container resolution.
            arguments=[
                "--model-package-group-name", ANOMALY_MODEL_GROUP,
                "--region", concrete_region,
                # Warm-start must only consider this camera's own model
                # history; without the filter a camera could warm-start from
                # another camera's model.
                "--camera-id", camera_id,
            ],
        )
        step_fetch_anomalib = ProcessingStep(
            name="FetchLatestAnomalibModel",
            step_args=fetch_anomalib_args,
            cache_config=cache_config,
        )

        # Classifier warm-start mirror: same lookup against the classifier
        # package group, with its own execution-scoped output prefix. Both
        # branches warm-start from their own camera-filtered lineage.
        fetch_classifier_args = fetch_processor.run(
            inputs=[],
            outputs=[
                ProcessingOutput(
                    output_name="pretrained_model",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("pretrained", "classifier"),
                        local_path="/opt/ml/processing/output/pretrained_model",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
            code="src/model/fetch_latest_model.py",
            arguments=[
                "--model-package-group-name", CLASSIFIER_MODEL_GROUP,
                "--region", concrete_region,
                "--camera-id", camera_id,
            ],
        )
        step_fetch_classifier = ProcessingStep(
            name="FetchLatestClassifierModel",
            step_args=fetch_classifier_args,
            cache_config=cache_config,
        )

        anomalib_input_data = [
            InputData(
                channel_name="train",
                data_source=step_cv_process_anomalib.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/x-tar",
            ),
            InputData(
                channel_name="validation",
                data_source=step_cv_process_anomalib.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="application/x-tar",
            ),
            InputData(
                channel_name="pretrained_model",
                data_source=step_fetch_anomalib.properties.ProcessingOutputConfig.Outputs[
                    "pretrained_model"
                ].S3Output.S3Uri,
                content_type="application/x-tar",
            ),
        ]

        anomalib_trainer = ModelTrainer(
            training_image=custom_training_image_uri,
            compute=Compute(
                instance_type=training_instance_type.to_string(),
                instance_count=1,
            ),
            base_job_name="cv-pipeline-anomalib-train",
            sagemaker_session=pipeline_session,
            role=role_arn,
            hyperparameters={
                "program": "train_anomalib.py",
                "model_name": "autoencoder",
                "image_size": "256",
                "num_epochs": "3",
                "batch_size": "16",
                "learning_rate": "0.001",
            },
            input_data_config=anomalib_input_data,
            output_data_config=OutputDataConfig(
                s3_output_path=run_prefix("anomalib-training")
            ),
        ).with_metric_definitions(
            [
                MetricDefinition(
                    name="image_auroc",
                    regex=r"validation:image_auroc=([0-9]*\.?[0-9]+)",
                ),
            ]
        )

        anomalib_train_args = anomalib_trainer.train()
        step_train_anomalib = TrainingStep(
            name="TrainAnomalibModel",
            step_args=anomalib_train_args,
            cache_config=cache_config,
            depends_on=[step_cv_process_anomalib, step_fetch_anomalib],
        )

        anomalib_model_data = step_train_anomalib.properties.ModelArtifacts.S3ModelArtifacts
        eval_processor = ScriptProcessor(
            image_uri=custom_training_image_uri,
            command=["python3"],
            instance_type=processing_instance_type,
            instance_count=1,
            base_job_name="cv-pipeline-onnx-eval",
            sagemaker_session=pipeline_session,
            role=role_arn,
        )

        anomalib_eval_args = eval_processor.run(
            inputs=[
                ProcessingInput(
                    input_name="model",
                    s3_input=ProcessingS3Input(
                        s3_uri=anomalib_model_data,
                        local_path="/opt/ml/processing/input/model",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                ),
                ProcessingInput(
                    input_name="test",
                    s3_input=ProcessingS3Input(
                        s3_uri=step_cv_process_anomalib.properties.ProcessingOutputConfig.Outputs[
                            "test"
                        ].S3Output.S3Uri,
                        local_path="/opt/ml/processing/input/test",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("evaluation", "anomalib"),
                        local_path="/opt/ml/processing/output/evaluation",
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ],
            code="src/model/evaluate_anomalib_onnx.py",
            arguments=[
                "--image-size",
                "256",
            ],
        )
        step_eval_anomalib = ProcessingStep(
            name="EvaluateAnomalibOnnx",
            step_args=anomalib_eval_args,
            cache_config=cache_config,
            depends_on=[step_train_anomalib],
            property_files=[
                PropertyFile(
                    name="evaluation",
                    output_name="evaluation",
                    path="model_quality_statistics.json",
                ),
            ],
        )

        classifier_input_data = [
            InputData(
                channel_name="train",
                data_source=step_cv_process_anomalib.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/x-tar",
            ),
            InputData(
                channel_name="validation",
                data_source=step_cv_process_anomalib.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="application/x-tar",
            ),
            InputData(
                channel_name="pretrained_model",
                data_source=step_fetch_classifier.properties.ProcessingOutputConfig.Outputs[
                    "pretrained_model"
                ].S3Output.S3Uri,
                content_type="application/x-tar",
            ),
        ]

        classifier_trainer = ModelTrainer(
            training_image=custom_training_image_uri,
            compute=Compute(
                instance_type=training_instance_type.to_string(),
                instance_count=1,
            ),
            base_job_name="cv-pipeline-classifier-train",
            sagemaker_session=pipeline_session,
            role=role_arn,
            hyperparameters={
                "program": "train_classifier.py",
                "model_name": "resnet18_classifier",
                "image_size": "224",
                "num_epochs": "5",
                "batch_size": "16",
                "learning_rate": "0.001",
            },
            input_data_config=classifier_input_data,
            output_data_config=OutputDataConfig(
                s3_output_path=run_prefix("classifier-training")
            ),
        ).with_metric_definitions(
            [
                MetricDefinition(
                    name="accuracy",
                    regex=r"validation:accuracy=([0-9]*\.?[0-9]+)",
                ),
            ]
        )

        classifier_train_args = classifier_trainer.train()
        step_train_classifier = TrainingStep(
            name="TrainClassifierModel",
            step_args=classifier_train_args,
            cache_config=cache_config,
            depends_on=[step_cv_process_anomalib, step_fetch_classifier],
        )

        classifier_model_data = step_train_classifier.properties.ModelArtifacts.S3ModelArtifacts

        classifier_eval_args = eval_processor.run(
            inputs=[
                ProcessingInput(
                    input_name="model",
                    s3_input=ProcessingS3Input(
                        s3_uri=classifier_model_data,
                        local_path="/opt/ml/processing/input/model",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                ),
                ProcessingInput(
                    input_name="test",
                    s3_input=ProcessingS3Input(
                        s3_uri=step_cv_process_anomalib.properties.ProcessingOutputConfig.Outputs[
                            "test"
                        ].S3Output.S3Uri,
                        local_path="/opt/ml/processing/input/test",
                        s3_data_type="S3Prefix",
                        s3_input_mode="File",
                    ),
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    s3_output=ProcessingS3Output(
                        s3_uri=run_prefix("evaluation", "classifier"),
                        local_path="/opt/ml/processing/output/evaluation",
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ],
            code="src/model/evaluate_classifier_onnx.py",
            arguments=[
                "--image-size",
                "224",
            ],
        )
        step_eval_classifier = ProcessingStep(
            name="EvaluateClassifierOnnx",
            step_args=classifier_eval_args,
            cache_config=cache_config,
            depends_on=[step_train_classifier],
            property_files=[
                PropertyFile(
                    name="evaluation",
                    output_name="evaluation",
                    path="model_quality_statistics.json",
                ),
            ],
        )

        neo_target_device = ParameterString(name="NeoTargetDevice", default_value="none")
        neo_target_os = ParameterString(name="NeoTargetOs", default_value="LINUX")
        neo_target_arch = ParameterString(name="NeoTargetArch", default_value="X86_64")

        def make_neo_step(name: str, model_type: str, image_size: str, model_data):
            compile_args = eval_processor.run(
                inputs=[
                    ProcessingInput(
                        input_name="model",
                        s3_input=ProcessingS3Input(
                            s3_uri=model_data,
                            local_path="/opt/ml/processing/input/model",
                            s3_data_type="S3Prefix",
                            s3_input_mode="File",
                        ),
                    ),
                ],
                outputs=[
                    ProcessingOutput(
                        output_name="compiled",
                        s3_output=ProcessingS3Output(
                            s3_uri=run_prefix("neo", model_type, "result"),
                            local_path="/opt/ml/processing/output/compiled",
                            s3_upload_mode="EndOfJob",
                        ),
                    ),
                ],
                code="src/model/compile_neo_step.py",
                arguments=[
                    "--region", concrete_region,
                    "--role-arn", role_arn,
                    "--bucket", bucket_name,
                    "--execution-id", ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "--runs-prefix", RUNS_PREFIX,
                    "--model-type", model_type,
                    "--image-size", image_size,
                    "--camera-id", camera_id,
                    "--target-device", neo_target_device,
                    "--target-os", neo_target_os,
                    "--target-arch", neo_target_arch,
                ],
            )
            return ProcessingStep(
                name=name,
                step_args=compile_args,
                cache_config=cache_config,
            )

        step_compile_anomalib = make_neo_step(
            "CompileAnomalibNeo", "anomalib", "256", anomalib_model_data
        )
        step_compile_classifier = make_neo_step(
            "CompileClassifierNeo", "classifier", "224", classifier_model_data
        )

        # --- Register model steps ---
        anomalib_model_builder = ModelBuilder(
            s3_model_data_url=anomalib_model_data,
            image_uri=custom_training_image_uri,
            sagemaker_session=pipeline_session,
            role_arn=role_arn,
        )
        anomalib_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                content_type="application/json",
                s3_uri=Join(
                    on="/",
                    values=[
                        step_eval_anomalib.properties.ProcessingOutputConfig.Outputs[
                            "evaluation"
                        ].S3Output.S3Uri,
                        "model_quality_statistics.json",
                    ],
                ),
            )
        )
        # Registration depends on compilation (gate -> compile -> register):
        # only models that compile for the edge target become Approved
        # packages, so a registry entry always has a device-ready artifact.
        step_register_anomalib = ModelStep(
            name="RegisterAnomalibModel",
            depends_on=[step_compile_anomalib],
            step_args=anomalib_model_builder.register(
                model_package_group_name=ANOMALY_MODEL_GROUP,
                content_types=["application/x-image", "image/png", "image/jpeg"],
                response_types=["application/json"],
                # CPU only: docker/training/Dockerfile installs CPU-only PyTorch
                # wheels, so advertising a GPU instance here would invite readers
                # to pay for hardware this image cannot use.
                inference_instances=["ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"],
                approval_status="Approved",
                model_metrics=anomalib_metrics,
                description="Anomalib model package from pipeline training run",
                # Fleet lineage: lets warm-start lookup and operators map every
                # registry version back to the device that produced it.
                customer_metadata_properties={"CameraId": camera_id},
            ),
        )

        # --- Quality gates: condition → register (pass) or fail ---
        step_fail_anomalib = FailStep(
            name="FailAnomalibQualityGate",
            error_message=Join(
                on=" ",
                values=["Anomalib AUROC below threshold", min_auroc],
            ),
        )
        step_check_anomalib = ConditionStep(
            name="CheckAnomalibQuality",
            conditions=[
                ConditionGreaterThanOrEqualTo(
                    left=JsonGet(
                        step_name=step_eval_anomalib.name,
                        property_file="evaluation",
                        json_path="binary_classification_metrics.auc.value",
                    ),
                    right=min_auroc,
                ),
            ],
            if_steps=[step_compile_anomalib, step_register_anomalib],
            else_steps=[step_fail_anomalib],
            depends_on=[step_eval_anomalib],
        )

        # Classifier quality gate
        classifier_model_builder = ModelBuilder(
            s3_model_data_url=classifier_model_data,
            image_uri=custom_training_image_uri,
            sagemaker_session=pipeline_session,
            role_arn=role_arn,
        )
        classifier_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                content_type="application/json",
                s3_uri=Join(
                    on="/",
                    values=[
                        step_eval_classifier.properties.ProcessingOutputConfig.Outputs[
                            "evaluation"
                        ].S3Output.S3Uri,
                        "model_quality_statistics.json",
                    ],
                ),
            )
        )
        step_register_classifier = ModelStep(
            name="RegisterClassifierModel",
            depends_on=[step_compile_classifier],
            step_args=classifier_model_builder.register(
                model_package_group_name=CLASSIFIER_MODEL_GROUP,
                content_types=["application/x-image", "image/png", "image/jpeg"],
                response_types=["application/json"],
                # CPU only: docker/training/Dockerfile installs CPU-only PyTorch
                # wheels, so advertising a GPU instance here would invite readers
                # to pay for hardware this image cannot use.
                inference_instances=["ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"],
                approval_status="Approved",
                model_metrics=classifier_metrics,
                description="Classifier model package from pipeline training run",
                customer_metadata_properties={"CameraId": camera_id},
            ),
        )
        step_fail_classifier = FailStep(
            name="FailClassifierQualityGate",
            error_message=Join(
                on=" ",
                values=[
                    "Classifier below quality gate: balanced accuracy threshold",
                    min_accuracy,
                    "or defect recall threshold",
                    min_defect_recall,
                ],
            ),
        )
        # Both conditions must hold (ConditionStep ANDs its conditions).
        step_check_classifier = ConditionStep(
            name="CheckClassifierQuality",
            conditions=[
                ConditionGreaterThanOrEqualTo(
                    left=JsonGet(
                        step_name=step_eval_classifier.name,
                        property_file="evaluation",
                        json_path="balanced_accuracy",
                    ),
                    right=min_accuracy,
                ),
                ConditionGreaterThanOrEqualTo(
                    left=JsonGet(
                        step_name=step_eval_classifier.name,
                        property_file="evaluation",
                        json_path="defect_recall",
                    ),
                    right=min_defect_recall,
                ),
            ],
            if_steps=[step_compile_classifier, step_register_classifier],
            else_steps=[step_fail_classifier],
            depends_on=[step_eval_classifier],
        )

        # Create the pipeline
        pipeline = Pipeline(
            name=PIPELINE_NAME,
            parameters=[
                training_instance_type,
                processing_instance_type,
                num_samples,
                min_auroc,
                min_accuracy,
                min_defect_recall,
                training_data_s3_uri,
                camera_id,
                neo_target_device,
                neo_target_os,
                neo_target_arch,
            ],
            steps=[
                step_cv_process_anomalib,
                step_fetch_anomalib,
                step_fetch_classifier,
                step_train_anomalib,
                step_train_classifier,
                step_eval_anomalib,
                step_eval_classifier,
                step_check_anomalib,
                step_check_classifier,
            ],
            sagemaker_session=pipeline_session,
        )

        # Generate and return the pipeline definition JSON
        pipeline_definition_json = json.dumps(
            json.loads(pipeline.definition()), indent=2, sort_keys=True
        )

        return pipeline_definition_json
