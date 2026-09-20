# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Regression tests for the in-pipeline Neo compilation step.'
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "model"))

from config import (  # noqa: E402
    ANOMALY_MODEL_GROUP,
    CLASSIFIER_MODEL_GROUP,
    PIPELINE_STACK_NAME,
)

import compile_neo_step  # noqa: E402


class TestDeterministicJobName:
    def test_same_inputs_same_name(self):
        a = compile_neo_step.deterministic_job_name("39nf6hplr201", "anomalib")
        b = compile_neo_step.deterministic_job_name("39nf6hplr201", "anomalib")
        assert a == b

    def test_no_timestamp_or_random_component(self):
        name = compile_neo_step.deterministic_job_name("exec123", "classifier")
        assert name == "neo-classifier-exec123"

    def test_sanitizes_to_api_charset_and_length(self):
        name = compile_neo_step.deterministic_job_name("a" * 100 + ":/x_", "anomalib")
        assert len(name) <= 63
        assert name == name.rstrip("-")
        for ch in name:
            assert ch.isalnum() or ch == "-"


class TestOutputConfigTargets:
    def test_platform_used_when_device_is_none(self):
        config = compile_neo_step.build_output_config(
            "bucket", "prefix/", "none", "LINUX", "X86_64"
        )
        assert config["TargetPlatform"] == {"Os": "LINUX", "Arch": "X86_64"}
        assert "TargetDevice" not in config

    def test_device_wins_when_set(self):
        config = compile_neo_step.build_output_config(
            "bucket", "prefix/", "jetson_xavier", "LINUX", "X86_64"
        )
        assert config["TargetDevice"] == "jetson_xavier"
        assert "TargetPlatform" not in config

    def test_never_both(self):
        for device in ("none", "rasp4b"):
            config = compile_neo_step.build_output_config(
                "bucket", "prefix/", device, "LINUX", "X86_64"
            )
            assert ("TargetDevice" in config) != ("TargetPlatform" in config)


class TestSubmitOrResume:
    def _client_error(self, code, message=""):
        from botocore.exceptions import ClientError

        return ClientError(
            {"Error": {"Code": code, "Message": message}}, "CreateCompilationJob"
        )

    def test_creates_when_absent(self):
        sm = mock.Mock()
        compile_neo_step.submit_or_resume(sm, "job", {"RoleArn": "arn"})
        sm.create_compilation_job.assert_called_once_with(
            CompilationJobName="job", RoleArn="arn"
        )

    def test_resumes_when_job_already_exists(self):
        sm = mock.Mock()
        sm.create_compilation_job.side_effect = self._client_error(
            "ResourceInUse", "Job already exists"
        )
        compile_neo_step.submit_or_resume(sm, "job", {"RoleArn": "arn"})

    def test_reraises_other_errors(self):
        sm = mock.Mock()
        sm.create_compilation_job.side_effect = self._client_error("ValidationException")
        with pytest.raises(Exception):
            compile_neo_step.submit_or_resume(sm, "job", {"RoleArn": "arn"})


def _aws_identity():
    'Return (account, region) from live credentials, or None if unavailable.'
    try:
        from aws_env import AwsEnvError, resolve_env
    except ImportError:
        return None

    from botocore.exceptions import (  # noqa: PLC0415
        NoCredentialsError,
        NoRegionError,
        PartialCredentialsError,
        ProfileNotFound,
    )

    absent_causes = (NoRegionError, NoCredentialsError, PartialCredentialsError,
                     ProfileNotFound)
    try:
        return resolve_env()
    except AwsEnvError as exc:
        cause = exc.__cause__ or exc.__context__
        if isinstance(cause, absent_causes):
            return None
        if cause is None and "could not be resolved" in str(exc).lower():
            return None
        raise


@pytest.fixture(scope="module")
def pipeline_template():
    'The synthesized template in BOOTSTRAP mode.'
    aws_cdk = pytest.importorskip("aws_cdk")
    from sagemaker_cv_pipeline_stack import SageMakerCVPipelineStack

    app = aws_cdk.App(context={"bootstrap": "true"})
    SageMakerCVPipelineStack(
        app,
        PIPELINE_STACK_NAME,
        env=aws_cdk.Environment(account="111111111111", region="us-east-1"),
    )
    return app.synth().get_stack_by_name(PIPELINE_STACK_NAME).template


def test_bootstrap_creates_deletable_prerequisites_without_a_pipeline(pipeline_template):
    resources = list(pipeline_template["Resources"].values())
    types = [resource["Type"] for resource in resources]
    assert "AWS::SageMaker::Pipeline" not in types
    assert types.count("AWS::SageMaker::ModelPackageGroup") == 2
    for resource in resources:
        if resource["Type"] in ("AWS::S3::Bucket", "AWS::ECR::Repository"):
            assert resource["DeletionPolicy"] == "Delete"
        if resource["Type"] == "AWS::ECR::Repository":
            assert resource["Properties"]["EmptyOnDelete"] is True


@pytest.mark.parametrize("fleet,expected_count", [("false", 1), ("true", 2)])
def test_fleet_stack_is_optional(tmp_path, fleet, expected_count):
    from app import build_app

    app = build_app("111111111111", "us-east-1",
                    context={"bootstrap": "true", "fleet": fleet}, outdir=str(tmp_path))
    assert len(app.synth().stacks) == expected_count


@pytest.fixture(scope="module")
def pipeline_steps():
    identity = _aws_identity()
    if identity is None:
        pytest.skip(
            "pipeline definition generation requires AWS credentials and a "
            "resolvable region (SDK v3 validates the execution role against "
            "live IAM and uploads step code to S3)"
        )
    account, region = identity
    aws_cdk = pytest.importorskip("aws_cdk")
    from sagemaker_cv_pipeline_stack import SageMakerCVPipelineStack

    app = aws_cdk.App()
    stack = SageMakerCVPipelineStack(
        app,
        PIPELINE_STACK_NAME,
        env=aws_cdk.Environment(account=account, region=region),
    )
    definition = json.loads(
        stack.pipeline.pipeline_definition["PipelineDefinitionBody"]
    )
    return {step["Name"]: step for step in definition["Steps"]}, definition


@pytest.mark.integration
def test_compile_steps_nested_under_passing_gates(pipeline_steps):
    steps, _ = pipeline_steps
    for gate, compile_name in (
        ("CheckAnomalibQuality", "CompileAnomalibNeo"),
        ("CheckClassifierQuality", "CompileClassifierNeo"),
    ):
        if_names = [s["Name"] for s in steps[gate]["Arguments"]["IfSteps"]]
        assert compile_name in if_names


@pytest.mark.integration
def test_registration_depends_on_compilation(pipeline_steps):
    steps, _ = pipeline_steps
    for gate, compile_name, register_prefix in (
        ("CheckAnomalibQuality", "CompileAnomalibNeo", "RegisterAnomalibModel"),
        ("CheckClassifierQuality", "CompileClassifierNeo", "RegisterClassifierModel"),
    ):
        if_steps = {s["Name"]: s for s in steps[gate]["Arguments"]["IfSteps"]}
        register_steps = [n for n in if_steps if n.startswith(register_prefix)]
        assert register_steps, f"no register step under {gate}"
        depends = if_steps[register_steps[0]].get("DependsOn", [])
        assert compile_name in depends


@pytest.mark.integration
def test_neo_parameters_present(pipeline_steps):
    _, definition = pipeline_steps
    names = {p["Name"] for p in definition["Parameters"]}
    assert {"NeoTargetDevice", "NeoTargetOs", "NeoTargetArch"} <= names


@pytest.mark.integration
def test_classifier_warm_start_wiring(pipeline_steps):
    steps, _ = pipeline_steps
    assert "FetchLatestClassifierModel" in steps
    fetch_args = " ".join(
        str(a)
        for a in steps["FetchLatestClassifierModel"]["Arguments"]["AppSpecification"][
            "ContainerArguments"
        ]
    )
    assert CLASSIFIER_MODEL_GROUP in fetch_args
    assert "--camera-id" in fetch_args

    train = steps["TrainClassifierModel"]
    channels = {
        c["ChannelName"] for c in train["Arguments"]["InputDataConfig"]
    }
    assert "pretrained_model" in channels
    assert "FetchLatestClassifierModel" in train.get("DependsOn", [])


@pytest.mark.integration
def test_both_fetch_steps_filter_by_camera(pipeline_steps):
    steps, _ = pipeline_steps
    for name, group in (
        ("FetchLatestAnomalibModel", ANOMALY_MODEL_GROUP),
        ("FetchLatestClassifierModel", CLASSIFIER_MODEL_GROUP),
    ):
        args = steps[name]["Arguments"]["AppSpecification"]["ContainerArguments"]
        joined = " ".join(str(a) for a in args)
        assert group in joined
        assert "--camera-id" in joined


@pytest.mark.integration
def test_compile_step_shapes_and_region(pipeline_steps):
    steps, _ = pipeline_steps
    for gate, name, size in (
        ("CheckAnomalibQuality", "CompileAnomalibNeo", "256"),
        ("CheckClassifierQuality", "CompileClassifierNeo", "224"),
    ):
        if_steps = {s["Name"]: s for s in steps[gate]["Arguments"]["IfSteps"]}
        args = if_steps[name]["Arguments"]["AppSpecification"]["ContainerArguments"]
        joined = [str(a) for a in args]
        assert "--image-size" in joined
        assert size == joined[joined.index("--image-size") + 1]
        assert "--target-device" in joined


PULL_ACTIONS = frozenset({
    "ecr:BatchCheckLayerAvailability",
    "ecr:BatchGetImage",
    "ecr:GetDownloadUrlForLayer",
})
METRIC_NAMESPACES = frozenset({
    "/aws/sagemaker/TrainingJobs",
    "/aws/sagemaker/ProcessingJobs",
})


IAM_ASSERTIONS = (
    "test_the_execution_role_uses_only_policy_forms_these_tests_can_evaluate",
    "test_ecr_pull_is_scoped_to_the_training_repository",
    "test_no_statement_grants_a_wildcard_action_over_the_protected_actions",
    "test_put_metric_data_is_confined_by_namespace",
    "test_the_namespace_condition_survives_a_request_without_context",
)
