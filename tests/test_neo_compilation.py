# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Regression tests for the in-pipeline Neo compilation step.

Two layers:

1. Pure unit tests on src/model/compile_neo_step.py helpers -- deterministic
   job naming, TargetDevice/TargetPlatform mutual exclusion, and resume-on-
   retry handling. These run everywhere.

2. Structure assertions on the generated pipeline definition -- the compile
   steps sit under the passing gate branches, registration depends on
   compilation, and the Neo parameters exist. Generating the definition
   makes the SageMaker SDK v3 validate the execution role against live IAM,
   so these tests require AWS credentials and the deployed bootstrap role;
   they skip cleanly otherwise (same constraint as deploying the stack).
"""
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "model"))

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


def _aws_reachable():
    try:
        import boto3

        boto3.client("sts").get_caller_identity()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def pipeline_steps():
    if not _aws_reachable():
        pytest.skip("pipeline definition generation requires AWS credentials "
                    "(SDK v3 validates the execution role against live IAM)")
    aws_cdk = pytest.importorskip("aws_cdk")
    from sagemaker_cv_pipeline_stack import SageMakerCVPipelineStack

    app = aws_cdk.App()
    try:
        stack = SageMakerCVPipelineStack(app, "SageMakerCVPipelineStack")
    except Exception as exc:  # noqa: BLE001 -- bootstrap role absent
        pytest.skip(f"pipeline definition unavailable in this account: {exc}")
    definition = json.loads(
        stack.pipeline.pipeline_definition["PipelineDefinitionBody"]
    )
    return {step["Name"]: step for step in definition["Steps"]}, definition


def test_compile_steps_nested_under_passing_gates(pipeline_steps):
    steps, _ = pipeline_steps
    for gate, compile_name in (
        ("CheckAnomalibQuality", "CompileAnomalibNeo"),
        ("CheckClassifierQuality", "CompileClassifierNeo"),
    ):
        if_names = [s["Name"] for s in steps[gate]["Arguments"]["IfSteps"]]
        assert compile_name in if_names


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


def test_neo_parameters_present(pipeline_steps):
    _, definition = pipeline_steps
    names = {p["Name"] for p in definition["Parameters"]}
    assert {"NeoTargetDevice", "NeoTargetOs", "NeoTargetArch"} <= names


def test_classifier_warm_start_wiring(pipeline_steps):
    steps, _ = pipeline_steps
    assert "FetchLatestClassifierModel" in steps
    fetch_args = " ".join(
        str(a)
        for a in steps["FetchLatestClassifierModel"]["Arguments"]["AppSpecification"][
            "ContainerArguments"
        ]
    )
    assert "sm1k-classifier-model-group" in fetch_args
    assert "--camera-id" in fetch_args

    train = steps["TrainClassifierModel"]
    channels = {
        c["ChannelName"] for c in train["Arguments"]["InputDataConfig"]
    }
    assert "pretrained_model" in channels
    assert "FetchLatestClassifierModel" in train.get("DependsOn", [])


def test_both_fetch_steps_filter_by_camera(pipeline_steps):
    steps, _ = pipeline_steps
    for name, group in (
        ("FetchLatestAnomalibModel", "sm1k-anomaly-model-group"),
        ("FetchLatestClassifierModel", "sm1k-classifier-model-group"),
    ):
        args = steps[name]["Arguments"]["AppSpecification"]["ContainerArguments"]
        joined = " ".join(str(a) for a in args)
        assert group in joined
        assert "--camera-id" in joined


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
