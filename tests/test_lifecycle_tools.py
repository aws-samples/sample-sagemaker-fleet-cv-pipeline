# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The reader's run and cleanup commands must verify outcomes and stop on live work."""
import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from config import ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cleanup = load_script("cleanup")
runner = load_script("run_pipeline")


@pytest.fixture
def cleanup_setup(monkeypatch):
    sm = Mock()
    packages = {ANOMALY_MODEL_GROUP: ["anomaly/1"], CLASSIFIER_MODEL_GROUP: ["classifier/1"]}
    sm.delete_model_package.side_effect = lambda ModelPackageName: [
        values.remove(ModelPackageName) for values in packages.values() if ModelPackageName in values
    ]
    monkeypatch.setattr(cleanup.boto3, "client", lambda *a, **k: sm)
    monkeypatch.setattr(cleanup, "resolve_region", lambda _: "us-east-2")
    monkeypatch.setattr(cleanup, "stack_description", lambda *a: {"StackStatus": "CREATE_COMPLETE"})
    monkeypatch.setattr(cleanup, "resolve_deployment", lambda _: {})
    monkeypatch.setattr(cleanup, "report_inflight", lambda *a: 0)
    monkeypatch.setattr(cleanup, "active_compilation_jobs", lambda *a: [])
    monkeypatch.setattr(cleanup, "list_packages", lambda _, group: list(packages[group]))
    monkeypatch.setattr(sys, "argv", ["cleanup.py"])
    return sm


def test_core_cleanup_deletes_both_groups_and_prints_region(cleanup_setup, capsys):
    assert cleanup.main() == 0
    assert cleanup_setup.delete_model_package.call_count == 2
    output = capsys.readouterr().out
    assert "AWS_DEFAULT_REGION=us-east-2" in output
    assert "-c fleet=true" in output


@pytest.mark.parametrize("active", ["pipeline", "compilation"])
def test_cleanup_refuses_active_work(cleanup_setup, monkeypatch, active):
    if active == "pipeline":
        monkeypatch.setattr(cleanup, "report_inflight", lambda *a: 1)
    else:
        monkeypatch.setattr(cleanup, "active_compilation_jobs", lambda *a: ["neo-running"])
    assert cleanup.main() == 1
    cleanup_setup.delete_model_package.assert_not_called()


def test_fleet_cleanup_closes_before_inspecting_ownership(cleanup_setup, monkeypatch):
    events = []
    monkeypatch.setattr(cleanup, "resolve_deployment", lambda _: {cleanup.OUTPUT_LEASE_TABLE_NAME: "leases"})
    monkeypatch.setattr(cleanup, "close_admission", lambda *a: events.append("close"))
    monkeypatch.setattr(cleanup, "disable_queue_consumption", lambda *a: events.append("disable"))
    monkeypatch.setattr(cleanup, "retained_leases", lambda *a: events.append("scan") or ["owner"])
    assert cleanup.main() == 1
    assert events == ["close", "disable", "scan"]
    cleanup_setup.delete_model_package.assert_not_called()


def test_dry_run_does_not_change_admission_or_packages(cleanup_setup, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cleanup.py", "--dry-run"])
    close = Mock()
    monkeypatch.setattr(cleanup, "close_admission", close)
    assert cleanup.main() == 0
    close.assert_not_called()
    cleanup_setup.delete_model_package.assert_not_called()


def test_missing_core_stack_does_not_authorize_cleanup(cleanup_setup, monkeypatch):
    monkeypatch.setattr(cleanup, "stack_description", lambda *a: None)
    assert cleanup.main() == 2
    cleanup_setup.delete_model_package.assert_not_called()


def test_access_denied_is_not_reported_as_an_absent_stack(monkeypatch):
    cfn = Mock()
    cfn.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "DescribeStacks")
    monkeypatch.setattr(cleanup.boto3, "client", lambda *a, **k: cfn)
    with pytest.raises(ClientError):
        cleanup.stack_description("us-east-2", "sample")


def test_package_listing_includes_all_pages():
    sm = Mock()
    sm.get_paginator.return_value.paginate.return_value = [
        {"ModelPackageSummaryList": [{"ModelPackageArn": "v1"}]},
        {"ModelPackageSummaryList": [{"ModelPackageArn": "v2"}]},
    ]
    assert cleanup.list_packages(sm, "group") == ["v1", "v2"]


@pytest.fixture
def completed_execution():
    sm, s3 = Mock(), Mock()
    steps, results, packages = [], {}, {}
    for model_type, group, label in [
        ("anomalib", ANOMALY_MODEL_GROUP, "Anomalib"),
        ("classifier", CLASSIFIER_MODEL_GROUP, "Classifier"),
    ]:
        steps.extend([
            {"StepName": f"Compile{label}Neo", "StepStatus": "Succeeded",
             "Metadata": {"ProcessingJob": {"Arn": f"arn/job/{model_type}"}}},
            {"StepName": f"Register{label}Model", "StepStatus": "Succeeded",
             "Metadata": {"RegisterModel": {"Arn": f"arn/package/{group}/1"}}},
        ])
        results[model_type] = {
            "compilation_job_name": model_type,
            "compiled_artifact_s3_uri": f"s3://artifacts/{model_type}.tar.gz",
            "model_type": model_type, "output_config": {"TargetPlatform": {"Os": "LINUX", "Arch": "X86_64"}},
        }
        packages[f"arn/package/{group}/1"] = {
            "ModelApprovalStatus": "Approved", "ModelPackageStatus": "Completed",
            "ModelPackageGroupName": group, "ModelPackageVersion": 1,
        }
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Succeeded"}
    sm.get_paginator.return_value.paginate.return_value = [{"PipelineExecutionSteps": steps}]
    sm.describe_model_package.side_effect = lambda ModelPackageName: packages[ModelPackageName]
    sm.describe_processing_job.side_effect = lambda ProcessingJobName: {
        "ProcessingOutputConfig": {"Outputs": [{"S3Output": {"S3Uri": f"s3://artifacts/{ProcessingJobName}"}}]}}
    sm.describe_compilation_job.side_effect = lambda CompilationJobName: {
        "CompilationJobName": CompilationJobName, "CompilationJobStatus": "COMPLETED",
        "ModelArtifacts": {"S3ModelArtifacts": results[CompilationJobName]["compiled_artifact_s3_uri"]}}
    s3.get_object.side_effect = lambda Bucket, Key: {
        "Body": io.BytesIO(json.dumps(results[Key.split("/")[0]]).encode())}
    s3.head_object.return_value = {"ContentLength": 1024}
    return sm, s3, steps, packages


def test_verification_records_this_executions_packages_and_compilations(completed_execution):
    sm, s3, _, _ = completed_execution
    result = runner.verify_execution(sm, s3, "execution")
    assert {p["group"] for p in result["packages"]} == {ANOMALY_MODEL_GROUP, CLASSIFIER_MODEL_GROUP}
    assert len(result["compilations"]) == 2
    assert result["execution_arn"] == "execution"


@pytest.mark.parametrize("defect", ["missing-registration", "unapproved", "missing-artifact", "failed-step"])
def test_verification_refuses_incomplete_or_ineligible_results(completed_execution, defect):
    sm, s3, steps, packages = completed_execution
    if defect == "missing-registration":
        steps.pop()
    elif defect == "unapproved":
        next(iter(packages.values()))["ModelApprovalStatus"] = "Rejected"
    elif defect == "missing-artifact":
        s3.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    else:
        steps[0]["StepStatus"] = "Failed"
    with pytest.raises((RuntimeError, ClientError)):
        runner.verify_execution(sm, s3, "execution")
