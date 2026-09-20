# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Retrying must be able to make progress.'
import importlib.util
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestration"))

from model.compile_neo_step import (  # noqa: E402
    deterministic_job_name,
    resolve_attempt,
)


class FakeCompilation:
    """Only the two calls resolve_attempt makes."""

    def __init__(self, statuses: dict):
        self.statuses = statuses
        self.described = []

    def describe_compilation_job(self, CompilationJobName):
        self.described.append(CompilationJobName)
        if CompilationJobName not in self.statuses:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFound", "Message": "absent"}},
                "DescribeCompilationJob",
            )
        return {"CompilationJobStatus": self.statuses[CompilationJobName]}


def test_a_fresh_execution_uses_the_first_attempt():
    sm = FakeCompilation({})
    assert resolve_attempt(sm, "exec-1", "classifier") == deterministic_job_name(
        "exec-1", "classifier", 1
    )


@pytest.mark.parametrize("status", ["INPROGRESS", "STARTING", "COMPLETED"])
def test_an_existing_nonterminal_or_succeeded_job_is_resumed(status):
    """An ambiguous create retry must not start a second compilation."""
    first = deterministic_job_name("exec-1", "classifier", 1)
    sm = FakeCompilation({first: status})
    assert resolve_attempt(sm, "exec-1", "classifier") == first


def test_a_failed_job_advances_to_a_new_attempt():
    """The defect: this returned the FAILED job and the step failed forever."""
    first = deterministic_job_name("exec-1", "classifier", 1)
    sm = FakeCompilation({first: "FAILED"})
    resolved = resolve_attempt(sm, "exec-1", "classifier")
    assert resolved != first
    assert resolved == deterministic_job_name("exec-1", "classifier", 2)


def test_a_stopped_job_also_advances():
    first = deterministic_job_name("exec-1", "classifier", 1)
    sm = FakeCompilation({first: "STOPPED"})
    assert resolve_attempt(sm, "exec-1", "classifier") == deterministic_job_name(
        "exec-1", "classifier", 2
    )


def test_it_walks_past_several_failures_to_a_running_one():
    names = [deterministic_job_name("exec-1", "anomalib", n) for n in (1, 2, 3)]
    sm = FakeCompilation({names[0]: "FAILED", names[1]: "FAILED",
                          names[2]: "INPROGRESS"})
    assert resolve_attempt(sm, "exec-1", "anomalib") == names[2]


def test_exhausting_every_attempt_fails_loudly():
    """Retrying forever hides a real export or target problem."""
    statuses = {
        deterministic_job_name("exec-1", "classifier", n): "FAILED"
        for n in range(1, 11)
    }
    sm = FakeCompilation(statuses)
    with pytest.raises(RuntimeError, match="have failed"):
        resolve_attempt(sm, "exec-1", "classifier")


def test_job_names_stay_within_the_api_constraint():
    """[a-zA-Z0-9-]{1,63}, with the suffix applied after truncation."""
    import re

    for attempt in (1, 2, 10, 99):
        name = deterministic_job_name("e" * 200, "classifier", attempt)
        assert len(name) <= 63, name
        assert re.match(r"^[a-zA-Z0-9-]{1,63}$", name), name
        if attempt > 1:
            assert name.endswith(f"-a{attempt}"), (
                "the attempt suffix must survive truncation, or two attempts "
                "collide on one name"
            )


def test_distinct_attempts_never_collide():
    names = {deterministic_job_name("e" * 200, "classifier", n) for n in range(1, 11)}
    assert len(names) == 10


@pytest.fixture(scope="module")
def inference_demo():
    spec = importlib.util.spec_from_file_location(
        "inference_demo_recovery", REPO_ROOT / "scripts/inference_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["inference_demo_recovery"] = module
    spec.loader.exec_module(module)
    return module


def _package(arn, approval="Approved", status="Completed", camera="cam-001"):
    return {
        "ModelPackageArn": arn,
        "ModelApprovalStatus": approval,
        "ModelPackageStatus": status,
        "CustomerMetadataProperties": {"CameraId": camera},
        "InferenceSpecification": {"Containers": [{"ModelDataUrl": f"s3://b/{arn}"}]},
    }


class FakeRegistry:
    """Enough of the SageMaker client for resolve_model_artifact's search."""

    def __init__(self, pages):
        self.pages = pages
        self.by_arn = {p["ModelPackageArn"]: p for page in pages for p in page}

    def get_paginator(self, _name):
        registry = self

        class _Paginator:
            def paginate(self, **kwargs):
                for page in registry.pages:
                    yield {"ModelPackageSummaryList": [
                        {"ModelPackageArn": p["ModelPackageArn"]} for p in page
                    ]}

        return _Paginator()

    def describe_model_package(self, ModelPackageName):
        return self.by_arn[ModelPackageName]


def test_an_ineligible_newest_package_is_skipped(inference_demo, monkeypatch):
    """The newest match was still creating; an older Approved one existed."""
    registry = FakeRegistry([[
        _package("arn:new", status="InProgress"),
        _package("arn:old"),
    ]])
    monkeypatch.setattr(inference_demo.boto3, "client", lambda *a, **k: registry)
    assert inference_demo.get_model_artifact(
        model_group="g", region="us-east-2", camera_id="cam-001"
    ) == "s3://b/arn:old"


def test_selection_continues_across_pages(inference_demo, monkeypatch):
    registry = FakeRegistry([
        [_package("arn:new", approval="Rejected")],
        [_package("arn:older")],
    ])
    monkeypatch.setattr(inference_demo.boto3, "client", lambda *a, **k: registry)
    assert inference_demo.get_model_artifact(
        model_group="g", region="us-east-2", camera_id="cam-001"
    ) == "s3://b/arn:older"


def test_a_different_camera_is_never_selected(inference_demo, monkeypatch):
    registry = FakeRegistry([[_package("arn:other", camera="cam-002")]])
    monkeypatch.setattr(inference_demo.boto3, "client", lambda *a, **k: registry)
    with pytest.raises(RuntimeError):
        inference_demo.get_model_artifact(
            model_group="g", region="us-east-2", camera_id="cam-001"
        )


def test_all_candidates_ineligible_reports_why(inference_demo, monkeypatch):
    """A bare "not found" would hide that packages existed but were unusable."""
    registry = FakeRegistry([[
        _package("arn:a", status="InProgress"),
        _package("arn:b", approval="Rejected"),
    ]])
    monkeypatch.setattr(inference_demo.boto3, "client", lambda *a, **k: registry)
    with pytest.raises(RuntimeError, match="ineligible"):
        inference_demo.get_model_artifact(
            model_group="g", region="us-east-2", camera_id="cam-001"
        )
