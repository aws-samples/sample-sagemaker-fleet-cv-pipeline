# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Source-level contracts for the inference notebook.'
import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NOTEBOOK = REPO_ROOT / "notebooks" / "inference_demo.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text())


def test_notebook_is_valid_under_its_declared_schema(notebook):
    'Validate the whole document, not just its source strings.'
    pytest.importorskip("nbformat")
    from nbformat.validator import iter_validate

    errors = list(iter_validate(notebook))
    assert not errors, "; ".join(e.message for e in errors)[:1000]


def test_every_code_cell_parses(notebook):
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.strip().startswith("!"):
            continue  # shell escape, not Python
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"cell {i} does not parse: {exc}")


_EXEC_ARN = "arn:aws:sagemaker:us-east-2:111111111111:pipeline/p/execution/AAA"
_GROUP = "fleet-cv-anomaly"
_ARN_V7 = f"arn:aws:sagemaker:us-east-2:111111111111:model-package/{_GROUP}/7"
_ARN_V8 = f"arn:aws:sagemaker:us-east-2:111111111111:model-package/{_GROUP}/8"


def _selection_source() -> str:
    """The selection helpers, lifted verbatim from the notebook cell."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if "def registered_by_this_execution():" in source:
            start = source.index("def registered_by_this_execution():")
            end = source.index("def fetch_metrics(desc):")
            return source[start:end]
    raise AssertionError("the selection helpers are not in the notebook")


class _FakeRegistry:
    def __init__(self, descriptions):
        self.descriptions = descriptions

    def describe_model_package(self, ModelPackageName):
        return self.descriptions[ModelPackageName]


def _select(steps, descriptions, camera="default"):
    namespace = {
        "steps": steps,
        "EXECUTION_ARN": _EXEC_ARN,
        "CAMERA_ID": camera,
        "sm": _FakeRegistry(descriptions),
    }
    exec(_selection_source(), namespace)  # noqa: S102 - notebook code under test
    return namespace["get_model_from_this_execution"]


def _approved(arn, camera="default", **overrides):
    description = {
        "ModelPackageArn": arn,
        "ModelApprovalStatus": "Approved",
        "ModelPackageStatus": "Completed",
        "CustomerMetadataProperties": {"CameraId": camera},
    }
    description.update(overrides)
    return {arn: description}


def _registration_step(arn):
    return {"StepName": "RegisterModel", "Metadata": {"RegisterModel": {"Arn": arn}}}


def test_it_selects_this_executions_model_not_the_newest():
    'The defect, stated as a test: a NEWER package must not win.'
    select = _select(
        [_registration_step(_ARN_V7)],
        {**_approved(_ARN_V7), **_approved(_ARN_V8)},
    )
    assert select(_GROUP)["ModelPackageArn"] == _ARN_V7


def test_a_rejected_package_is_refused():
    select = _select(
        [_registration_step(_ARN_V7)],
        _approved(_ARN_V7, ModelApprovalStatus="Rejected"),
    )
    with pytest.raises(RuntimeError, match="not Approved"):
        select(_GROUP)


def test_another_cameras_package_is_refused():
    select = _select([_registration_step(_ARN_V7)], _approved(_ARN_V7), camera="cam-999")
    with pytest.raises(RuntimeError, match="CameraId"):
        select(_GROUP)




def _start_cell_source() -> str:
    """The dataset-selection and start cell, verbatim from the notebook."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if "sm.start_pipeline_execution(" in source and "TRAINING_DATA_S3_URI" in source:
            return source
    raise AssertionError("the start cell is not in the notebook")


class _RecordingSageMaker:
    def __init__(self):
        self.started = []

    def start_pipeline_execution(self, **kwargs):
        self.started.append(kwargs)
        return {"PipelineExecutionArn": "arn:aws:sagemaker:us-east-2:1:pipeline/p/execution/e"}

    def describe_pipeline_execution(self, **kwargs):
        return {"PipelineExecutionStatus": "Succeeded",
                "PipelineExecutionArn": kwargs.get("PipelineExecutionArn", "")}

    def list_pipeline_execution_steps(self, **kwargs):
        return {"PipelineExecutionSteps": []}


def _run_start_cell(uri, bucket, sm):
    "Execute the notebook's real start cell against a real publication index on disk."
    import datetime
    import json as _json
    import os
    import tempfile
    import time as _time
    from pathlib import Path as _P

    source = _start_cell_source()
    index_name = "publication_index.json"
    with tempfile.TemporaryDirectory() as tmp:
        root = _P(tmp)
        (root / "notebooks").mkdir()
        published = root / "data" / "bccd"
        published.mkdir(parents=True)
        (published / index_name).write_text(_json.dumps({
            "schema_version": 1,
            "cameras": {"default": {"s3_uri": uri, "dataset_id": "abc123",
                                    "local_path": str(published)}},
        }))
        ns = {
            "BUCKET": bucket,
            "PUBLICATION_INDEX_NAME": index_name,
            "PIPELINE_NAME": "fleet-cv-pipeline",
            "sm": sm,
            "Path": _P,
            "json": _json,
            "time": _time,
            "datetime": datetime.datetime,
            "timezone": datetime.timezone,
            "print": lambda *a, **k: None,
        }
        cwd = os.getcwd()
        try:
            os.chdir(root / "notebooks")      # the cell resolves ../data/bccd
            exec(compile(source, "<start-cell>", "exec"), ns)   # noqa: S102
        finally:
            os.chdir(cwd)
    return ns


def test_the_notebook_refuses_a_publication_from_another_deployment():
    'EXECUTES the real start cell. A foreign bucket must raise BEFORE any billed start.'
    import pytest
    sm = _RecordingSageMaker()
    with pytest.raises(RuntimeError) as exc:
        _run_start_cell("s3://pipeline-east/cv-pipeline/datasets/abc123",
                        "pipeline-west", sm)

    assert "pipeline-west" in str(exc.value), (
        f"the refusal does not name the deployment's own bucket: {exc.value}"
    )
    assert sm.started == [], (
        f"a pipeline execution was STARTED against a foreign bucket: {sm.started}"
    )


def test_the_notebook_starts_normally_when_the_publication_matches():
    """The guard must not break the ordinary path, and must preserve the URI exactly."""
    sm = _RecordingSageMaker()
    uri = "s3://pipeline-west/cv-pipeline/datasets/abc123"
    _run_start_cell(uri, "pipeline-west", sm)

    assert len(sm.started) == 1, f"expected exactly one start, got {len(sm.started)}"
    params = {p["Name"]: p["Value"] for p in sm.started[0]["PipelineParameters"]}
    assert params["TrainingDataS3Uri"] == uri, (
        f"the guard altered the dataset URI: {params['TrainingDataS3Uri']!r} != {uri!r}"
    )
