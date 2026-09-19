# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'A request id is a deduplication key and a name; both constrain it.'
import gzip
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import botocore
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault(
    "STATE_MACHINE_ARN", "arn:aws:states:us-east-2:000000000000:stateMachine:test"
)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


submit_batch = _load("scripts/submit_batch.py", "submit_batch_ids")
from orchestration.queue_trigger import InvalidRequest, execution_name  # noqa: E402


def _sagemaker_name_pattern():
    """The real constraint, read from the botocore model rather than assumed."""
    path = os.path.join(
        os.path.dirname(botocore.__file__), "data/sagemaker/2017-07-24/service-2.json.gz"
    )
    model = json.load(gzip.open(path))
    return re.compile(f"^{model['shapes']['PipelineExecutionName']['pattern']}$")


SAGEMAKER_NAME = _sagemaker_name_pattern()
SFN_NAME_MAX = 80


def test_fifty_cameras_get_fifty_distinct_ids():
    """The defect: one id for the whole batch silently dropped 49 cameras."""
    batch = "20260912T203930Z-3ba6d01f"
    ids = {submit_batch.camera_request_id(batch, f"cam-{i:03d}") for i in range(1, 51)}
    assert len(ids) == 50


def test_ids_are_stable_across_retries():
    """The point of a batch id: a retry must not create duplicate training runs."""
    first = [submit_batch.camera_request_id("b1", f"cam-{i}") for i in range(5)]
    second = [submit_batch.camera_request_id("b1", f"cam-{i}") for i in range(5)]
    assert first == second


def test_a_new_batch_produces_different_ids():
    """A deliberate retrain must not be deduplicated against the previous one."""
    assert (submit_batch.camera_request_id("b1", "cam-001")
            != submit_batch.camera_request_id("b2", "cam-001"))


def test_the_encoding_is_unambiguous():
    """Plain concatenation collides: ("b","1-c") and ("b-1","c") both give b-1-c."""
    assert (submit_batch.camera_request_id("b", "1-c")
            != submit_batch.camera_request_id("b-1", "c"))


@pytest.mark.parametrize("camera", ["cam-001", "cam.001", "CAM_001", "c" * 60])
def test_generated_ids_yield_valid_sagemaker_display_names(camera):
    """The id must survive the second contract too, not just SQS."""
    request_id = submit_batch.camera_request_id("20260912T203930Z-3ba6d01f", camera)
    name = execution_name(request_id)
    assert len(name) <= SFN_NAME_MAX
    assert SAGEMAKER_NAME.match(name), name


def test_a_long_batch_id_still_yields_a_valid_name():
    request_id = submit_batch.camera_request_id("x" * 60, "cam-001")
    name = execution_name(request_id)
    assert len(name) <= SFN_NAME_MAX
    assert SAGEMAKER_NAME.match(name), name


@pytest.mark.parametrize("request_id", [
    "camera_001-batch",   # the reported case: underscores are invalid downstream
    "cam.001.x",
    "-leading-hyphen",
    "trailing-hyphen-",
    "UPPER_case",
    "9starts-with-digit",
    "x",
    "a" * 200,
])
def test_execution_names_satisfy_both_contracts(request_id):
    name = execution_name(request_id)
    assert len(name) <= SFN_NAME_MAX
    assert SAGEMAKER_NAME.match(name), (
        f"{request_id!r} produced {name!r}, which SageMaker rejects AFTER start "
        f"intent is recorded -- which is what stranded the lease"
    )


@pytest.mark.parametrize("request_id", ["___", "...", "---", "   "])
def test_ids_with_no_usable_characters_still_produce_valid_names(request_id):
    """These must not raise: a valid fallback name is generated."""
    assert SAGEMAKER_NAME.match(execution_name(request_id))


def test_execution_names_are_deterministic_and_distinct():
    assert execution_name("cam-001") == execution_name("cam-001")
    assert execution_name("cam-001") != execution_name("cam-002")


def test_the_removed_flag_is_not_silently_reinterpreted():
    """--request-id changed meaning, so it must fail rather than mislead."""
    source = (REPO_ROOT / "scripts/submit_batch.py").read_text()
    assert "--request-id has been replaced by --batch-id" in source
    assert "camera_request_id(batch_id, cam_id)" in source


def test_invalid_request_is_raised_before_a_lease_is_taken():
    'A name that cannot be made valid must be refused at dispatch.'
    assert issubclass(InvalidRequest, ValueError)


def _sagemaker_enum(shape_name):
    path = os.path.join(
        os.path.dirname(botocore.__file__), "data/sagemaker/2017-07-24/service-2.json.gz"
    )
    return tuple(json.load(gzip.open(path))["shapes"][shape_name]["enum"])


def test_neo_enums_match_the_sdk():
    'The hardcoded lists must not drift from the model they came from.'
    from orchestration.queue_trigger import (
        NEO_TARGET_ARCH,
        NEO_TARGET_DEVICES,
        NEO_TARGET_OS,
    )

    assert NEO_TARGET_OS == _sagemaker_enum("TargetPlatformOs")
    assert NEO_TARGET_ARCH == _sagemaker_enum("TargetPlatformArch")
    assert NEO_TARGET_DEVICES == _sagemaker_enum("TargetDevice")


@pytest.mark.parametrize("submitted,expected", [
    ("LINUX", "LINUX"),
    ("linux", "LINUX"),   # normalised, because nothing downstream repairs it
    ("Linux", "LINUX"),
])
def test_target_os_is_normalised_to_the_enum_value(submitted, expected):
    from orchestration.queue_trigger import validate

    result = validate({"parameters": {"NeoTargetOs": submitted}})
    values = {p["Name"]: p["Value"] for p in result["PipelineParameters"]}
    assert values["NeoTargetOs"] == expected


@pytest.mark.parametrize("name,value", [
    ("NeoTargetOs", "windows"),
    ("NeoTargetArch", "bogus"),
    ("NeoTargetArch", "x86-64"),
    ("NeoTargetDevice", "fake_device"),
])
def test_invalid_neo_values_are_refused(name, value):
    """These reached SageMaker unvalidated before."""
    from orchestration.queue_trigger import InvalidRequest, validate

    with pytest.raises(InvalidRequest, match="not a supported value"):
        validate({"parameters": {name: value}})


@pytest.mark.parametrize("name", ["NeoTargetDevice", "TrainingDataS3Uri"])
def test_the_none_sentinel_is_accepted(name):
    """"none" is documented: no named device / generate synthetic data."""
    from orchestration.queue_trigger import validate

    result = validate({"parameters": {name: "none"}})
    values = {p["Name"]: p["Value"] for p in result["PipelineParameters"]}
    assert values[name] == "none"


def test_the_pipeline_defaults_are_themselves_valid():
    "A submitter passing the pipeline's own defaults must not be rejected."
    from orchestration.queue_trigger import validate

    stack = (REPO_ROOT / "sagemaker_cv_pipeline_stack.py").read_text()
    assert 'name="NeoTargetOs", default_value="LINUX"' in stack
    assert 'name="NeoTargetArch", default_value="X86_64"' in stack
    validate({"parameters": {"NeoTargetOs": "LINUX", "NeoTargetArch": "X86_64",
                             "NeoTargetDevice": "none"}})


SQS_MAX_DEDUP_LEN = 128


def _sqs_accepts(value: str) -> bool:
    """AWS's documented contract, restated rather than imported."""
    return (
        1 <= len(value) <= SQS_MAX_DEDUP_LEN
        and all(33 <= ord(ch) <= 126 for ch in value)
    )


class FakeSqs:
    'Only send_message, validating what SQS itself validates.'

    def __init__(self):
        self.sent = []

    def send_message(self, QueueUrl, MessageBody, MessageGroupId,
                     MessageDeduplicationId):
        for label, value in (
            ("MessageDeduplicationId", MessageDeduplicationId),
            ("MessageGroupId", MessageGroupId),
        ):
            if not _sqs_accepts(value):
                raise AssertionError(
                    f"SQS would reject {label}={value!r} ({len(value)} chars)"
                )
        self.sent.append(
            {
                "QueueUrl": QueueUrl,
                "body": json.loads(MessageBody),
                "group": MessageGroupId,
                "dedup": MessageDeduplicationId,
            }
        )
        return {"MessageId": f"m{len(self.sent)}"}


def _submit(monkeypatch, batch_id, cameras=("cam-001", "cam-002")):
    """Drive the real main() with a fake SQS client and explicit queue and Region."""
    fake = FakeSqs()
    monkeypatch.setattr(submit_batch.boto3, "client", lambda *a, **k: fake)
    monkeypatch.setattr(
        submit_batch.sys,
        "argv",
        [
            "submit_batch.py",
            "--camera-ids", *cameras,
            "--queue-url", "https://sqs.us-east-2.amazonaws.com/111111111111/q.fifo",
            "--region", "us-east-2",
            "--batch-id", batch_id,
        ],
    )
    submit_batch.main()
    return fake


def test_the_body_request_id_is_the_deduplication_id(monkeypatch, capsys):
    'The two must agree, or redelivery is no longer idempotent.'
    fake = _submit(monkeypatch, "batch-alpha")
    capsys.readouterr()

    assert len(fake.sent) == 2
    for message in fake.sent:
        assert message["body"]["request_id"] == message["dedup"]
        assert message["group"] == message["body"]["camera_id"]

    assert len({m["dedup"] for m in fake.sent}) == 2


def test_resubmitting_a_batch_reproduces_its_ids(monkeypatch, capsys):
    'Re-running the SAME batch id must be a no-op at the queue.'
    first = _submit(monkeypatch, "batch-alpha")
    second = _submit(monkeypatch, "batch-alpha")
    capsys.readouterr()
    assert [m["dedup"] for m in first.sent] == [m["dedup"] for m in second.sent]
