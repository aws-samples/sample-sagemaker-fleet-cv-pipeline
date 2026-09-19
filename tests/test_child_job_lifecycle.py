# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"The child compilation job's lifecycle, from the driver through to cleanup."
import importlib.util
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestration"))

from test_lease_manager import FakeDynamo, FakeS3, FakeSageMaker  # noqa: E402

import lease_handler  # noqa: E402
import neo_jobs  # noqa: E402

from config import (  # noqa: E402
    ANOMALY_MODEL_GROUP,
    OUTPUT_LEASE_TABLE_NAME,
    OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME,
    OUTPUT_QUEUE_URL,
    OUTPUT_STATE_MACHINE_ARN,
)

SFN_EXECUTION = "batch-2026-cam-001"
SM_EXECUTION_ID = "7f3d9b2e1a4c"
EXECUTION_ARN = (
    f"arn:aws:sagemaker:us-east-2:111111111111:pipeline/p/execution/{SM_EXECUTION_ID}"
)


def _load_cleanup():
    spec = importlib.util.spec_from_file_location(
        "cleanup_connected", REPO_ROOT / "scripts" / "cleanup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cleanup = _load_cleanup()

QUEUE_URL = "https://sqs.us-east-2.amazonaws.com/111111111111/training.fifo"
QUEUE_ARN = "arn:aws:sqs:us-east-2:111111111111:training.fifo"
TRIGGER_FN = "fleet-cv-queue-trigger"
LEASE_TABLE = "fleet-cv-leases"


class _SharedDynamoForCleanup:
    "Cleanup's DynamoDB view of the SAME store the lease handler writes to."

    def __init__(self, fake: FakeDynamo):
        self.fake = fake
        self.scans = []

    def put_item(self, TableName, Item):
        return self.fake.put_item(TableName=TableName, Item=Item)

    def scan(self, TableName, FilterExpression=None, ExpressionAttributeValues=None,
             ExclusiveStartKey=None, ConsistentRead=False):
        self.scans.append({"consistent": ConsistentRead})
        return self.fake.scan(
            TableName=TableName,
            FilterExpression=FilterExpression,
            ExpressionAttributeValues=ExpressionAttributeValues,
            ExclusiveStartKey=ExclusiveStartKey,
        )


class _FakeCfn:
    def describe_stacks(self, StackName):
        return {"Stacks": [{"Outputs": [
            {"OutputKey": OUTPUT_LEASE_TABLE_NAME, "OutputValue": LEASE_TABLE},
            {"OutputKey": OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME, "OutputValue": TRIGGER_FN},
            {"OutputKey": OUTPUT_QUEUE_URL, "OutputValue": QUEUE_URL},
            {"OutputKey": OUTPUT_STATE_MACHINE_ARN,
             "OutputValue": "arn:aws:states:us-east-2:111111111111:stateMachine:sm"},
        ]}]}


class _FakeSqs:
    def get_queue_attributes(self, QueueUrl, AttributeNames):
        if "QueueArn" in AttributeNames:
            return {"Attributes": {"QueueArn": QUEUE_ARN}}
        return {"Attributes": {"ApproximateNumberOfMessages": "0",
                               "ApproximateNumberOfMessagesNotVisible": "0"}}


class _FakeLambda:
    def __init__(self):
        self.updated = []

    def list_event_source_mappings(self, FunctionName):
        return {"EventSourceMappings": [
            {"UUID": "uuid-1", "EventSourceArn": QUEUE_ARN, "State": "Enabled"}
        ]}

    def update_event_source_mapping(self, UUID, Enabled):
        self.updated.append((UUID, Enabled))
        return {"UUID": UUID, "State": "Disabling"}

    def get_event_source_mapping(self, UUID):
        return {"UUID": UUID, "State": "Disabled"}


class _CleanupSageMaker:
    'Model packages, the empty pipeline-execution list, and compilation jobs.'

    def __init__(self, packages, compilation_jobs=()):
        self.packages = list(packages)
        self.deleted = []
        self.compilation_jobs = list(compilation_jobs)
        self.status_sequence = {}
        self._seen = {}

    def _next_status(self, name, default):
        'The status to report for this observation, advancing any scripted sequence.'
        sequence = self.status_sequence.get(name)
        if not sequence:
            return default
        index = self._seen.get(name, 0)
        self._seen[name] = index + 1
        return sequence[min(index, len(sequence) - 1)]

    def get_paginator(self, name):
        outer = self

        class _P:
            def paginate(self, **kwargs):
                if name == "list_pipeline_executions":
                    return [{"PipelineExecutionSummaries": []}]
                if name == "list_compilation_jobs":
                    wanted = kwargs.get("StatusEquals")
                    contains = kwargs.get("NameContains", "")
                    jobs = [
                        (n, outer._next_status(n, s))
                        for n, s in outer.compilation_jobs
                        if contains in n
                    ]
                    if wanted:
                        jobs = [(n, s) for n, s in jobs if s == wanted]
                    return [
                        {"CompilationJobSummaries": [
                            {"CompilationJobName": n, "CompilationJobStatus": s}
                        ]}
                        for n, s in jobs
                    ] or [{"CompilationJobSummaries": []}]
                return [{"ModelPackageSummaryList": [
                    {"ModelPackageArn": a} for a in outer.packages
                ]}]

        return _P()

    def delete_model_package(self, ModelPackageName):
        self.deleted.append(ModelPackageName)
        self.packages.remove(ModelPackageName)
        return {}


@pytest.fixture()
def connected(monkeypatch):
    """One lease store, shared by the lease handler and cleanup."""
    store = FakeDynamo()
    monkeypatch.setattr(lease_handler, "dynamodb", store)
    monkeypatch.setattr(lease_handler, "MAX_CONCURRENCY", 3)

    sagemaker = FakeSageMaker(statuses={EXECUTION_ARN: "Succeeded"})
    monkeypatch.setattr(lease_handler, "sagemaker", sagemaker)

    intent = FakeS3()
    monkeypatch.setattr(lease_handler, "s3", intent)
    monkeypatch.setattr(lease_handler, "RUNS_BUCKET", "test-bucket")
    monkeypatch.setattr(lease_handler, "RUNS_PREFIX", "cv-pipeline/runs")

    cleanup_sm = _CleanupSageMaker(["arn:pkg/1", "arn:pkg/2"])
    clients = {
        "cloudformation": _FakeCfn(),
        "dynamodb": _SharedDynamoForCleanup(store),
        "lambda": _FakeLambda(),
        "sqs": _FakeSqs(),
        "sagemaker": cleanup_sm,
        "stepfunctions": type("_S", (), {
            "get_paginator": lambda self, n: type("_P", (), {
                "paginate": lambda self, **k: [{"executions": []}]})()
        })(),
    }
    monkeypatch.setattr(cleanup.boto3, "client", lambda s, region_name=None: clients[s])
    monkeypatch.setattr(cleanup, "resolve_region", lambda explicit: "us-east-2")
    monkeypatch.setattr(cleanup, "MODEL_GROUPS", (ANOMALY_MODEL_GROUP,))
    monkeypatch.setattr(cleanup.time, "sleep", lambda _s: None)

    return type("Connected", (), {
        "store": store, "sagemaker": sagemaker, "cleanup_sm": cleanup_sm,
        "intent": intent,
    })()


def _run_cleanup(monkeypatch, argv=()):
    monkeypatch.setattr(cleanup.sys, "argv", ["cleanup.py", *argv])
    return cleanup.main()


def _admitted_execution_with_a_live_child(connected, child_status="INPROGRESS"):
    'A lease whose pipeline succeeded while one compilation child is still live.'
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)
    child = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "anomalib", 1)
    connected.sagemaker.compilation_jobs[child] = child_status
    return child


def test_release_retains_the_slot_while_a_compilation_child_runs(connected, monkeypatch):
    'The core of the defect, end to end.'
    child = _admitted_execution_with_a_live_child(connected)

    outcome = lease_handler.release(SFN_EXECUTION)
    assert outcome["released"] is False
    assert outcome["reason"] == "compilation_children_active"

    owner_key = lease_handler.OWNER_PREFIX + SFN_EXECUTION
    assert owner_key in connected.store.items

    assert _run_cleanup(monkeypatch) == 1
    assert connected.cleanup_sm.deleted == []
    assert connected.cleanup_sm.deleted == []

    connected.sagemaker.compilation_jobs[child] = "COMPLETED"
    assert lease_handler.release(SFN_EXECUTION)["released"] is True
    assert owner_key not in connected.store.items
    assert _run_cleanup(monkeypatch) == 0
    assert connected.cleanup_sm.deleted == ["arn:pkg/1", "arn:pkg/2"]


def test_all_terminal_children_release(connected, monkeypatch):
    """Both branches compiled and finished, so the slot must come back."""
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)
    for model_type, status in (("anomalib", "COMPLETED"), ("classifier", "FAILED")):
        name = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, model_type, 1)
        connected.sagemaker.compilation_jobs[name] = status

    assert lease_handler.release(SFN_EXECUTION)["released"] is True
    assert _run_cleanup(monkeypatch) == 0
    assert connected.cleanup_sm.deleted == ["arn:pkg/1", "arn:pkg/2"]


def _intent_for(connected, model_type, attempt=1, submission_id=None):
    'Record ONE immutable submission intent naming a specific attempt, as the driver does.'
    submission_id = submission_id or f"sub-{model_type}-{attempt}"
    job_name = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, model_type, attempt)
    key = neo_jobs.submission_key(
        "cv-pipeline/runs", SM_EXECUTION_ID, model_type, submission_id
    )
    connected.intent.put_record(key, neo_jobs.submission_record(
        submission_id=submission_id,
        execution_id=SM_EXECUTION_ID,
        model_type=model_type,
        job_name=job_name,
        request_digest="d" * 64,
        recorded_at=1,
    ))
    return key


def test_a_pipeline_that_never_compiled_releases(connected, monkeypatch):
    'Failing before the compile steps must not strand the slot.'
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)

    assert lease_handler.release(SFN_EXECUTION)["released"] is True
    assert _run_cleanup(monkeypatch) == 0


def test_recorded_intent_with_no_visible_job_retains(connected, monkeypatch):
    'The decisive case: create was accepted, its response lost, the job not yet visible.'
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)
    _intent_for(connected, "anomalib")   # submitted; job not observable
    name = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "classifier", 1)
    connected.sagemaker.compilation_jobs[name] = "COMPLETED"
    _intent_for(connected, "classifier")

    outcome = lease_handler.release(SFN_EXECUTION)
    assert outcome["released"] is False
    assert connected.intent.reads, "the intent marker was never read"

    assert _run_cleanup(monkeypatch) == 1
    assert connected.cleanup_sm.deleted == []

    late = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "anomalib", 1)
    connected.sagemaker.compilation_jobs[late] = "COMPLETED"
    assert lease_handler.release(SFN_EXECUTION)["released"] is True
    assert _run_cleanup(monkeypatch) == 0


def test_an_older_terminal_attempt_does_not_settle_a_newer_submission(connected, monkeypatch):
    """Attempt 1 FAILED, attempt 2 submitted and not yet visible. Must RETAIN."""
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)

    first = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "anomalib", 1)
    connected.sagemaker.compilation_jobs[first] = "FAILED"
    _intent_for(connected, "anomalib", attempt=2)

    done = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "classifier", 1)
    connected.sagemaker.compilation_jobs[done] = "COMPLETED"
    _intent_for(connected, "classifier", attempt=1)

    outcome = lease_handler.release(SFN_EXECUTION)
    assert outcome["released"] is False, (
        "an older FAILED attempt settled the branch while the recorded attempt 2 was "
        "still unobservable, so the lease was released with a compilation in flight"
    )
    assert outcome["reason"] == "compilation_children_unreadable"

    assert _run_cleanup(monkeypatch) == 1
    assert connected.cleanup_sm.deleted == []

    second = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "anomalib", 2)
    connected.sagemaker.compilation_jobs[second] = "COMPLETED"
    assert lease_handler.release(SFN_EXECUTION)["released"] is True
    assert _run_cleanup(monkeypatch) == 0


class _DriverS3:
    """Enough S3 for the driver: upload_file, put_object, get_object over one dict."""

    def __init__(self):
        self.objects = {}
        self.put_error = None
        self.puts = []

    def upload_file(self, filename, Bucket, Key):
        self.objects[Key] = b"tar"

    def put_object(self, Bucket, Key, Body):
        self.puts.append(Key)
        if self.put_error is not None:
            raise self.put_error
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.objects[Key])}

    def submissions(self, model_type="anomalib"):
        prefix = neo_jobs.submissions_prefix(
            "cv-pipeline/runs", SM_EXECUTION_ID, model_type)
        return {k: json.loads(v) for k, v in self.objects.items() if k.startswith(prefix)}


class _DriverSm:
    """describe/create with scripted create behaviour and a visible-job set."""

    def __init__(self, create_effects, observer=None):
        self.create_effects = list(create_effects)
        self.visible = {}
        self.creates = []
        self.observer = observer
        self.record_at_create = None

    def describe_compilation_job(self, CompilationJobName):
        if CompilationJobName in self.visible:
            return {"CompilationJobStatus": self.visible[CompilationJobName],
                    "ModelArtifacts": {"S3ModelArtifacts": "s3://b/model.tar.gz"}}
        raise ClientError(
            {"Error": {"Code": "ValidationException",
                       "Message": f"No Compilation Job with name {CompilationJobName} found"}},
            "DescribeCompilationJob",
        )

    def create_compilation_job(self, CompilationJobName, **kwargs):
        self.creates.append(CompilationJobName)
        self.record_at_create = None
        if self.observer is not None:
            self.record_at_create = self.observer(CompilationJobName)
        if self.create_effects:
            raise self.create_effects.pop(0)


def _run_driver(monkeypatch, s3, sm, tmp_path, model_type="anomalib"):
    'Invoke the production main() with fakes, returning the exception it raised.'
    import compile_neo_step

    with monkeypatch.context() as scoped:
        scoped.setattr(compile_neo_step, "OUTPUT_DIR", tmp_path / "out")
        scoped.setattr(compile_neo_step, "repackage_onnx_only",
                       lambda d: tmp_path / "model.tar.gz")
        scoped.setattr(compile_neo_step.boto3, "client",
                       lambda service, region_name=None: s3 if service == "s3" else sm)
        scoped.setattr(sys, "argv", [
            "compile_neo_step.py",
            "--region", "us-east-2", "--role-arn", "arn:aws:iam::1:role/r",
            "--bucket", "bkt", "--execution-id", SM_EXECUTION_ID,
            "--runs-prefix", "cv-pipeline/runs", "--model-type", model_type,
            "--image-size", "224", "--camera-id", "cam-001",
        ])
        try:
            compile_neo_step.main()
            return None
        except ClientError as exc:
            return exc
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"the driver raised {type(exc).__name__}, which is not a service failure: "
                f"{exc}"
            ) from exc


def test_the_driver_records_one_immutable_intent_before_creating(monkeypatch, tmp_path):
    'The record must be DURABLE at the moment create is called, and name the exact job.'
    s3 = _DriverS3()
    sm = _DriverSm(
        [ClientError({"Error": {"Code": "RequestTimeout", "Message": "read timeout"}},
                     "CreateCompilationJob")],
        observer=lambda name: dict(s3.submissions()),
    )
    exc = _run_driver(monkeypatch, s3, sm, tmp_path)
    assert isinstance(exc, ClientError)
    assert exc.response["Error"]["Code"] == "RequestTimeout"
    assert exc.operation_name == "CreateCompilationJob"

    assert len(sm.creates) == 1, f"create was called {len(sm.creates)} times"
    assert sm.record_at_create, "create ran with no submission record durable"
    at_create = next(iter(sm.record_at_create.values()))
    assert at_create["job_name"] == sm.creates[0]

    records = s3.submissions()
    assert len(records) == 1, records
    record = next(iter(records.values()))
    assert record["job_name"] == neo_jobs.deterministic_job_name(
        SM_EXECUTION_ID, "anomalib", 1)
    assert record["schema_version"] == neo_jobs.SUBMISSION_SCHEMA_VERSION
    assert len(record["request_digest"]) == 64
    assert "outcome" not in record
    assert "attempts" not in record


def test_a_create_failure_leaves_the_record_untouched(monkeypatch, tmp_path):
    """Whatever the failure, the driver writes the record exactly once."""
    for code in ("RequestTimeout", "ValidationException", "ResourceLimitExceeded",
                 "AccessDeniedException", "ThrottlingException"):
        s3 = _DriverS3()
        sm = _DriverSm([ClientError(
            {"Error": {"Code": code, "Message": code}}, "CreateCompilationJob")])
        exc = _run_driver(monkeypatch, s3, sm, tmp_path)
        assert isinstance(exc, ClientError) and exc.response["Error"]["Code"] == code
        assert len(sm.creates) == 1, (code, sm.creates)

        records = s3.submissions()
        assert len(records) == 1, (code, records)
        assert "outcome" not in next(iter(records.values())), code
        intent_puts = [k for k in s3.puts if "/submissions/" in k]
        assert len(intent_puts) == 1, (code, s3.puts)


def test_a_failed_intent_write_prevents_the_create(monkeypatch, tmp_path):
    """An unrecorded compilation is the one state nothing downstream can reason about."""
    s3 = _DriverS3()
    s3.put_error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject")
    sm = _DriverSm([])
    exc = _run_driver(monkeypatch, s3, sm, tmp_path)
    assert isinstance(exc, ClientError)
    assert exc.response["Error"]["Code"] == "AccessDenied"
    assert exc.operation_name == "PutObject"
    assert sm.creates == [], "create was called without a durable submission record"


def _receipt(connected, model_type, submission_id, decision=None, **overrides):
    'Write a COMPLETE receipt, bound to the record reconciliation will read.'
    intent_key = overrides.pop("intent_key", neo_jobs.submission_key(
        "cv-pipeline/runs", SM_EXECUTION_ID, model_type, submission_id))
    raw = connected.intent.objects.get(intent_key, b"")
    digest = overrides.pop("intent_digest", hashlib.sha256(raw).hexdigest())
    recorded = json.loads(raw) if raw else {}
    job_name = overrides.pop("job_name", recorded.get("job_name") or "")

    key = lease_handler._adjudication_key(SM_EXECUTION_ID, model_type, submission_id)
    item = {
        "LeaseId": key["LeaseId"],
        "Decision": {"S": decision or lease_handler.ADJUDICATION_NEVER_CREATED},
        "Operator": {"S": "sample-operator"},
        "ModelType": {"S": model_type},
        "SubmissionId": {"S": submission_id},
        "JobName": {"S": job_name},
        "IntentKey": {"S": intent_key},
        "IntentDigest": {"S": digest},
    }
    for field, value in overrides.items():
        item[field] = {"S": value}
    connected.store.items[key["LeaseId"]["S"]] = item
    return item


def _adjudicate(connected, **overrides):
    kwargs = dict(
        execution_name=SFN_EXECUTION, model_type="anomalib", submission_id="s1",
        operator="sample-operator", note="create refused before any request was sent",
        evidence="request-id abc123; CloudTrail 2026-09-12T23:00Z; no retries recorded",
    )
    kwargs.update(overrides)
    return lease_handler.adjudicate_compilation_submission(**kwargs)


def _held_lease_with_absent_submission(connected, attempt=1, submission_id="s1"):
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)
    _intent_for(connected, "anomalib", attempt=attempt, submission_id=submission_id)
    connected.sagemaker.statuses[EXECUTION_ARN] = "Failed"      # parent terminal
    return submission_id


def test_adjudication_records_a_receipt_and_reconciliation_releases(connected, monkeypatch):
    """The happy path -- and it must NOT release the lease itself."""
    _held_lease_with_absent_submission(connected)
    done = neo_jobs.deterministic_job_name(SM_EXECUTION_ID, "classifier", 1)
    connected.sagemaker.compilation_jobs[done] = "COMPLETED"
    _intent_for(connected, "classifier", attempt=1, submission_id="c1")

    result = _adjudicate(connected)
    assert result["adjudicated"] is True, result
    assert result["released"] is False, "the action released the lease directly"
    assert lease_handler._read_owner(SFN_EXECUTION), "the owner was deleted"

    _receipt(connected, "classifier", "c1")
    assert lease_handler.release(SFN_EXECUTION)["released"] is True


def test_an_unreadable_parent_status_retains(connected, monkeypatch):
    """An unanswered question is not a terminal pipeline."""
    assert lease_handler.acquire(SFN_EXECUTION)["acquired"] is True
    lease_handler.record_start(SFN_EXECUTION, "token", EXECUTION_ARN,
                               display_name=SFN_EXECUTION)

    def _boom(arn):
        raise EndpointConnectionError(endpoint_url="https://api.sagemaker")
    monkeypatch.setattr(lease_handler, "_pipeline_status", _boom)

    result = lease_handler.release(SFN_EXECUTION)
    assert result["released"] is False
    assert result["reason"] == "parent_status_unavailable"
