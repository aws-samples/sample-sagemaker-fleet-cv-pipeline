# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Tests for the lease manager and the queue dispatcher.'
import copy
import io
import re
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "orchestration"))

os.environ.setdefault("LEASE_TABLE_NAME", "test-leases")
os.environ.setdefault("MAX_CONCURRENCY", "3")
os.environ.setdefault("PIPELINE_NAME", "test-pipeline")
os.environ.setdefault("STATE_MACHINE_ARN",
                      "arn:aws:states:us-east-1:111111111111:stateMachine:test-sm")

import lease_handler  # noqa: E402
import queue_trigger  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402


DYNAMO_RESERVED_WORDS = {
    "PARAMETERS", "PARAMETER", "STATUS", "VALUE", "VALUES", "NAME", "SIZE",
    "TIMESTAMP", "COUNT", "ORDER", "YEAR", "SOURCE", "TOKEN", "OWNER", "SLOT",
}


class FakeDynamo:
    'Enough of DynamoDB to exercise conditional TransactWriteItems.'

    def __init__(self):
        self.items = {}
        self.transact_calls = 0
        self.scan_calls = 0
        self.page_size = None
        self.eventually_hidden = set()

    def hide_from_eventual_reads(self, lease_id: str) -> None:
        """Commit an item but keep it out of Scan and non-consistent GetItem."""
        assert lease_id in self.items, f"cannot hide an absent item: {lease_id}"
        self.eventually_hidden.add(lease_id)

    def reveal(self, lease_id: str) -> None:
        """Let eventual consistency catch up for *lease_id*."""
        self.eventually_hidden.discard(lease_id)

    @staticmethod
    def _key(key: dict) -> str:
        return key["LeaseId"]["S"]

    def get_item(self, TableName, Key, ConsistentRead=False):
        key = self._key(Key)
        if not ConsistentRead and key in self.eventually_hidden:
            return {}
        item = self.items.get(key)
        return {"Item": copy.deepcopy(item)} if item else {}

    @staticmethod
    def _resolve_alias(token, names):
        'The attribute an expression token refers to, rejecting undefined aliases.'
        if token.startswith("#"):
            if token not in names:
                raise ClientError(
                    {"Error": {
                        "Code": "ValidationException",
                        "Message": (
                            f"Invalid UpdateExpression: An expression attribute name used "
                            f"in the document path is not defined; attribute name: {token}"
                        ),
                    }},
                    "TransactWriteItems",
                )
            return names[token]
        return token

    def _assert_aliases_defined(self, expression, names):
        """Every #alias in an expression must be mapped, wherever it appears."""
        if not expression:
            return
        for token in re.findall(r"#[A-Za-z0-9_]+", expression):
            self._resolve_alias(token, names or {})

    def _condition_holds(self, item, expression, names, values) -> bool:
        if not expression:
            return True
        self._assert_aliases_defined(expression, names)
        if " AND " in expression:
            return all(
                self._condition_holds(item, part.strip(), names, values)
                for part in expression.split(" AND ")
            )
        if expression.startswith("attribute_not_exists"):
            inner = expression[expression.index("(") + 1: expression.rindex(")")]
            attr = self._resolve_alias(inner, names or {})
            return item is None or attr not in item
        if expression.startswith("attribute_exists"):
            inner = expression[expression.index("(") + 1: expression.rindex(")")]
            attr = self._resolve_alias(inner, names or {})
            return item is not None and (attr == "LeaseId" or attr in item)
        if "=" in expression:
            left, right = (part.strip() for part in expression.split("="))
            attr = self._resolve_alias(left, names or {})
            expected = values[right]
            return item is not None and item.get(attr) == expected
        raise AssertionError(f"unsupported condition: {expression}")

    def put_item(self, TableName, Item, ConditionExpression=None,
                 ExpressionAttributeNames=None, ExpressionAttributeValues=None):
        key = self._key(Item)
        if not self._condition_holds(self.items.get(key), ConditionExpression,
                                    ExpressionAttributeNames or {},
                                    ExpressionAttributeValues or {}):
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException", "Message": "no"}},
                "PutItem",
            )
        self.items[key] = dict(Item)
        return {}

    def delete_item(self, TableName, Key, ConditionExpression=None,
                    ExpressionAttributeNames=None, ExpressionAttributeValues=None):
        key = self._key(Key)
        if not self._condition_holds(self.items.get(key), ConditionExpression,
                                    ExpressionAttributeNames or {},
                                    ExpressionAttributeValues or {}):
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException", "Message": "no"}},
                "DeleteItem",
            )
        self.items.pop(key, None)
        return {}

    def transact_write_items(self, TransactItems):
        self.transact_calls += 1
        planned = []
        reasons = []
        ok = True
        for entry in TransactItems:
            op, spec = next(iter(entry.items()))
            key = self._key(spec.get("Item", spec.get("Key")))
            current = self.items.get(key)
            holds = self._condition_holds(
                current,
                spec.get("ConditionExpression"),
                spec.get("ExpressionAttributeNames", {}),
                spec.get("ExpressionAttributeValues", {}),
            )
            reasons.append({"Code": "None" if holds else "ConditionalCheckFailed"})
            if not holds:
                ok = False
            planned.append((op, key, spec))
        if not ok:
            raise ClientError(
                {"Error": {"Code": "TransactionCanceledException",
                           "Message": "cancelled"},
                 "CancellationReasons": reasons},
                "TransactWriteItems",
            )
        for op, key, spec in planned:
            if op == "Put":
                self.items[key] = dict(spec["Item"])
            elif op == "Delete":
                self.items.pop(key, None)
            elif op == "Update":
                item = self.items[key]
                names = spec.get("ExpressionAttributeNames", {})
                values = spec.get("ExpressionAttributeValues", {})
                clause = spec["UpdateExpression"].strip()
                assert clause.upper().startswith("SET "), clause
                self._assert_aliases_defined(clause, names)
                for assignment in clause[4:].split(","):
                    raw, placeholder = (p.strip() for p in assignment.split("="))
                    item[self._resolve_alias(raw, names)] = values[placeholder]
            elif op == "ConditionCheck":
                pass
            else:
                raise AssertionError(f"unsupported transaction action: {op}")
        return {}

    def update_item(self, TableName, Key, UpdateExpression,
                    ExpressionAttributeValues, ConditionExpression=None,
                    ExpressionAttributeNames=None):
        key = self._key(Key)
        item = self.items.get(key)
        if not self._condition_holds(item, ConditionExpression,
                                     ExpressionAttributeNames or {},
                                     ExpressionAttributeValues):
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException", "Message": "no"}},
                "UpdateItem",
            )
        names = ExpressionAttributeNames or {}

        expression = UpdateExpression.strip()
        self._assert_aliases_defined(expression, ExpressionAttributeNames)
        self._assert_aliases_defined(
            ConditionExpression, ExpressionAttributeNames)
        set_clause, remove_clause = expression, ""
        if " REMOVE " in expression:
            set_clause, remove_clause = expression.split(" REMOVE ", 1)
        elif expression.upper().startswith("REMOVE "):
            set_clause, remove_clause = "", expression[len("REMOVE "):]

        for raw in (r.strip() for r in remove_clause.split(",") if r.strip()):
            if not raw.startswith("#") and raw.upper() in DYNAMO_RESERVED_WORDS:
                raise ClientError(
                    {"Error": {"Code": "ValidationException",
                               "Message": f"Invalid UpdateExpression: Attribute name "
                                          f"is a reserved keyword; keyword: {raw}"}},
                    "UpdateItem",
                )
            item.pop(self._resolve_alias(raw, names), None)

        set_clause = set_clause.replace("SET ", "", 1).strip()
        for assignment in (a for a in set_clause.split(",") if a.strip()):
            raw, placeholder = (p.strip() for p in assignment.split("="))
            if not raw.startswith("#") and raw.upper() in DYNAMO_RESERVED_WORDS:
                raise ClientError(
                    {"Error": {"Code": "ValidationException",
                               "Message": f"Invalid UpdateExpression: Attribute name "
                                          f"is a reserved keyword; keyword: {raw}"}},
                    "UpdateItem",
                )
            attr = self._resolve_alias(raw, names)
            item[attr] = ExpressionAttributeValues[placeholder]
        return {}

    def scan(self, TableName, FilterExpression=None, ExpressionAttributeValues=None,
             ExclusiveStartKey=None):
        "Paginated, and the page limit applies BEFORE the filter, as DynamoDB's 1 MB Scan limit does. ``page_size`` of None keeps the single-page behaviour; setting it exercises the caller's pagination."
        keys = sorted(k for k in self.items if k not in self.eventually_hidden)
        if ExclusiveStartKey is not None:
            after = self._key(ExclusiveStartKey)
            keys = [k for k in keys if k > after]
        page_keys = keys if self.page_size is None else keys[: self.page_size]
        prefix = ExpressionAttributeValues[":p"]["S"] if ExpressionAttributeValues else ""
        items = [copy.deepcopy(self.items[k]) for k in page_keys if k.startswith(prefix)]
        result = {"Items": items}
        remaining = keys if self.page_size is None else keys[self.page_size:]
        if remaining and page_keys:
            result["LastEvaluatedKey"] = {"LeaseId": {"S": page_keys[-1]}}
        self.scan_calls += 1
        return result


class FakeS3:
    'Objects by key, with list_objects_v2 pagination and get_object bodies.'

    def __init__(self):
        self.objects = {}          # key -> bytes
        self.denied = set()        # keys whose GET raises AccessDenied
        self.list_error = None     # exception to raise from the paginator
        self.fail_after_page = None
        self.reads = []

    def put_record(self, key, payload):
        self.objects[key] = json.dumps(payload).encode()

    def put_raw(self, key, raw: bytes):
        self.objects[key] = raw

    def get_object(self, Bucket, Key):
        self.reads.append(Key)
        if Key in self.denied:
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
            )
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2", name
        outer = self

        class _P:
            def paginate(self, Bucket=None, Prefix="", **kwargs):
                if outer.list_error is not None:
                    raise outer.list_error
                keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                for index, key in enumerate(keys):
                    if (outer.fail_after_page is not None
                            and index >= outer.fail_after_page):
                        raise ClientError(
                            {"Error": {"Code": "InternalError", "Message": "boom"}},
                            "ListObjectsV2",
                        )
                    yield {"Contents": [{"Key": key}]}

        return _P()


@pytest.fixture()
def dynamo(monkeypatch):
    fake = FakeDynamo()
    monkeypatch.setattr(lease_handler, "dynamodb", fake)
    monkeypatch.setattr(lease_handler, "MAX_CONCURRENCY", 3)
    monkeypatch.setattr(lease_handler, "s3", FakeS3())
    monkeypatch.setattr(lease_handler, "RUNS_BUCKET", "test-bucket")
    monkeypatch.setattr(lease_handler, "RUNS_PREFIX", "cv-pipeline/runs")
    return fake


def _age_lease(dynamo, execution_name, seconds):
    """Backdate a lease so the reconciler considers it stale."""
    import time as _time
    key = lease_handler.OWNER_PREFIX + execution_name
    dynamo.items[key]["AcquiredAt"] = {"N": str(int(_time.time()) - seconds)}


def test_capacity_is_structural(dynamo):
    """The cap cannot be exceeded, because there are only that many slots."""
    for i in range(3):
        assert lease_handler.acquire(f"exec-{i}")["acquired"] is True
    assert lease_handler.acquire("exec-3") == {
        "acquired": False, "reason": "no_capacity"
    }


def test_acquire_is_idempotent(dynamo):
    'A retry after a lost response must not take a SECOND slot.'
    first = lease_handler.acquire("exec-a")
    second = lease_handler.acquire("exec-a")
    assert first["slot"] == second["slot"]
    assert second["reused"] is True
    assert lease_handler.acquire("exec-b")["acquired"] is True
    assert lease_handler.acquire("exec-c")["acquired"] is True
    assert lease_handler.acquire("exec-d")["acquired"] is False


def test_acquire_recovers_a_commit_whose_response_was_lost(dynamo):
    'A transaction that COMMITTED but returned an error must not double-book.'
    original = dynamo.transact_write_items
    raised = []

    def commit_then_lose_the_response(TransactItems):
        result = original(TransactItems)  # the write really does commit
        if not raised:
            raised.append(True)
            raise ClientError(
                {"Error": {"Code": "InternalServerError", "Message": "lost"}},
                "TransactWriteItems",
            )
        return result

    dynamo.transact_write_items = commit_then_lose_the_response
    with pytest.raises(ClientError):
        lease_handler.acquire("exec-a")
    dynamo.transact_write_items = original

    assert raised, "the error was never raised; this test proved nothing"

    retry = lease_handler.acquire("exec-a")
    assert retry == {"acquired": True, "slot": 0, "reused": True}

    slots = [k for k in dynamo.items if k.startswith(lease_handler.SLOT_PREFIX)]
    assert len(slots) == 1, f"the retry took a second slot: {sorted(slots)}"

    assert lease_handler.release("exec-a")["released"] is True
    assert [lease_handler.acquire(f"exec-{n}")["acquired"] for n in "bcd"] == [
        True, True, True
    ]


def test_release_is_idempotent(dynamo):
    'A repeated release must not free capacity twice.'
    lease_handler.acquire("exec-a")
    lease_handler.acquire("exec-b")
    for _ in range(4):
        assert lease_handler.release("exec-a")["released"] is True
    assert lease_handler.acquire("exec-c")["acquired"] is True
    assert lease_handler.acquire("exec-d")["acquired"] is True
    assert lease_handler.acquire("exec-e")["acquired"] is False


def test_release_does_not_free_another_executions_slot(dynamo):
    """A stale release must not free a slot someone else has since taken."""
    lease_handler.acquire("exec-a")
    lease_handler.release("exec-a")
    lease_handler.acquire("exec-b")          # takes slot 0
    assert lease_handler.release("exec-a")["already"] is True
    assert dynamo.items[f"{lease_handler.SLOT_PREFIX}0"]["Owner"]["S"] == "exec-b"


def test_record_start_will_not_resurrect_a_released_lease(dynamo):
    lease_handler.acquire("exec-a")
    lease_handler.release("exec-a")
    with pytest.raises(ClientError):
        lease_handler.record_start("exec-a", "tok-123")


class FakeSageMaker:
    def __init__(self, statuses=None, executions=None, compilation_jobs=None):
        self.statuses = statuses or {}
        self.executions = executions or []
        self.compilation_jobs = dict(compilation_jobs or {})
        self.compilation_describes = []
        self.stopped_compilations = []
        self.compilation_errors = {}

    def describe_compilation_job(self, CompilationJobName):
        self.compilation_describes.append(CompilationJobName)
        if CompilationJobName in self.compilation_errors:
            raise self.compilation_errors[CompilationJobName]
        if not re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", CompilationJobName or ""):
            raise ClientError(
                {"Error": {
                    "Code": "ValidationException",
                    "Message": (
                        f"1 validation error detected: Value '{CompilationJobName}' at "
                        f"'compilationJobName' failed to satisfy constraint: Member must "
                        f"satisfy regular expression pattern: "
                        f"[a-zA-Z0-9](-*[a-zA-Z0-9]){{0,62}}"
                    ),
                }},
                "DescribeCompilationJob",
            )
        if CompilationJobName not in self.compilation_jobs:
            raise ClientError(
                {"Error": {
                    "Code": "ValidationException",
                    "Message": f"No Compilation Job with name {CompilationJobName} found",
                }},
                "DescribeCompilationJob",
            )
        return {"CompilationJobStatus": self.compilation_jobs[CompilationJobName]}

    def stop_compilation_job(self, CompilationJobName):
        self.stopped_compilations.append(CompilationJobName)
        self.compilation_jobs[CompilationJobName] = "STOPPING"
        return {}

    def describe_pipeline_execution(self, PipelineExecutionArn):
        if PipelineExecutionArn not in self.statuses:
            raise ClientError(
                {"Error": {"Code": "ValidationException", "Message": "gone"}},
                "DescribePipelineExecution",
            )
        entry = self.statuses[PipelineExecutionArn]
        return entry if isinstance(entry, dict) else {"PipelineExecutionStatus": entry}

    def get_paginator(self, name):
        outer = self

        class _P:
            def paginate(self, **kwargs):
                return [{"PipelineExecutionSummaries": outer.executions}]
        return _P()


def _age(dynamo, execution_name, seconds):
    key = f"{lease_handler.OWNER_PREFIX}{execution_name}"
    dynamo.items[key]["AcquiredAt"] = {"N": str(int(dynamo.items[key]["AcquiredAt"]["N"]) - seconds)}


def test_reconcile_releases_a_terminal_pipeline(dynamo, monkeypatch):
    monkeypatch.setattr(lease_handler, "sagemaker",
                        FakeSageMaker({"arn:exec-a": "Succeeded"}))
    lease_handler.acquire("exec-a")
    lease_handler.record_start("exec-a", "tok-a", "arn:exec-a")
    _age(dynamo, "exec-a", 999999)
    result = lease_handler.reconcile()
    assert ("exec-a", "Succeeded") in result["released"]
    assert lease_handler.acquire("exec-b")["slot"] == 0


def test_reconcile_retains_a_running_pipeline(dynamo, monkeypatch):
    'Age must never free the slot of a pipeline that is still running.'
    monkeypatch.setattr(lease_handler, "sagemaker",
                        FakeSageMaker({"arn:exec-a": "Executing"}))
    lease_handler.acquire("exec-a")
    lease_handler.record_start("exec-a", "tok-a", "arn:exec-a")
    _age(dynamo, "exec-a", 999999)
    result = lease_handler.reconcile()
    assert result["released"] == []
    assert ("exec-a", "Executing") in result["retained"]


def test_reconcile_retains_when_status_is_unreadable(dynamo, monkeypatch):
    """An unreadable status is not evidence the pipeline stopped."""
    monkeypatch.setattr(lease_handler, "sagemaker", FakeSageMaker({}))
    lease_handler.acquire("exec-a")
    lease_handler.record_start("exec-a", "tok-a", "arn:missing")
    _age(dynamo, "exec-a", 999999)
    result = lease_handler.reconcile()
    assert result["released"] == []
    assert ("exec-a", "status_unreadable") in result["retained"]


class FakeSfn:
    'Models the real StartExecution idempotency contract.'

    def __init__(self):
        self.started = []

    def _find(self, name):
        for execution in self.started:
            if execution["name"] == name:
                return execution
        return None

    def close(self, name, status="SUCCEEDED"):
        """Mark an execution terminal, so its name can no longer be reused."""
        execution = self._find(name)
        assert execution, f"no execution named {name}"
        execution["status"] = status

    def start_execution(self, stateMachineArn, name, input):
        existing = self._find(name)
        if existing is not None:
            if existing["status"] != "RUNNING":
                raise ClientError(
                    {"Error": {"Code": "ExecutionAlreadyExists",
                               "Message": f"execution {name} already exists"}},
                    "StartExecution",
                )
            if existing["input"] != input:
                raise ClientError(
                    {"Error": {"Code": "ExecutionAlreadyExists",
                               "Message": f"execution {name} already exists with "
                                          f"different input"}},
                    "StartExecution",
                )
            return {"executionArn": existing["arn"]}

        arn = f"arn:exec:{name}"
        self.started.append(
            {"name": name, "input": input, "arn": arn, "status": "RUNNING"}
        )
        return {"executionArn": arn}

    def describe_execution(self, executionArn):
        name = executionArn.rsplit(":", 1)[-1]
        execution = self._find(name)
        if execution is None:
            raise ClientError(
                {"Error": {"Code": "ExecutionDoesNotExist", "Message": "no"}},
                "DescribeExecution",
            )
        return {"input": execution["input"], "status": execution["status"]}


@pytest.fixture()
def fake_sfn(monkeypatch):
    fake = FakeSfn()
    monkeypatch.setattr(queue_trigger, "sfn", fake)
    return fake


def _record(body: dict, message_id="msg-1") -> dict:
    return {"body": json.dumps(body), "messageId": message_id}


def test_duplicate_delivery_starts_only_one_execution(fake_sfn):
    'SQS is at-least-once. A redelivery must not train the camera twice.'
    record = _record({"camera_id": "cam-001"})
    first = queue_trigger.dispatch(record)
    second = queue_trigger.dispatch(record)

    assert len(fake_sfn.started) == 1
    assert first["executionArn"] == second["executionArn"]

    assert second["created"] is not True


def test_the_same_name_with_different_input_is_refused(fake_sfn):
    """A collision must not silently drop one of the two requests."""
    first = _record({"camera_id": "cam-001", "request_id": "shared"})
    clashing = _record({"camera_id": "cam-002", "request_id": "shared"})
    queue_trigger.dispatch(first)
    with pytest.raises(queue_trigger.InvalidRequest, match="DIFFERENT input"):
        queue_trigger.dispatch(clashing)


def test_contradictory_camera_identity_is_rejected(fake_sfn):
    """A body naming cam-A whose parameters say cam-B trained B and reported A."""
    with pytest.raises(queue_trigger.InvalidRequest, match="Contradictory camera"):
        queue_trigger.dispatch(_record({
            "camera_id": "cam-A", "parameters": {"CameraId": "cam-B"},
        }))


def test_unknown_parameter_is_rejected(fake_sfn):
    """An unknown name is silently ignored by the API, so a typo must fail here."""
    with pytest.raises(queue_trigger.InvalidRequest, match="Unknown pipeline parameter"):
        queue_trigger.dispatch(_record({"parameters": {"NumSamplez": 400}}))


def test_a_definitive_rejection_eventually_restores_capacity(dynamo, monkeypatch):
    """Required regression: known rejection eventually restores capacity."""
    monkeypatch.setattr(lease_handler, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(lease_handler, "sagemaker", FakeSageMaker(executions=[]))
    lease_handler.acquire("exec-a")
    lease_handler.record_start("exec-a", "token-a", display_name="exec-a")
    lease_handler.record_start_rejected("exec-a", "ValidationException")
    _age_lease(dynamo, "exec-a", lease_handler.STALE_LEASE_SECONDS + 60)

    result = lease_handler.reconcile()

    assert ("exec-a", "start_rejected") in result["released"]
    assert lease_handler.acquire("exec-b")["acquired"] is True, (
        "capacity must be recoverable after a definitive rejection"
    )


def test_a_lost_successful_response_never_admits_excess_work(dynamo, monkeypatch):
    """Required regression: an ambiguous outcome must still retain the slot."""
    monkeypatch.setattr(lease_handler, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(lease_handler, "sagemaker", FakeSageMaker(executions=[]))
    lease_handler.acquire("exec-a")
    lease_handler.record_start("exec-a", "token-a", display_name="exec-a")
    lease_handler.record_ambiguous_attempt("exec-a")
    _age_lease(dynamo, "exec-a", lease_handler.STALE_LEASE_SECONDS + 60)

    result = lease_handler.reconcile()

    assert result["released"] == []
    assert ("exec-a", "start_outcome_unknown") in result["retained"]
    assert lease_handler.acquire("exec-b")["acquired"] is False


def test_closing_admission_refuses_a_new_acquisition(dynamo):
    assert lease_handler.close_admission("teardown")["closed"] is True
    assert lease_handler.acquire("exec-a") == {
        "acquired": False, "reason": "admission_closed"
    }
    assert not [k for k in dynamo.items if k.startswith(lease_handler.SLOT_PREFIX)]
    assert not [k for k in dynamo.items if k.startswith(lease_handler.OWNER_PREFIX)]


def test_an_already_admitted_execution_keeps_its_lease_after_closure(dynamo, monkeypatch):
    'Closure refuses NEW admissions; it does not revoke existing ones.'
    monkeypatch.setattr(lease_handler, "sagemaker", FakeSageMaker(statuses={
        "arn:aws:sagemaker:us-east-2:1:pipeline/p/execution/abc": "Succeeded"}))
    first = lease_handler.acquire("exec-a")
    lease_handler.close_admission()

    again = lease_handler.acquire("exec-a")
    assert again == {"acquired": True, "slot": first["slot"], "reused": True}

    assert lease_handler.record_start(
        "exec-a", "tok-a", "arn:aws:sagemaker:us-east-2:1:pipeline/p/execution/abc"
    ) == {"recorded": True}
    assert lease_handler.release("exec-a")["released"] is True


def test_only_an_explicit_reopen_clears_the_barrier(dynamo):
    lease_handler.close_admission()
    assert lease_handler.admission_status()["closed"] is True
    assert lease_handler.acquire("exec-a")["acquired"] is False

    assert lease_handler.reopen_admission() == {"reopened": True}
    assert lease_handler.admission_status() == {"closed": False}
    assert lease_handler.acquire("exec-a")["acquired"] is True
