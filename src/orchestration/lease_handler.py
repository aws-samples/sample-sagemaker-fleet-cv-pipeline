# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Concurrency control by per-execution lease, not by an arithmetic counter.'
import hashlib
import json
import logging
import os
import time
import uuid

from botocore.exceptions import BotoCoreError, ClientError

_TRANSPORT_ERRORS = (ClientError, BotoCoreError, OSError, ValueError)

import neo_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TABLE_NAME = os.environ["LEASE_TABLE_NAME"]
MAX_CONCURRENCY = int(os.environ["MAX_CONCURRENCY"])
PIPELINE_NAME = os.environ["PIPELINE_NAME"]
# A lease older than this is a candidate for reconciliation. It is NOT grounds to
# free the slot -- reconcile checks the real pipeline status.
STALE_LEASE_SECONDS = int(os.environ.get("STALE_LEASE_SECONDS", "21600"))  # 6h

SLOT_PREFIX = "slot#"
OWNER_PREFIX = "owner#"
TERMINAL_STATUSES = ("Succeeded", "Failed", "Stopped")

CONTROL_PREFIX = "control#"
ADMISSION_CLOSED_ID = f"{CONTROL_PREFIX}admission-closed"

ADJUDICATION_NEVER_CREATED = "no-compilation-job-was-created"


ATTR_NAMES = {
    "#token": "ClientToken",
    "#arn": "ExecutionArn",
    "#params": "Parameters",
    "#display": "DisplayName",
}


def _client(name):
    'Create a boto3 client on first use.'
    if name not in _CLIENTS:
        import boto3  # noqa: PLC0415 -- deliberately lazy, see above
        _CLIENTS[name] = boto3.client(name, config=_READ_CLIENT_CONFIG.get(name))
    return _CLIENTS[name]


_CLIENTS = {}


def _bounded_read_config():
    'Bound the READ clients so one unresponsive endpoint cannot eat the invocation.'
    from botocore.config import Config  # noqa: PLC0415 -- lazy, as above
    return Config(
        connect_timeout=2,
        read_timeout=3,
        retries={"mode": "standard", "total_max_attempts": 1},
    )


class _LazyReadConfig(dict):
    """Build the config on first use, so importing this module still touches no botocore."""

    def get(self, name, default=None):     # noqa: D102
        if name not in ("sagemaker", "s3"):
            return default
        if name not in self:
            self[name] = _bounded_read_config()
        return self[name]


_READ_CLIENT_CONFIG = _LazyReadConfig()


class _LazyClient:
    'Attribute proxy so ``dynamodb.get_item(...)`` still reads naturally.'

    def __init__(self, service):
        self._service = service

    def __getattr__(self, item):
        return getattr(_client(self._service), item)


dynamodb = _LazyClient("dynamodb")
sagemaker = _LazyClient("sagemaker")
s3 = _LazyClient("s3")

RUNS_BUCKET = os.environ.get("RUNS_BUCKET", "")
RUNS_PREFIX = os.environ.get("RUNS_PREFIX", "")


class LeaseError(RuntimeError):
    """Raised for conditions the state machine should treat as a failure."""


def _owner_key(execution_name: str) -> dict:
    return {"LeaseId": {"S": f"{OWNER_PREFIX}{execution_name}"}}


def _slot_key(slot: int) -> dict:
    return {"LeaseId": {"S": f"{SLOT_PREFIX}{slot}"}}


def _read_owner(execution_name: str) -> dict:
    """Strongly consistent read of an ownership record, or {} if absent."""
    response = dynamodb.get_item(
        TableName=TABLE_NAME, Key=_owner_key(execution_name), ConsistentRead=True
    )
    return response.get("Item", {})


def acquire(execution_name: str, request_id: str = None) -> dict:
    'Claim a slot for *execution_name*, or report that none is free.'
    existing = _read_owner(execution_name)
    if existing:
        slot = int(existing["Slot"]["N"])
        logger.info("Execution %s already holds slot %d", execution_name, slot)
        return {"acquired": True, "slot": slot, "reused": True}

    now = int(time.time())
    for slot in range(MAX_CONCURRENCY):
        try:
            slot_action, owner_action, barrier_action = 0, 1, 2
            dynamodb.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": TABLE_NAME,
                            "Item": {
                                **_slot_key(slot),
                                "Owner": {"S": execution_name},
                                "AcquiredAt": {"N": str(now)},
                            },
                            # Capacity: the slot must be free. Structural, so no
                            # arithmetic can exceed max_concurrency.
                            "ConditionExpression": "attribute_not_exists(#o)",
                            "ExpressionAttributeNames": {"#o": "Owner"},
                        }
                    },
                    {
                        "Put": {
                            "TableName": TABLE_NAME,
                            "Item": {
                                **_owner_key(execution_name),
                                "Slot": {"N": str(slot)},
                                "AcquiredAt": {"N": str(now)},
                                "ExecutionName": {"S": execution_name},
                                **({"RequestId": {"S": request_id}} if request_id else {}),
                            },
                            # Idempotency: a retry whose first attempt committed
                            # finds its own record and is rejected here, then the
                            # _read_owner above returns the slot it already has.
                            "ConditionExpression": "attribute_not_exists(LeaseId)",
                        }
                    },
                    {
                        "ConditionCheck": {
                            "TableName": TABLE_NAME,
                            "Key": {"LeaseId": {"S": ADMISSION_CLOSED_ID}},
                            "ConditionExpression": "attribute_not_exists(LeaseId)",
                        }
                    },
                ]
            )
            logger.info("Execution %s acquired slot %d", execution_name, slot)
            return {"acquired": True, "slot": slot, "reused": False}
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "TransactionCanceledException":
                raise
            reasons = [
                r.get("Code") for r in exc.response.get("CancellationReasons", [])
            ]

            def failed(index: int) -> bool:
                return len(reasons) > index and reasons[index] == "ConditionalCheckFailed"

            if failed(owner_action):
                existing = _read_owner(execution_name)
                if existing:
                    held = int(existing["Slot"]["N"])
                    logger.info(
                        "Execution %s already held slot %d (lost response)",
                        execution_name, held,
                    )
                    return {"acquired": True, "slot": held, "reused": True}

            if failed(barrier_action):
                logger.warning(
                    "Admission is closed; refusing to acquire a slot for %s",
                    execution_name,
                )
                return {"acquired": False, "reason": "admission_closed"}

            if not failed(slot_action):
                logger.error(
                    "Unrecognised transaction cancellation acquiring for %s: %s",
                    execution_name, reasons,
                )
                raise

            # This slot is taken; try the next one.
            continue

    logger.info("No free slot for %s (cap %d)", execution_name, MAX_CONCURRENCY)
    return {"acquired": False, "reason": "no_capacity"}


def close_admission(reason: str = None) -> dict:
    'Close admission. Idempotent, and durable until explicitly reopened.'
    existing = dynamodb.get_item(
        TableName=TABLE_NAME,
        Key={"LeaseId": {"S": ADMISSION_CLOSED_ID}},
        ConsistentRead=True,
    ).get("Item", {})
    created_at = existing.get("CreatedAt", {"N": str(int(time.time()))})
    item = {
        "LeaseId": {"S": ADMISSION_CLOSED_ID},
        "CreatedAt": created_at,
        "ReassertedAt": {"N": str(int(time.time()))},
    }
    if reason:
        item["Reason"] = {"S": reason}
    dynamodb.put_item(TableName=TABLE_NAME, Item=item)
    logger.warning(
        "Admission CLOSED (%s). No new slot can be acquired until it is reopened.",
        reason or "no reason given",
    )
    return {"closed": True, "already": bool(existing)}


def reopen_admission() -> dict:
    'Reopen admission. The ONLY way the barrier is cleared.'
    dynamodb.delete_item(
        TableName=TABLE_NAME, Key={"LeaseId": {"S": ADMISSION_CLOSED_ID}}
    )
    logger.warning("Admission REOPENED; new executions can acquire slots again.")
    return {"reopened": True}


def admission_status() -> dict:
    'Whether admission is closed, read consistently.'
    item = dynamodb.get_item(
        TableName=TABLE_NAME,
        Key={"LeaseId": {"S": ADMISSION_CLOSED_ID}},
        ConsistentRead=True,
    ).get("Item", {})
    if not item:
        return {"closed": False}
    return {
        "closed": True,
        "closed_at": int(item["CreatedAt"]["N"]),
        "reason": item.get("Reason", {}).get("S"),
    }


def record_start(execution_name: str, client_token: str, pipeline_arn: str = None,
                 parameters=None, display_name: str = None) -> dict:
    'Persist the start identity on the ownership record.'
    updates = ["#token = :t"]
    values = {":t": {"S": client_token}}
    names = {"#token": ATTR_NAMES["#token"]}
    if pipeline_arn:
        updates.append("#arn = :a")
        values[":a"] = {"S": pipeline_arn}
        names["#arn"] = ATTR_NAMES["#arn"]
    if parameters is not None:
        updates.append("#params = :p")
        values[":p"] = {"S": json.dumps(parameters, sort_keys=True)}
        names["#params"] = ATTR_NAMES["#params"]
    if display_name:
        updates.append("#display = :d")
        values[":d"] = {"S": display_name}
        names["#display"] = ATTR_NAMES["#display"]
    dynamodb.update_item(
        TableName=TABLE_NAME,
        Key=_owner_key(execution_name),
        UpdateExpression="SET " + ", ".join(updates),
        ExpressionAttributeValues=values,
        ExpressionAttributeNames=names,
        # Do not resurrect a released lease.
        ConditionExpression="attribute_exists(LeaseId)",
    )
    return {"recorded": True}


def release_rejected(execution_name: str) -> dict:
    'Release a lease whose start SageMaker definitively refused.'
    existing = _read_owner(execution_name)
    if not existing:
        return {"released": True, "already": True}

    slot = int(existing["Slot"]["N"])
    try:
        dynamodb.transact_write_items(
            TransactItems=[
                {
                    "Delete": {
                        "TableName": TABLE_NAME,
                        "Key": _slot_key(slot),
                        "ConditionExpression": "#o = :owner",
                        "ExpressionAttributeNames": {"#o": "Owner"},
                        "ExpressionAttributeValues": {":owner": {"S": execution_name}},
                    }
                },
                {
                    "Delete": {
                        "TableName": TABLE_NAME,
                        "Key": _owner_key(execution_name),
                        "ConditionExpression": (
                            "attribute_exists(LeaseId) AND attribute_exists(#r) "
                            "AND attribute_not_exists(#a) AND attribute_not_exists(#e)"
                        ),
                        "ExpressionAttributeNames": {
                            "#r": "StartRejected",
                            "#a": "AmbiguousAttempt",
                            "#e": "ExecutionArn",
                        },
                    }
                },
            ]
        )
        logger.info("Released slot %d for definitively rejected %s", slot,
                    execution_name)
        return {"released": True, "already": False, "slot": slot}
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "TransactionCanceledException":
            raise
        if not _read_owner(execution_name):
            return {"released": True, "already": True}
        logger.warning(
            "Not releasing %s: a precondition changed after the scan (ambiguity "
            "recorded, an execution ARN appeared, or the rejection was cleared). "
            "RETAINING.", execution_name,
        )
        return {
            "released": False,
            "retained": True,
            "reason": "rejection_preconditions_changed",
        }


def _persist_discovered_arn(execution_name: str, arn: str) -> None:
    'Record an ARN recovered by display name, without clobbering a concurrent write.'
    try:
        dynamodb.update_item(
            TableName=TABLE_NAME,
            Key=_owner_key(execution_name),
            UpdateExpression="SET #arn = :a",
            ConditionExpression="attribute_exists(LeaseId) AND attribute_not_exists(#arn)",
            ExpressionAttributeNames={"#arn": ATTR_NAMES["#arn"]},
            ExpressionAttributeValues={":a": {"S": arn}},
        )
        logger.warning("Recorded ARN %s discovered for lease %s by display name",
                       arn, execution_name)
        return "persisted"
    except _TRANSPORT_ERRORS as exc:
        if (not isinstance(exc, ClientError)
                or exc.response["Error"]["Code"] != "ConditionalCheckFailedException"):
            logger.warning("Could not record the discovered ARN for %s: %s",
                           execution_name, exc)
            return "failed"
        try:
            current = _read_owner(execution_name).get("ExecutionArn", {}).get("S")
        except _TRANSPORT_ERRORS as read_exc:
            logger.warning(
                "Could not establish the authoritative execution for %s after a persistence "
                "conflict: %s", execution_name, read_exc,
            )
            return "failed"
        if current and current != arn:
            logger.warning(
                "Lease %s now records execution %s, not the %s just correlated; the "
                "correlation is stale", execution_name, current, arn,
            )
            return "conflict"
        return "already"


def adjudicate_start_never_occurred(execution_name: str, operator: str,
                                    note: str = None, evidence: str = None) -> dict:
    "Record an operator's judgement that no pipeline was ever started, then release."
    if not operator:
        raise LeaseError(
            "adjudicate_start_never_occurred requires an 'Operator' identifying who is "
            "asserting that no pipeline was started. The judgement is recorded, so it "
            "must be attributable."
        )

    # BEFORE reading or mutating ownership. Refused rather than raised, so a sweep calling this
    # is not aborted by a missing field.
    if not isinstance(evidence, str) or not evidence.strip():
        return {
            "released": False,
            "retained": True,
            "reason": "missing_evidence",
            "detail": ("Evidence must account for every start attempt that could have been "
                       "delivered. Console absence alone is insufficient: a point-in-time "
                       "listing showing no execution does not establish that none was "
                       "created, and releasing on it has been shown to break the "
                       "concurrency cap."),
        }

    existing = _read_owner(execution_name)
    if not existing:
        logger.info("No lease for %s; already released", execution_name)
        return {"released": True, "already": True}

    if existing.get("ExecutionArn", {}).get("S"):
        return {
            "released": False,
            "retained": True,
            "reason": "execution_arn_present",
            "detail": (
                "An execution ARN has been recorded for this lease, so a pipeline WAS "
                "started. Use reconcile, which will resolve it against real status."
            ),
        }

    display_name = existing.get("DisplayName", {}).get("S")
    if display_name:
        try:
            correlated_arn, correlated_status = _find_execution_by_display_name(display_name)
        except ClientError as exc:
            logger.warning("Cannot correlate %s before adjudicating: %s",
                           execution_name, exc)
            return {
                "released": False,
                "retained": True,
                "reason": "correlation_unavailable",
                "detail": (
                    "Whether a pipeline was started could not be established, so the "
                    "lease is retained. Retry once SageMaker is reachable."
                ),
            }
        if correlated_arn:
            # Persist what was discovered so the next reconcile resolves it from the record
            # rather than rediscovering it, and so a second operator sees it too. Written
            # conditionally: if another writer recorded an ARN meanwhile, theirs stands.
            _persist_discovered_arn(execution_name, correlated_arn)
            return {
                "released": False,
                "retained": True,
                "reason": "execution_found_by_display_name",
                "detail": (
                    f"A pipeline execution IS correlated with this lease "
                    f"({correlated_arn}, status {correlated_status}). The ARN write was "
                    f"lost, not the start. Its ARN has now been recorded; use reconcile, "
                    f"which resolves it against real status."
                ),
            }

    blocked = _children_block_release(
        execution_name, existing, correlation=(None, None) if display_name else None
    )
    if blocked and blocked != "compilation_children_unnameable":
        return {"released": False, "retained": True, "reason": blocked}

    slot = int(existing["Slot"]["N"])
    now = int(time.time())

    receipt_id = f"adjudication#start#{execution_name}#{uuid.uuid4().hex}"
    receipt = {
        "LeaseId": {"S": receipt_id},
        "Decision": {"S": "no-pipeline-execution-was-created"},
        "ExecutionName": {"S": execution_name},
        "Slot": {"N": str(slot)},
        "AdjudicatedBy": {"S": operator},
        "AdjudicatedAt": {"N": str(now)},
        "AdjudicationEvidence": {"S": evidence},
    }
    if note:
        receipt["AdjudicationNote"] = {"S": note}
    # Carry the identifying fields forward, or the receipt cannot be tied back to what it judged.
    for field in ("ClientToken", "DisplayName", "AcquiredAt", "RequestId"):
        if field in existing:
            receipt[field] = existing[field]

    try:
        dynamodb.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": TABLE_NAME,
                        "Item": receipt,
                        "ConditionExpression": "attribute_not_exists(LeaseId)",
                    }
                },
                {
                    "Delete": {
                        "TableName": TABLE_NAME,
                        "Key": _slot_key(slot),
                        "ConditionExpression": "#o = :owner",
                        "ExpressionAttributeNames": {"#o": "Owner"},
                        "ExpressionAttributeValues": {":owner": {"S": execution_name}},
                    }
                },
                {
                    "Delete": {
                        "TableName": TABLE_NAME,
                        "Key": _owner_key(execution_name),
                        # The judgement is only valid for the state it was made about.
                        "ConditionExpression": (
                            "attribute_exists(LeaseId) AND attribute_not_exists(#arn)"
                        ),
                        "ExpressionAttributeNames": {"#arn": ATTR_NAMES["#arn"]},
                    }
                },
            ]
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "TransactionCanceledException":
            raise
        try:
            current_owner = _read_owner(execution_name)
        except _TRANSPORT_ERRORS as read_exc:
            logger.warning("Cannot reread adjudicated lease %s: %s", execution_name, read_exc)
        else:
            if not current_owner:
                return {"released": True, "already": True}
        return {
            "released": False,
            "retained": True,
            "reason": "state_changed_during_adjudication",
        }

    # a convenience copy. The DURABLE record is the receipt written in the release transaction;
    # this line exists so the release is visible in CloudWatch without a table read, and carries
    # the receipt id so the two can be tied together.
    logger.warning(
        "ADJUDICATED RELEASE slot=%d lease=%s operator=%s note=%s evidence=%s "
        "receipt=%s judged_at=%d: no pipeline was started, so the recorded start intent was "
        "cleared by hand",
        slot, execution_name, operator, note or "<none>", evidence, receipt_id, now,
    )
    return {
        "released": True,
        "already": False,
        "slot": slot,
        "adjudicated_by": operator,
        "adjudication_note": note,
        # Returned as well as stored: an operator who ran this needs to be able to find the
        # receipt afterwards without scanning the table.
        "adjudication_evidence": evidence,
        "adjudication_receipt": receipt_id,
        "adjudicated_at": now,
    }


def _sagemaker_execution_id(owner_item: dict):
    'The SageMaker pipeline execution id for this lease, or None.'
    arn = owner_item.get("ExecutionArn", {}).get("S")
    if not arn:
        return None
    return arn.rsplit("/", 1)[-1]


def _adjudication_key(execution_id: str, model_type: str, submission_id: str) -> dict:
    "Key of the immutable receipt recording an operator's judgement on ONE submission."
    return {"LeaseId": {"S": f"adjudication#compilation#{execution_id}#{model_type}"
                             f"#{submission_id}"}}
def _submission_was_adjudicated(execution_id: str, model_type: str, submission_id: str,
                                job_name: str, intent_key: str,
                                intent_digest: str) -> tuple:
    "(adjudicated, readable) for one submission's operator receipt."
    try:
        item = dynamodb.get_item(
            TableName=TABLE_NAME,
            Key=_adjudication_key(execution_id, model_type, submission_id),
            ConsistentRead=True,
        ).get("Item")
    except _TRANSPORT_ERRORS as exc:
        logger.warning("Cannot read the adjudication receipt for submission %s: %s",
                       submission_id, exc)
        return False, False
    if not item:
        return False, True
    if item.get("Decision", {}).get("S") != ADJUDICATION_NEVER_CREATED:
        return False, True

    # The receipt must name THIS submission. There is no alias space any more: every identity is
    # a UUID written by the current writer, so this is direct equality.
    expected = {
        "ModelType": model_type,
        "JobName": job_name,
        "IntentKey": intent_key,
        "IntentDigest": intent_digest,
    }
    recorded_id = item.get("SubmissionId", {}).get("S")
    if recorded_id != submission_id:
        logger.warning(
            "Adjudication receipt names submission %r; expected %r. Ignoring the receipt and "
            "retaining.", recorded_id, submission_id,
        )
        return False, True
    for field, want in expected.items():
        got = item.get(field, {}).get("S")
        if got != want:
            logger.warning(
                "Adjudication receipt for submission %s does not bind to the record just "
                "read: %s is %r, expected %r. Ignoring the receipt and RETAINING.",
                submission_id, field, got, want,
            )
            return False, True
    return True, True


def _recorded_submissions(execution_id: str, model_type: str) -> tuple:
    '(state, intents) for one branch.'
    if not (RUNS_BUCKET and RUNS_PREFIX):
        # Previously this reported "nothing recorded", which BYPASSED marker protection
        # entirely for any deployment missing the configuration. The deployed stack always
        # supplies both, so an absent value means something is wrong -- retain and say so.
        logger.warning(
            "RUNS_BUCKET/RUNS_PREFIX are not configured, so compilation submissions cannot "
            "be discovered at all; RETAINING rather than assuming none exist",
        )
        return "unreadable", []

    prefix = neo_jobs.submissions_prefix(RUNS_PREFIX, execution_id, model_type)
    intents = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=RUNS_BUCKET, Prefix=prefix):
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if not key.endswith(".json"):
                    continue
                body = s3.get_object(Bucket=RUNS_BUCKET, Key=key)["Body"].read()
                # The record must agree with WHERE it was found. A body claiming another
                # execution, branch or submission id is a mismatch, not a record to act on.
                expected_id = neo_jobs.submission_id_from_key(key)
                parsed = neo_jobs.parse_submission_record(
                    json.loads(body), execution_id=execution_id, model_type=model_type,
                    submission_id=expected_id,
                )
                if parsed is None:
                    logger.warning(
                        "Submission record s3://%s/%s is unusable -- unrecognised schema, "
                        "missing field, or identity disagreeing with its key; RETAINING",
                        RUNS_BUCKET, key,
                    )
                    return "unreadable", intents
                job_name, submission_id = parsed
                if any(existing[1] == submission_id for existing in intents):
                    logger.warning(
                        "Two records claim submission id %r under s3://%s/%s; RETAINING",
                        submission_id, RUNS_BUCKET, prefix,
                    )
                    return "unreadable", intents
                intents.append((
                    job_name, submission_id, key, hashlib.sha256(body).hexdigest(),
                ))
    except _TRANSPORT_ERRORS as exc:
        # Normalized into a retained result rather than propagating: an exception here used
        # to abort the reconciliation of every LATER owner in the same sweep.
        logger.warning("Cannot list submission intents under s3://%s/%s: %s",
                       RUNS_BUCKET, prefix, exc)
        return "unreadable", intents

    retired = _retired_records_exist(execution_id, model_type)
    if retired is None:
        logger.warning(
            "Cannot establish whether execution %s branch %s has retired-format records, so "
            "whether it submitted anything is UNKNOWN; RETAINING", execution_id, model_type,
        )
        return "unreadable", intents
    if retired:
        logger.warning(
            "Execution %s branch %s has records only in the retired format, which this build "
            "does not read. RETAINING: what it submitted cannot be enumerated.",
            execution_id, model_type,
        )
        return "retired_format", intents
    return "ok", intents


# The retired format's object name. Kept ONLY so its existence can be detected; it is never
# opened, parsed, or trusted.
RETIRED_RECORD_FILENAME = "submission-intent.json"


def _retired_records_exist(execution_id: str, model_type: str):
    'True if this branch has a retired-format record, None if that cannot be established.'
    key = (f"{RUNS_PREFIX.strip('/')}/{execution_id}/neo/{model_type}/"
           f"{RETIRED_RECORD_FILENAME}")
    # A LISTING on the exact key, not a head. HeadObject needs s3:GetObject; the lease role has
    # prefix-scoped s3:ListBucket over the runs prefix, and this object is under it -- so this
    # needs no new permission, and it never opens the object.
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=RUNS_BUCKET, Prefix=key):
            for item in page.get("Contents", []) or []:
                if item.get("Key") == key:
                    return True
        return False
    except _TRANSPORT_ERRORS as exc:
        logger.warning("Cannot establish whether %s exists: %s", key, exc)
        return None


def _active_compilation_children(execution_id: str):
    "(active, unresolved) for this execution's compilation jobs, per branch."
    active, unresolved = [], False
    # One Describe per DISTINCT job across the whole guard, shared with the classifier below.
    describe_cache = {}
    for model_type in neo_jobs.MODEL_TYPES:
        observed = {}          # job name -> status, for every candidate that answered
        absent = set()         # candidates the service says do not exist
        for attempt in range(1, neo_jobs.MAX_ATTEMPTS + 1):
            name = neo_jobs.deterministic_job_name(execution_id, model_type, attempt)
            try:
                status = sagemaker.describe_compilation_job(
                    CompilationJobName=name
                )["CompilationJobStatus"]
                describe_cache[name] = (
                    ("terminal" if neo_jobs.is_terminal(status) else "active"), status)
            except ClientError as exc:
                if neo_jobs.is_absent_job_error(exc.response.get("Error", {})):
                    absent.add(name)
                    describe_cache[name] = ("absent", None)
                    continue
                logger.warning(
                    "Cannot read compilation job %s: %s. Abandoning discovery for this owner; "
                    "it will be retained and retried on the next sweep.", name, exc,
                )
                return active, True
            except _TRANSPORT_ERRORS as exc:
                # A streaming or connection failure. ClientError is handled above, so the
                # recognised-absence classification is preserved and this only catches what
                # used to propagate and abort the whole sweep. Same early return: see above.
                logger.warning(
                    "Transport failure reading compilation job %s: %s. Abandoning discovery "
                    "for this owner; it will be retained and retried.", name, exc,
                )
                return active, True
            observed[name] = status
            if not neo_jobs.is_terminal(status):
                active.append((name, status))

        state, intents = _recorded_submissions(execution_id, model_type)

        if state != "ok":
            logger.warning(
                "Branch %s of execution %s: its submission intents could not be fully "
                "discovered, so whether a compilation was submitted cannot be established; "
                "RETAINING", model_type, execution_id,
            )
            unresolved = True
            continue

        if not intents:
            # Complete discovery found no intent, so nothing was submitted for this branch.
            # Any job that DID answer is already accounted for -- terminal ones settled,
            # non-terminal ones in `active`.
            continue

        for intent in intents:
            entry = classify_submission(execution_id, model_type, intent,
                                        cache=describe_cache)
            state = entry["state"]
            if state in SETTLED_STATES:
                continue
            if state == "active":
                # Already captured by the enumeration when the name is a candidate; append it
                # when it is not, so an out-of-range live job still holds the lease.
                name = entry["job_name"]
                if not any(existing[0] == name for existing in active):
                    active.append((name, entry["job_status"]))
                continue

            logger.warning(
                "Branch %s of execution %s: submission %s of %s is %s; RETAINING",
                model_type, execution_id, entry["submission_id"], entry["job_name"], state,
            )
            unresolved = True

    return active, unresolved


def _children_block_release(execution_name: str, owner_item: dict,
                            correlation=None):
    'Reason to retain this lease because a compilation child is still live, or None.'
    execution_id = _sagemaker_execution_id(owner_item)
    if execution_id is None:
        # No ARN recorded. If a start was never attempted there is nothing to check.
        if not owner_item.get("ClientToken", {}).get("S"):
            return None
        if correlation is not None:
            arn, status = correlation
            if arn:
                logger.warning(
                    "Lease %s correlates to execution %s (status %s) via display name; "
                    "its compilation children are named from that", execution_name, arn,
                    status,
                )
        else:
            display_name = owner_item.get("DisplayName", {}).get("S")
            try:
                arn, _ = _find_execution_by_display_name(display_name)
            except ClientError as exc:
                logger.warning("Cannot correlate %s to recover its execution id: %s",
                               execution_name, exc)
                # A FAILED lookup is not an unnameable child -- it is an unanswered
                # question, and it must not share a reason that adjudication is allowed to
                # override. Conflating the two let an AccessDenied here end in a release.
                return "compilation_children_correlation_failed"
        if not arn:
            logger.warning(
                "Lease %s recorded a start but no execution ARN, and no execution "
                "carries display name %r, so its compilation children cannot be "
                "named; RETAINING", execution_name,
                owner_item.get("DisplayName", {}).get("S"),
            )
            return "compilation_children_unnameable"
        execution_id = arn.rsplit("/", 1)[-1]

    active, unresolved = _active_compilation_children(execution_id)
    if active:
        logger.warning(
            "Lease %s has %d compilation job(s) still live (%s); RETAINING",
            execution_name, len(active),
            ", ".join(f"{n}={s}" for n, s in active[:4]),
        )
        return "compilation_children_active"
    if unresolved:
        return "compilation_children_unreadable"
    return None


def _retained(execution_name: str, owner_item: dict, reason: str, **extra) -> dict:
    'A retained release result, MARKED for prompt reconciliation.'
    if not owner_item.get("OrchestrationEnded", {}).get("N"):
        try:
            _mark_orchestration_ended(execution_name)
        except _TRANSPORT_ERRORS as exc:
            logger.warning("Could not mark %s for eager recheck: %s", execution_name, exc)
    return {"released": False, "retained": True, "reason": reason, **extra}


def release(execution_name: str, require_no_start_intent: bool = False) -> dict:
    'Free the slot held by *execution_name*.'
    existing = _read_owner(execution_name)
    if not existing:
        logger.info("No lease for %s; already released", execution_name)
        return {"released": True, "already": True}

    if require_no_start_intent and existing.get("ClientToken", {}).get("S"):
        # Committed between the reconciler's scan and now.
        logger.warning(
            "Lease %s recorded a start after the reconcile scan; RETAINING",
            execution_name,
        )
        return _retained(execution_name, existing, "start_committed_concurrently")

    recorded_arn = existing.get("ExecutionArn", {}).get("S")
    correlation = None
    if not recorded_arn and existing.get("ClientToken", {}).get("S"):
        display_name = existing.get("DisplayName", {}).get("S")
        if display_name:
            try:
                recorded_arn, correlated_status = _find_execution_by_display_name(display_name)
            except _TRANSPORT_ERRORS as exc:
                logger.warning("Cannot correlate %s before releasing: %s",
                               execution_name, exc)
                return _retained(execution_name, existing, "correlation_unavailable")
            correlation = (recorded_arn, correlated_status)
            if recorded_arn:
                outcome = _persist_discovered_arn(execution_name, recorded_arn)
                if outcome in ("conflict", "failed"):
                    # The authoritative identity is not the one just verified, or could not be
                    # established. Retain and let the next attempt start from the record.
                    return _retained(execution_name, existing,
                                     "execution_identity_changed")

    if recorded_arn:
        try:
            parent_status = _pipeline_status(recorded_arn)
        except _TRANSPORT_ERRORS as exc:
            logger.warning("Cannot read the parent pipeline status for %s: %s",
                           execution_name, exc)
            return _retained(execution_name, existing, "parent_status_unavailable")
        if parent_status not in TERMINAL_STATUSES:
            logger.warning(
                "Lease %s belongs to execution %s which is %s; RETAINING. Compilation is the "
                "last step, so an absence of children says nothing about a running pipeline.",
                execution_name, recorded_arn, parent_status,
            )
            return _retained(execution_name, existing, "parent_not_terminal",
                             detail=f"The pipeline is {parent_status}.")

    blocked = _children_block_release(execution_name, existing, correlation=correlation)
    if blocked:
        # Mark it for eager rechecking. Without this the age gate defers a
        # child-draining lease for the full staleness window -- about six hours against
        # an hourly sweep -- so the slot sits idle long after the compilation finishes.
        return _retained(execution_name, existing, blocked)

    slot = int(existing["Slot"]["N"])
    owner_delete = {
        "TableName": TABLE_NAME,
        "Key": _owner_key(execution_name),
        "ConditionExpression": "attribute_exists(LeaseId)",
    }
    if require_no_start_intent or not existing.get("ClientToken", {}).get("S"):
        owner_delete["ConditionExpression"] = (
            "attribute_exists(LeaseId) AND attribute_not_exists(#t)"
        )
        owner_delete["ExpressionAttributeNames"] = {"#t": "ClientToken"}
    elif recorded_arn:
        # Bind the delete to the execution whose terminality was actually verified. Re-reading
        # is not enough: the ARN can change between the read and the delete, and then the owner
        # of a running pipeline is removed on the strength of a check against a different one.
        owner_delete["ConditionExpression"] = (
            "attribute_exists(LeaseId) AND #arn = :checked"
        )
        owner_delete["ExpressionAttributeNames"] = {"#arn": ATTR_NAMES["#arn"]}
        owner_delete["ExpressionAttributeValues"] = {":checked": {"S": recorded_arn}}

    try:
        dynamodb.transact_write_items(
            TransactItems=[
                {
                    "Delete": {
                        "TableName": TABLE_NAME,
                        "Key": _slot_key(slot),
                        # Only release a slot THIS execution owns. Without this a
                        # stale retry could free a slot another execution has
                        # since taken.
                        "ConditionExpression": "#o = :owner",
                        "ExpressionAttributeNames": {"#o": "Owner"},
                        "ExpressionAttributeValues": {":owner": {"S": execution_name}},
                    }
                },
                {"Delete": owner_delete},
            ]
        )
        logger.info("Execution %s released slot %d", execution_name, slot)
        return {"released": True, "already": False, "slot": slot}
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "TransactionCanceledException":
            raise
        reasons = [
            (entry or {}).get("Code")
            for entry in (exc.response.get("CancellationReasons") or [])
        ]
        # KEEP the item this check reads. Discarding it and passing the older `current or existing`
        # to _retained() meant the eager-recheck decision was made from a stale snapshot, which is
        # not preserved across the same-name recreation that motivated deleting the classifier.
        latest = _read_owner(execution_name)
        if not latest:
            logger.info(
                "The owner record for %s is gone, so its release completed concurrently "
                "(cancellation reasons: %s)", execution_name,
                ", ".join(r for r in reasons if r) or "unspecified",
            )
            return {"released": True, "already": True}

        logger.warning(
            "Releasing %s was cancelled and its owner record is still present; RETAINING for the "
            "next sweep. Cancellation reasons: %s. If this persists, the runbook's direct-deletion "
            "recovery applies -- reconciliation deliberately has no override.",
            execution_name, ", ".join(r for r in reasons if r) or "unspecified",
        )
        return _retained(execution_name, latest, "release_cancelled")
    except _TRANSPORT_ERRORS as exc:
        logger.warning("Transport failure releasing %s: %s", execution_name, exc)
        return _retained(execution_name, existing, "release_unavailable")


def inspect_compilation_submissions(execution_name: str) -> dict:
    'Read-only inspection of current submission records and retired-format presence.'
    existing = _read_owner(execution_name)
    if not existing:
        return {"lease": execution_name, "held": False, "submissions": []}

    execution_arn = existing.get("ExecutionArn", {}).get("S")
    if not execution_arn:
        return {"lease": execution_name, "held": True, "execution_arn": None,
                "detail": "No pipeline execution is recorded for this lease, so its "
                          "compilation submissions cannot be named. Run reconcile first: it "
                          "correlates by display name and records what it finds.",
                "submissions": []}
    execution_id = execution_arn.rsplit("/", 1)[-1]

    report = {"lease": execution_name, "held": True, "execution_arn": execution_arn,
              "submissions": []}
    # One Describe per DISTINCT job for the whole report, not one per intent.
    describe_cache = {}
    for model_type in neo_jobs.MODEL_TYPES:
        state, intents = _recorded_submissions(execution_id, model_type)
        if state == "retired_format":
            report["submissions"].append({
                "branch": model_type, "state": "retired_format",
                "settles": "BLOCKED -- this branch's records are in the retired format, which "
                           "this build does not read, so what it submitted cannot be enumerated. "
                           "There is nothing to adjudicate. This inspection does NOT establish "
                           "that a release is safe: it checks neither the parent's status nor "
                           "whether anything is still submitting. Follow the retired-format "
                           "recovery in README.md, which carries those preconditions.",
                "detail": "A retired-format record exists for this branch. Its contents are "
                          "deliberately not read or trusted; only its existence is detected, "
                          "and that is enough to know an unenumerable submission happened.",
            })
            continue
        if state != "ok":
            report["submissions"].append({
                "branch": model_type, "state": "discovery_incomplete",
                # Carries `settles` like every other entry. Omitting it made the runbook's
                # instruction to "read each entry's settles field" undefined for this shape, and
                # a caller reading it with .get() saw None -- indistinguishable from settled.
                "settles": "BLOCKED -- submission records could not be fully discovered, so "
                           "what this branch submitted is unknown and no receipt can settle it. "
                           "Fix the read first.",
                "detail": "Submission records could not be fully discovered, so this branch "
                          "retains. Fix the read before adjudicating anything.",
            })
            continue
        for intent in intents:
            # The SAME classifier reconciliation uses, so the report cannot promise a
            # settlement reconciliation would refuse.
            entry = classify_submission(execution_id, model_type, intent, cache=describe_cache)
            entry["settles"] = {
                "settled_terminal": "yes -- its job is terminal",
                "active": "not yet -- its job is still running",
                "settled_adjudicated": "yes -- adjudicated",
                "needs_adjudication": "NO -- needs adjudicate_compilation_submission",
                "unreadable": "BLOCKED -- its job status could not be read, so absence is "
                              "unconfirmed and no receipt can settle it. Fix the read first.",
                "unsupported_name": "BLOCKED -- outside the enumerated candidates, so no "
                                    "receipt can be evaluated. Needs a code change.",
            }[entry["state"]]
            report["submissions"].append(entry)
    return report


def adjudicate_compilation_submission(execution_name: str, model_type: str,
                                      submission_id: str, operator: str,
                                      note: str, evidence: str,
                                      expected_job_name: str = None,
                                      expected_digest: str = None) -> dict:
    "Record an operator's judgement that ONE recorded submission never created a job."
    if model_type not in neo_jobs.MODEL_TYPES:
        return {"adjudicated": False, "reason": "unknown_branch",
                "detail": f"{model_type!r} is not one of {sorted(neo_jobs.MODEL_TYPES)}"}
    for field, value in (("operator", operator), ("note", note), ("evidence", evidence)):
        if not (value or "").strip():
            return {"adjudicated": False, "reason": f"missing_{field}",
                    "detail": f"{field} is required: this judgement is attributable, and "
                              f"the evidence must cover every create attempt the SDK made "
                              f"for this submission, not just the final exception"}

    existing = _read_owner(execution_name)
    if not existing:
        return {"adjudicated": False, "reason": "no_lease",
                "detail": f"No lease is held for {execution_name}; nothing to adjudicate."}

    execution_arn = existing.get("ExecutionArn", {}).get("S")
    if not execution_arn:
        return {"adjudicated": False, "reason": "no_execution_recorded",
                "detail": "This lease records no pipeline execution, so a compilation "
                          "submission within it cannot be identified. If you believe no "
                          "pipeline started at all, use the start adjudication instead."}
    execution_id = execution_arn.rsplit("/", 1)[-1]

    # 1. The intent must be readable, well formed, and the one named.
    if not (RUNS_BUCKET and RUNS_PREFIX):
        return {"adjudicated": False, "reason": "unconfigured",
                "detail": "RUNS_BUCKET/RUNS_PREFIX are not configured, so the submission "
                          "record cannot be read."}

    # A submission recovered from the retired marker format lives in a DIFFERENT object, so
    key = neo_jobs.submission_key(RUNS_PREFIX, execution_id, model_type, submission_id)
    try:
        response = s3.get_object(Bucket=RUNS_BUCKET, Key=key)
        body = response["Body"].read()
        version_id = response.get("VersionId")
    except _TRANSPORT_ERRORS as exc:
        # A MISSING record is not permission to manufacture a resolution.
        return {"adjudicated": False, "reason": "intent_unreadable",
                "detail": f"Cannot read the submission record at s3://{RUNS_BUCKET}/{key}: "
                          f"{exc}. A record that cannot be read cannot be adjudicated."}
    body_digest = hashlib.sha256(body).hexdigest()

    try:
        payload = json.loads(body)
    except ValueError as exc:
        return {"adjudicated": False, "reason": "intent_malformed",
                "detail": f"The submission record is not usable JSON: {exc}"}
    parsed = neo_jobs.parse_submission_record(
        payload, execution_id=execution_id, model_type=model_type,
        submission_id=submission_id,
    )
    if parsed is None:
        return {"adjudicated": False, "reason": "intent_unsupported",
                "detail": "The submission record has an unrecognised schema version, a "
                          "missing field, or an identity disagreeing with its location; "
                          "this operation will not guess at its meaning."}
    job_name, _ = parsed
    if expected_job_name and expected_job_name != job_name:
        return {"adjudicated": False, "reason": "job_name_mismatch",
                "detail": f"The record names {job_name}, not {expected_job_name}."}
    if expected_digest and expected_digest != body_digest:
        return {"adjudicated": False, "reason": "digest_mismatch",
                "detail": f"The record's body digest is {body_digest}, not {expected_digest}. "
                          f"You are not looking at the object you inspected."}

    # An adjudication that reconciliation cannot act on is worse than a refusal: it reports
    # success and the lease stays held forever.
    if job_name not in neo_jobs.candidate_job_names(execution_id):
        return {"adjudicated": False, "reason": "unsupported_job_name",
                "detail": f"{job_name} is not among the {neo_jobs.MAX_ATTEMPTS} candidate "
                          f"names reconciliation enumerates for this execution, so a receipt "
                          f"for it could never be evaluated. Resolving this needs a code "
                          f"change, not an adjudication."}

    # 2. The parent must be terminal. This is NOT full producer quiescence: the compilation
    # driver's Processing jobs are not enumerated, and an operator can redrive terminal work
    # afterwards -- which is why the runbook forbids redriving an adjudicated execution.
    try:
        parent_status = _pipeline_status(execution_arn)
    except _TRANSPORT_ERRORS as exc:
        return {"adjudicated": False, "reason": "parent_status_unavailable",
                "detail": f"Cannot read the parent pipeline's status: {exc}"}
    if parent_status not in TERMINAL_STATUSES:
        return {"adjudicated": False, "reason": "parent_not_terminal",
                "detail": f"The pipeline is {parent_status}. It can still submit another "
                          f"compilation, so no judgement about this one is safe yet."}

    # 3. A FRESH describe of the exact job. Absence is necessary and not sufficient.
    try:
        status = sagemaker.describe_compilation_job(
            CompilationJobName=job_name
        )["CompilationJobStatus"]
    except ClientError as exc:
        if not neo_jobs.is_absent_job_error(exc.response.get("Error", {})):
            return {"adjudicated": False, "reason": "job_status_unavailable",
                    "detail": f"Describing {job_name} failed with something other than a "
                              f"recognised absence: {exc}. Retaining."}
    else:
        if neo_jobs.is_terminal(status):
            return {"adjudicated": False, "reason": "job_exists_terminal",
                    "detail": f"{job_name} exists and is {status}. Ordinary reconciliation "
                              f"resolves this; no adjudication is needed."}
        return {"adjudicated": False, "reason": "job_exists_active",
                "detail": f"{job_name} exists and is {status}. It was created after all."}

    return _write_adjudication_receipt(
        execution_name, execution_id, model_type, submission_id, job_name, key,
        body_digest, version_id, operator, note, evidence, parent_status, execution_arn,
    )


def _describe_compilation_status(job_name: str, cache: dict = None):
    '("terminal"|"active"|"absent"|"unreadable", status) for one job, cached per invocation.'
    if cache is not None and job_name in cache:
        return cache[job_name]
    try:
        status = sagemaker.describe_compilation_job(
            CompilationJobName=job_name
        )["CompilationJobStatus"]
    except ClientError as exc:
        result = (("absent", None) if neo_jobs.is_absent_job_error(
            exc.response.get("Error", {})) else ("unreadable", None))
    except _TRANSPORT_ERRORS:
        result = ("unreadable", None)
    else:
        result = (("terminal" if neo_jobs.is_terminal(status) else "active"), status)
    if cache is not None:
        cache[job_name] = result
    return result


def classify_submission(execution_id: str, model_type: str, intent, cache: dict = None) -> dict:
    'How ONE recorded submission settles. The single authority for that question.'
    job_name, submission_id, intent_key, intent_digest = intent
    entry = {
        "branch": model_type,
        "submission_id": submission_id,
        "job_name": job_name,
        "intent_key": intent_key,
        "intent_digest": intent_digest,
    }

    def _receipt():
        adjudicated, readable = _submission_was_adjudicated(
            execution_id, model_type, submission_id, job_name, intent_key, intent_digest)
        entry["adjudicated"] = adjudicated
        entry["receipt_readable"] = readable
        return adjudicated

    status_kind, status = _describe_compilation_status(job_name, cache)
    entry["job_status"] = status or status_kind
    if status_kind == "terminal":
        entry["state"] = "settled_terminal"
        return entry
    if status_kind == "active":
        entry["state"] = "active"
        return entry
    if status_kind == "unreadable":
        # A receipt is reported if present, but it does NOT settle: absence was never
        # confirmed, so the job may exist.
        _receipt()
        entry["state"] = "unreadable"
        return entry

    if job_name not in neo_jobs.candidate_job_names(execution_id):
        _receipt()
        entry["state"] = "unsupported_name"
        return entry
    entry["state"] = "settled_adjudicated" if _receipt() else "needs_adjudication"
    return entry


SETTLED_STATES = frozenset({"settled_terminal", "settled_adjudicated"})
def _write_adjudication_receipt(execution_name, execution_id, model_type, submission_id,
                                job_name, key, body_digest, version_id, operator, note,
                                evidence, parent_status, execution_arn) -> dict:
    'Persist the immutable receipt, conditionally, and report what happened.'
    # 4. Persist the receipt conditionally, under its own key so releasing the owner cannot
    #    delete the evidence.
    now = int(time.time())
    receipt_key = _adjudication_key(execution_id, model_type, submission_id)
    item = {
        **receipt_key,
        "Decision": {"S": ADJUDICATION_NEVER_CREATED},
        "Operator": {"S": operator},
        "Note": {"S": note},
        "Evidence": {"S": evidence},
        "AdjudicatedAt": {"N": str(now)},
        "ExecutionArn": {"S": execution_arn},
        "LeaseName": {"S": execution_name},
        "ModelType": {"S": model_type},
        "JobName": {"S": job_name},
        "SubmissionId": {"S": submission_id},
        "IntentKey": {"S": key},
        "IntentDigest": {"S": body_digest},
        "ParentStatus": {"S": parent_status},
    }
    if version_id:
        item["IntentVersionId"] = {"S": version_id}
    try:
        dynamodb.put_item(
            TableName=TABLE_NAME,
            Item=item,
            ConditionExpression="attribute_not_exists(LeaseId)",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        # A receipt already exists. An identical re-run is idempotent; a DIFFERENT judgement
        # is refused rather than overwritten -- the first decision stands and its evidence
        # must not be replaced.
        prior = dynamodb.get_item(TableName=TABLE_NAME, Key=receipt_key,
                                  ConsistentRead=True).get("Item", {})
        if prior.get("Decision", {}).get("S") != ADJUDICATION_NEVER_CREATED:
            return {"adjudicated": False, "reason": "conflicting_receipt",
                    "detail": f"A different judgement is already recorded for submission "
                              f"{submission_id}: {prior.get('Decision', {}).get('S')!r}"}
        differing = [
            field for field, want in (
                ("ModelType", model_type), ("SubmissionId", submission_id),
                ("JobName", job_name), ("IntentKey", key), ("IntentDigest", body_digest),
            )
            if prior.get(field, {}).get("S") != want
        ]
        if differing:
            return {"adjudicated": False, "reason": "stale_receipt",
                    "detail": f"A receipt already exists for submission {submission_id} but "
                              f"does not describe the object just read ({', '.join(differing)} "
                              f"differ), so reconciliation will keep rejecting it. It cannot "
                              f"be overwritten. Restore the reviewed object version, or "
                              f"re-list to obtain the current identity and adjudicate that."}
        logger.warning("Adjudication receipt for submission %s already exists; returning it",
                       submission_id)
        return {"adjudicated": True, "already": True, "submission_id": submission_id,
                "job_name": job_name,
                "operator": prior.get("Operator", {}).get("S")}

    logger.warning(
        "ADJUDICATED COMPILATION SUBMISSION lease=%s execution=%s branch=%s submission=%s "
        "job=%s operator=%s note=%s evidence=%s parent=%s at=%d: recorded that no "
        "compilation job was created. The lease is NOT released here -- run reconcile.",
        execution_name, execution_arn, model_type, submission_id, job_name, operator, note,
        evidence, parent_status, now,
    )
    return {
        "adjudicated": True,
        "already": False,
        "submission_id": submission_id,
        "job_name": job_name,
        "operator": operator,
        "released": False,
        "next": "Run the reconcile action; it re-checks the parent and every compilation "
                "child before releasing anything.",
    }


def _pipeline_status(execution_arn: str) -> str:
    return sagemaker.describe_pipeline_execution(
        PipelineExecutionArn=execution_arn
    )["PipelineExecutionStatus"]


def _find_execution_by_display_name(display_name: str):
    'Resolve an ambiguous start by matching the deterministic DISPLAY NAME.'
    if not display_name:
        return None, None
    paginator = sagemaker.get_paginator("list_pipeline_executions")
    for page in paginator.paginate(PipelineName=PIPELINE_NAME):
        for summary in page["PipelineExecutionSummaries"]:
            if summary.get("PipelineExecutionDisplayName") == display_name:
                return (
                    summary["PipelineExecutionArn"],
                    summary.get("PipelineExecutionStatus"),
                )
    return None, None


def record_start_rejected(execution_name: str, error: str = None) -> dict:
    'Record that SageMaker DEFINITIVELY refused to start this pipeline.'
    if not _read_owner(execution_name):
        return {"recorded": False, "reason": "no_lease"}

    try:
        dynamodb.update_item(
            TableName=TABLE_NAME,
            Key=_owner_key(execution_name),
            UpdateExpression="SET #r = :now, #e = :err",
            ExpressionAttributeNames={"#r": "StartRejected", "#e": "StartError",
                                      "#a": "AmbiguousAttempt"},
            ExpressionAttributeValues={
                ":now": {"N": str(int(time.time()))},
                ":err": {"S": str(error or "unspecified")[:512]},
            },
            ConditionExpression=(
                "attribute_exists(LeaseId) AND attribute_not_exists(#a)"
            ),
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        logger.warning(
            "Not marking %s as rejected: an attempt ended ambiguously, so a "
            "pipeline may already be running for it", execution_name,
        )
        return {"recorded": False, "reason": "ambiguity_preserved"}
    logger.info("Recorded definitive start rejection for %s: %s",
                execution_name, error)
    return {"recorded": True}


def record_ambiguous_attempt(execution_name: str) -> dict:
    'Record that an attempt for this lease ended with an UNKNOWN outcome.'
    existing = _read_owner(execution_name)
    if not existing:
        return {"recorded": False, "reason": "no_lease"}
    dynamodb.update_item(
        TableName=TABLE_NAME,
        Key=_owner_key(execution_name),
        UpdateExpression="SET #a = :now REMOVE #r",
        ExpressionAttributeNames={"#a": "AmbiguousAttempt", "#r": "StartRejected"},
        ExpressionAttributeValues={":now": {"N": str(int(time.time()))}},
        ConditionExpression="attribute_exists(LeaseId)",
    )
    logger.info("Recorded ambiguous start attempt for %s", execution_name)
    return {"recorded": True}


def _mark_orchestration_ended(execution_name: str) -> None:
    "Record that this lease's orchestration ended, for eager rechecking."
    try:
        dynamodb.update_item(
            TableName=TABLE_NAME,
            Key=_owner_key(execution_name),
            UpdateExpression="SET #e = :now",
            ExpressionAttributeNames={"#e": "OrchestrationEnded"},
            ExpressionAttributeValues={":now": {"N": str(int(time.time()))}},
            ConditionExpression="attribute_exists(LeaseId)",
        )
        logger.info("Marked %s for eager reconciliation", execution_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        # The lease was resolved concurrently; nothing to mark.


def _guard_owner(execution_name: str, what: str, operation, *args, **kwargs):
    'Run ONE per-owner operation inside the transport boundary.'
    try:
        return True, operation(*args, **kwargs)
    except _TRANSPORT_ERRORS as exc:
        logger.warning(
            "Transport failure during %s for %s; retaining this owner and continuing the "
            "sweep: %s", what, execution_name, exc,
        )
        return False, None


def _release_one(execution_name: str, **kwargs) -> dict:
    'release() for ONE owner inside a transport boundary.'
    try:
        return release(execution_name, **kwargs)
    except _TRANSPORT_ERRORS as exc:
        logger.warning(
            "Transport failure releasing %s; retaining this owner and continuing the sweep: %s",
            execution_name, exc,
        )
        return {"released": False, "retained": True, "reason": "release_unavailable"}


def reconcile(target_names=()) -> dict:
    'Resolve stale leases against real pipeline status.'
    target_names = set(target_names or ())
    now = int(time.time())

    scanned = []
    seen_ids = set()
    unreadable_targets = []
    for name in sorted(target_names):
        # Guarded per target: one unreadable owner used to raise out of the whole sweep, so a
        # single unreachable read cost every other owner its turn.
        ok, item = _guard_owner(name, "targeted owner read", _read_owner, name)
        if not ok:
            unreadable_targets.append(name)
            continue
        if item:
            scanned.append(item)
            seen_ids.add(item["LeaseId"]["S"])
        else:
            logger.info("Targeted execution %s has no lease record", name)

    # Paginate. DynamoDB applies its 1 MB Scan limit BEFORE the filter, so a
    # single response can contain no owner records while more pages remain --
    # reading only the first page would silently skip the only releasable lease.
    start_key = None
    while True:
        kwargs = {
            "TableName": TABLE_NAME,
            "FilterExpression": "begins_with(LeaseId, :p)",
            "ExpressionAttributeValues": {":p": {"S": OWNER_PREFIX}},
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        page = dynamodb.scan(**kwargs)
        for item in page.get("Items", []):
            if item["LeaseId"]["S"] in seen_ids:
                continue  # already fetched consistently above
            seen_ids.add(item["LeaseId"]["S"])
            scanned.append(item)
        start_key = page.get("LastEvaluatedKey")
        if not start_key:
            break

    checked, released, retained = 0, [], []
    provisional = {name: "owner_read_unavailable" for name in unreadable_targets}
    for item in scanned:
        execution_name = item["ExecutionName"]["S"]
        age = now - int(item["AcquiredAt"]["N"])

        targeted = execution_name in target_names
        orchestration_ended = bool(item.get("OrchestrationEnded", {}).get("N"))
        if not targeted and not orchestration_ended and age < STALE_LEASE_SECONDS:
            continue
        checked += 1

        arn = item.get("ExecutionArn", {}).get("S")
        display_name = item.get("DisplayName", {}).get("S")
        has_start_intent = bool(item.get("ClientToken", {}).get("S"))

        if not has_start_intent:
            outcome = _release_one(execution_name, require_no_start_intent=True)
            if outcome.get("released"):
                released.append((execution_name, "no_start_recorded"))
            else:
                retained.append(
                    (execution_name, outcome.get("reason", "start_committed_concurrently"))
                )
            continue

        if targeted and not orchestration_ended:
            # Guarded: an unguarded write here raised out of the sweep and left every other
            # owner held. A failed mark only costs this owner a slower recheck.
            ok, _ = _guard_owner(execution_name, "eager-recheck marking",
                                 _mark_orchestration_ended, execution_name)
            if not ok:
                retained.append((execution_name, "marking_unavailable"))
                continue
            orchestration_ended = True

        status = None
        if arn:
            try:
                status = _pipeline_status(arn)
            except _TRANSPORT_ERRORS as exc:
                # Widened past ClientError: a connection or streaming failure here used to
                # propagate and abort the sweep, so one unreachable owner cost every owner
                # after it.
                logger.warning("Cannot read %s: %s", arn, exc)
                retained.append((execution_name, "status_unreadable"))
                continue
        else:
            definitively_rejected = (
                bool(item.get("StartRejected", {}).get("N"))
                and not item.get("AmbiguousAttempt", {}).get("N")
            )

            arn, status = None, None
            try:
                arn, status = _find_execution_by_display_name(display_name)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in ("ResourceNotFound", "ResourceNotFoundException"):
                    logger.info(
                        "Pipeline does not exist, so nothing can have started for "
                        "%s; treating the lookup as a conclusive miss",
                        execution_name,
                    )
                else:
                    # Per-owner, so one unreadable lease cannot stop the sweep.
                    logger.warning(
                        "Cannot look up display name %r for %s: %s",
                        display_name, execution_name, exc,
                    )
                    retained.append((execution_name, "lookup_failed"))
                    continue
            except _TRANSPORT_ERRORS as exc:
                # A transport failure is an unanswered question, never the conclusive miss
                # that the ResourceNotFound arm above represents. Retained per owner rather
                # than propagating, which used to abort the sweep.
                logger.warning("Transport failure correlating %s: %s", execution_name, exc)
                retained.append((execution_name, "lookup_failed"))
                continue
            if arn is None and definitively_rejected:
                logger.info(
                    "Lease %s was definitively rejected (%s) and no execution "
                    "matches display name %r; releasing the slot",
                    execution_name,
                    item.get("StartError", {}).get("S", "unspecified"),
                    display_name,
                )
                ok, outcome = _guard_owner(
                    execution_name, "rejected-start release", release_rejected,
                    execution_name,
                )
                if not ok:
                    retained.append((execution_name, "release_unavailable"))
                    continue
                if outcome.get("released"):
                    released.append((execution_name, "start_rejected"))
                else:
                    retained.append(
                        (execution_name,
                         outcome.get("reason", "rejection_release_failed"))
                    )
                continue
            if arn is None:
                logger.warning(
                    "Lease %s recorded a start but no execution matches display "
                    "name %r; RETAINING the slot for operator review",
                    execution_name, display_name,
                )
                retained.append((execution_name, "start_outcome_unknown"))
                continue

        if status in TERMINAL_STATUSES:
            outcome = _release_one(execution_name)
            if outcome.get("released"):
                released.append((execution_name, status))
            else:
                retained.append(
                    (execution_name, outcome.get("reason", "release_refused"))
                )
        else:
            # Still running. The eager-recheck marker was already recorded above,
            # before any lookup that could fail.
            retained.append((execution_name, status))

    decided = {name for name, _ in released} | {name for name, _ in retained}
    for name, reason in provisional.items():
        if name not in decided:
            retained.append((name, reason))

    result = {
        "leases_total": len(scanned),
        "stale_checked": checked,
        "released": released,
        "retained": retained,
        "targeted": sorted(target_names),
    }
    logger.info("Reconcile: %s", json.dumps(result))
    return result


def handler(event, context):
    """Dispatch on ``event["action"]``."""
    action = event.get("action")
    if action == "acquire":
        return acquire(event["executionName"], event.get("requestId"))
    if action == "record_start":
        return record_start(
            event["executionName"],
            event["clientToken"],
            event.get("executionArn"),
            event.get("parameters"),
            event.get("displayName"),
        )
    if action == "release":
        return release(event["executionName"])
    if action == "record_start_rejected":
        return record_start_rejected(event["ExecutionName"], event.get("Error"))
    if action == "record_ambiguous_attempt":
        return record_ambiguous_attempt(event["ExecutionName"])
    if action == "reconcile":
        targets = event.get("TargetExecutionNames") or []
        if isinstance(targets, str):
            targets = [targets]
        single = event.get("ExecutionName")
        if single:
            targets.append(single)
        arn = event.get("ExecutionArn") or event.get("executionArn")
        if arn:
            targets.append(str(arn).rsplit(":", 1)[-1])
        return reconcile(target_names=targets)
    if action == "close_admission":
        return close_admission(event.get("Reason"))
    if action == "reopen_admission":
        return reopen_admission()
    if action == "admission_status":
        return admission_status()
    if action == "adjudicate_start_never_occurred":
        return adjudicate_start_never_occurred(
            event["ExecutionName"], event.get("Operator"), event.get("Note"),
            event.get("Evidence"),
        )
    if action == "inspect_compilation_submissions":
        return inspect_compilation_submissions(event["ExecutionName"])
    if action == "adjudicate_compilation_submission":
        return adjudicate_compilation_submission(
            event["ExecutionName"],
            event["ModelType"],
            event["SubmissionId"],
            event.get("Operator"),
            event.get("Note"),
            event.get("Evidence"),
            expected_job_name=event.get("ExpectedJobName"),
            expected_digest=event.get("ExpectedIntentDigest"),
        )
    raise LeaseError(
        f"Unknown action {action!r}; expected acquire, record_start, "
        f"record_start_rejected, record_ambiguous_attempt, release, reconcile, "
        f"close_admission, reopen_admission, admission_status, "
        f"adjudicate_start_never_occurred, adjudicate_compilation_submission or "
        f"inspect_compilation_submissions."
    )
