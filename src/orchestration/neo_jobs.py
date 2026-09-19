# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Compilation job identity, shared by the driver that creates jobs and the lease manager that has to find them again.'
import re

# Both branches of the pipeline compile. Enumerated rather than discovered, because the
# reconciler must be able to name every candidate without reading the pipeline
# definition.
MODEL_TYPES = ("anomalib", "classifier")

# The driver walks attempts 1..MAX_ATTEMPTS looking for a usable name, advancing past a
# terminally failed job so an intentional retry is not blocked forever. The reconciler
# must therefore consider the same range: any of them may name a live child.
MAX_ATTEMPTS = 10

# SageMaker's constraint on CompilationJobName.
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$")
MAX_NAME_LENGTH = 63


def deterministic_job_name(execution_id: str, model_type: str,
                          attempt: int = 1) -> str:
    'Compilation job name stable across retries of the same step ATTEMPT.'
    suffix = "" if attempt <= 1 else f"-a{int(attempt)}"
    raw = f"neo-{model_type}-{execution_id}"
    body = re.sub(r"[^a-zA-Z0-9-]", "-", raw)[: MAX_NAME_LENGTH - len(suffix)].rstrip("-")
    return f"{body}{suffix}"


def candidate_job_names(execution_id: str) -> list:
    'Every compilation job name this execution could have created.'
    return [
        deterministic_job_name(execution_id, model_type, attempt)
        for model_type in MODEL_TYPES
        for attempt in range(1, MAX_ATTEMPTS + 1)
    ]


# Statuses that mean a compilation job is finished and can no longer touch the
# deployment's bucket or consume its quota. Anything else -- including STOPPING, and
# including a status this code does not recognise -- means the lease must be retained.
TERMINAL_COMPILATION_STATUSES = frozenset({"COMPLETED", "FAILED", "STOPPED"})

NEO_JOB_PREFIX = "neo-"


# The documented in-flight statuses, listed for readability. Membership is NOT used to
# decide retention: retention is the default for everything not terminal, so a status
# added by a future API version retains rather than releasing.
ACTIVE_COMPILATION_STATUSES = frozenset({"STARTING", "INPROGRESS", "STOPPING"})


def is_terminal(status) -> bool:
    'True only for a status documented as finished.'
    return status in TERMINAL_COMPILATION_STATUSES


_ABSENT_JOB_CODES = frozenset({"ResourceNotFound", "ResourceNotFoundException"})
_ABSENT_JOB_MESSAGE = re.compile(
    r"^No Compilation Job with name .* found$", re.IGNORECASE
)


def is_absent_job_error(error: dict) -> bool:
    'True if this error means the compilation job does not exist.'
    code = (error or {}).get("Code", "")
    if code in _ABSENT_JOB_CODES:
        return True
    if code != "ValidationException":
        return False
    return bool(_ABSENT_JOB_MESSAGE.match((error or {}).get("Message", "").strip()))


SUBMISSIONS_DIRNAME = "submissions"

# Bumped when the record's meaning changes. A reader that does not recognise a version must
# retain the lease rather than guess, so an old lease manager cannot misread a new record.
SUBMISSION_SCHEMA_VERSION = 1


def submissions_prefix(runs_prefix: str, execution_id: str, model_type: str) -> str:
    """S3 prefix holding every submission intent for one branch of one execution."""
    return (
        f"{runs_prefix.strip('/')}/{execution_id}/neo/{model_type}/{SUBMISSIONS_DIRNAME}/"
    )


def submission_key(runs_prefix: str, execution_id: str, model_type: str,
                   submission_id: str) -> str:
    """S3 key of ONE immutable submission intent record."""
    return (
        f"{submissions_prefix(runs_prefix, execution_id, model_type)}{submission_id}.json"
    )


def submission_record(submission_id: str, execution_id: str, model_type: str,
                      job_name: str, request_digest: str, recorded_at: int,
                      producer: str = None) -> dict:
    'The body of an immutable submission intent.'
    record = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "submission_id": submission_id,
        "execution_id": execution_id,
        "model_type": model_type,
        "job_name": job_name,
        "request_digest": request_digest,
        "recorded_at": recorded_at,
    }
    if producer:
        record["producer"] = producer
    return record


def parse_submission_record(payload, execution_id=None, model_type=None,
                            submission_id=None):
    '(job_name, submission_id) from a record, or None if it cannot be trusted.'
    if not isinstance(payload, dict):
        return None
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    if version != SUBMISSION_SCHEMA_VERSION:
        return None

    job_name = payload.get("job_name")
    recorded_id = payload.get("submission_id")
    if not isinstance(job_name, str) or not job_name:
        return None
    if not isinstance(recorded_id, str) or not recorded_id:
        return None

    # Fields the writer always records. Their absence means this is not a record this version
    # wrote, whatever its declared version says.
    digest = payload.get("request_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    if not isinstance(payload.get("recorded_at"), int) or isinstance(
            payload.get("recorded_at"), bool):
        return None

    if execution_id is not None and payload.get("execution_id") != execution_id:
        return None
    if model_type is not None and payload.get("model_type") != model_type:
        return None
    if submission_id is not None and recorded_id != submission_id:
        return None
    return job_name, recorded_id


def submission_id_from_key(key: str):
    """The submission id a record's KEY claims, or None if the key is not one of ours."""
    tail = key.rsplit("/", 1)[-1]
    if not tail.endswith(".json"):
        return None
    stem = tail[: -len(".json")]
    return stem or None
