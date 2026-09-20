# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'SQS -> Step Functions dispatcher.'
import hashlib
import json
import logging
import os
import re

from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]

EXECUTION_NAME_MAX = 80
_SAFE_NAME = re.compile(r"[^A-Za-z0-9-]")
_SAGEMAKER_NAME = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,81}$")

# Camera identity is used in S3 prefixes, job names and registry metadata, so it
# is constrained rather than passed through.
CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

ALLOWED_PARAMETERS = {
    "TrainingInstanceType",
    "ProcessingInstanceType",
    "NumSamples",
    "MinAUROC",
    "MinBalancedAccuracy",
    "MinDefectRecall",
    "CameraId",
    "TrainingDataS3Uri",
    "NeoTargetDevice",
    "NeoTargetOs",
    "NeoTargetArch",
}
MAX_PARAMETER_VALUE_LEN = 1024

class _LazyStepFunctions:
    'Build the client on first call, not at import.'

    _client = None

    def __getattr__(self, item):
        if _LazyStepFunctions._client is None:
            import boto3  # noqa: PLC0415 -- deliberately lazy, see above
            _LazyStepFunctions._client = boto3.client("stepfunctions")
        return getattr(_LazyStepFunctions._client, item)


sfn = _LazyStepFunctions()


class InvalidRequest(ValueError):
    """The message cannot be dispatched. Raised so SQS retries then DLQs it."""


def execution_name(request_id: str) -> str:
    'A stable, valid execution name for *request_id*.'
    digest = hashlib.sha256(request_id.encode()).hexdigest()[:12]
    safe = _SAFE_NAME.sub("-", request_id).strip("-")
    prefix = safe[: EXECUTION_NAME_MAX - len(digest) - 1].strip("-")
    # The prefix must begin alphanumeric. Sanitising can leave a leading hyphen
    # (stripped above) or empty the string entirely, e.g. a punctuation-only id.
    if not prefix or not prefix[0].isalnum():
        prefix = f"req{prefix}" if prefix else "req"
        prefix = prefix[: EXECUTION_NAME_MAX - len(digest) - 1].strip("-")
    name = f"{prefix}-{digest}"  # hex digest, so it always ends alphanumeric

    # Fail loudly rather than let SageMaker reject the start AFTER intent has been
    # recorded, which is what strands the lease.
    if not _SAGEMAKER_NAME.match(name) or len(name) > EXECUTION_NAME_MAX:
        raise InvalidRequest(
            f"Request id {request_id!r} produced execution name {name!r}, which "
            f"does not satisfy the SageMaker pipeline execution name pattern "
            f"{_SAGEMAKER_NAME.pattern}. Use only letters, digits and hyphens."
        )
    return name


NEO_TARGET_OS = ("ANDROID", "LINUX")
NEO_TARGET_ARCH = ("X86_64", "X86", "ARM64", "ARM_EABI", "ARM_EABIHF")
NEO_TARGET_DEVICES = (
    "lambda", "ml_m4", "ml_m5", "ml_m6g", "ml_c4", "ml_c5", "ml_c6g", "ml_p2",
    "ml_p3", "ml_g4dn", "ml_inf1", "ml_inf2", "ml_trn1", "ml_eia2", "jetson_tx1",
    "jetson_tx2", "jetson_nano", "jetson_xavier", "rasp3b", "rasp4b", "imx8qm",
    "deeplens", "rk3399", "rk3288", "aisage", "sbe_c", "qcs605", "qcs603",
    "sitara_am57x", "amba_cv2", "amba_cv22", "amba_cv25", "x86_win32", "x86_win64",
    "coreml", "jacinto_tda4vm", "imx8mplus",
)

_RATE = (0.0, 1.0)
PARAMETER_SPECS = {
    "NumSamples": {"type": int, "bounds": (1, 1_000_000)},
    "MinAUROC": {"type": float, "bounds": _RATE},
    "MinBalancedAccuracy": {"type": float, "bounds": _RATE},
    "MinDefectRecall": {"type": float, "bounds": _RATE},
    "TrainingInstanceType": {"type": str, "pattern": re.compile(r"^ml\.[a-z0-9]+\.[a-z0-9]+$")},
    "ProcessingInstanceType": {"type": str, "pattern": re.compile(r"^ml\.[a-z0-9]+\.[a-z0-9]+$")},
    # "none" is the documented sentinel meaning "generate synthetic data instead of
    # reading a published dataset". A pattern-only check rejected it.
    "TrainingDataS3Uri": {
        "type": str,
        "pattern": re.compile(r"^s3://[a-z0-9.\-]{3,63}/.*[^/]$"),
        "sentinels": ("none",),
    },
    "NeoTargetDevice": {
        "type": str,
        "enum": NEO_TARGET_DEVICES,
        "sentinels": ("none",),
    },
    "NeoTargetOs": {"type": str, "enum": NEO_TARGET_OS, "upper": True},
    "NeoTargetArch": {"type": str, "enum": NEO_TARGET_ARCH, "upper": True},
}


def _validate_parameter(name: str, value):
    'Return the canonical string for *value*, or raise InvalidRequest.'
    spec = PARAMETER_SPECS.get(name)
    if spec is None:
        return str(value)  # allowlisted but unconstrained

    expected = spec["type"]
    if value is None:
        raise InvalidRequest(
            f"Parameter {name} is null. A missing value must be omitted so the "
            f"pipeline default applies, not sent as null -- str(None) would reach "
            f"SageMaker as the literal 'None'."
        )
    if expected is not str and isinstance(value, bool):
        raise InvalidRequest(
            f"Parameter {name} must be {expected.__name__}, got a boolean. "
            f"bool is a subclass of int, so True would pass as 1."
        )

    if expected is int:
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise InvalidRequest(
                    f"Parameter {name} must be an integer, got {value!r}."
                ) from None
        elif not isinstance(value, int):
            raise InvalidRequest(
                f"Parameter {name} must be an integer, got "
                f"{type(value).__name__} {value!r}."
            )
    elif expected is float:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise InvalidRequest(
                    f"Parameter {name} must be a number, got {value!r}."
                ) from None
        elif not isinstance(value, (int, float)):
            raise InvalidRequest(
                f"Parameter {name} must be a number, got "
                f"{type(value).__name__} {value!r}."
            )
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidRequest(
                f"Parameter {name} is {value!r}, which is not finite. A quality "
                f"threshold that cannot be ordered cannot fail closed."
            )
    else:
        if not isinstance(value, str):
            raise InvalidRequest(
                f"Parameter {name} must be a string, got {type(value).__name__}."
            )
        if value in spec.get("sentinels", ()):
            return value
        if "enum" in spec:
            # Normalise case for the uppercase enums, so a reader who types "linux"
            # gets the value SageMaker actually accepts rather than a rejection
            # downstream. Nothing repairs it later in the pipeline.
            candidate = value.upper() if spec.get("upper") else value
            if candidate not in spec["enum"]:
                allowed = ", ".join(spec["enum"])
                extra = spec.get("sentinels", ())
                if extra:
                    allowed += " (or " + ", ".join(repr(x) for x in extra) + ")"
                raise InvalidRequest(
                    f"Parameter {name}={value!r} is not a supported value. "
                    f"Allowed: {allowed}."
                )
            return candidate
        if not spec["pattern"].match(value):
            raise InvalidRequest(
                f"Parameter {name}={value!r} does not match "
                f"{spec['pattern'].pattern}."
            )
        return value

    low, high = spec["bounds"]
    if not low <= value <= high:
        raise InvalidRequest(
            f"Parameter {name}={value!r} is outside the supported range "
            f"[{low}, {high}]."
        )
    return str(value)


def validate(body: dict) -> dict:
    'Return the validated request, or raise InvalidRequest.'
    if not isinstance(body, dict):
        raise InvalidRequest(f"Message body must be a JSON object, got {type(body).__name__}")

    camera_id = body.get("camera_id", "default")
    if not isinstance(camera_id, str) or not CAMERA_ID_PATTERN.match(camera_id):
        raise InvalidRequest(
            f"camera_id {camera_id!r} is invalid: expected 1-63 characters matching "
            f"{CAMERA_ID_PATTERN.pattern} (it is used in S3 prefixes, job names and "
            f"registry metadata)."
        )

    raw_parameters = body.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise InvalidRequest(
            f"parameters must be a JSON object, got {type(raw_parameters).__name__}"
        )

    unknown = sorted(set(raw_parameters) - ALLOWED_PARAMETERS)
    if unknown:
        raise InvalidRequest(
            f"Unknown pipeline parameter(s): {', '.join(unknown)}. Allowed: "
            f"{', '.join(sorted(ALLOWED_PARAMETERS))}. An unknown name is silently "
            f"ignored by StartPipelineExecution, so a typo would take effect as the "
            f"default with no error reported anywhere."
        )

    # The two places a camera identity can appear must agree. A body naming cam-A
    # whose parameters override CameraId to cam-B trained cam-B and reported cam-A.
    if "CameraId" in raw_parameters:
        override = raw_parameters["CameraId"]
        if str(override) != camera_id:
            raise InvalidRequest(
                f"Contradictory camera identity: camera_id={camera_id!r} but "
                f"parameters.CameraId={override!r}. Set one, or set both to the "
                f"same value -- otherwise the model trains for one camera and the "
                f"notification reports another."
            )

    parameters = [{"Name": "CameraId", "Value": camera_id}]
    for name, value in sorted(raw_parameters.items()):
        if name == "CameraId":
            continue
        if isinstance(value, (dict, list)):
            raise InvalidRequest(
                f"Parameter {name} must be a scalar, got {type(value).__name__}. "
                f"Stringifying a structure would send an unusable value."
            )
        text = _validate_parameter(name, value)
        if len(text) > MAX_PARAMETER_VALUE_LEN:
            raise InvalidRequest(
                f"Parameter {name} is {len(text)} characters, over the "
                f"{MAX_PARAMETER_VALUE_LEN} limit."
            )
        parameters.append({"Name": name, "Value": text})

    return {"camera_id": camera_id, "PipelineParameters": parameters}


def _same_input(name: str, expected: str) -> bool:
    """True if the existing execution *name* was started with *expected* input."""
    arn_prefix = STATE_MACHINE_ARN.replace(":stateMachine:", ":execution:")
    try:
        described = sfn.describe_execution(executionArn=f"{arn_prefix}:{name}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ExecutionDoesNotExist":
            return False
        raise
    try:
        return json.loads(described["input"]) == json.loads(expected)
    except (KeyError, json.JSONDecodeError):
        return False


def dispatch(record: dict) -> dict:
    """Start one state-machine execution for one SQS record."""
    try:
        body = json.loads(record["body"])
    except json.JSONDecodeError as exc:
        raise InvalidRequest(f"Message body is not valid JSON: {exc}") from exc

    validated = validate(body)

    # Prefer an explicit request_id from the producer so a deliberate retrain gets
    # a new identity while a send retry reuses one. Fall back to the SQS messageId,
    # which is stable across redeliveries of the SAME message.
    request_id = body.get("request_id") or record["messageId"]
    if not isinstance(request_id, str) or not request_id:
        raise InvalidRequest(f"request_id must be a non-empty string, got {request_id!r}")

    name = execution_name(request_id)
    payload = json.dumps(
        {"request_id": request_id, **validated}, sort_keys=True
    )

    try:
        response = sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN, name=name, input=payload
        )
        logger.info("Dispatched %s for request %s", name, request_id)
        return {
            "executionArn": response["executionArn"],
            "duplicate": False,
            "created": None,
        }
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ExecutionAlreadyExists":
            raise
        # Already dispatched -- but only if it is the SAME request. A name
        # collision with different input is a genuine conflict, and swallowing it
        # would silently drop a request.
        if _same_input(name, payload):
            logger.info(
                "Request %s already dispatched as %s; treating redelivery as "
                "already handled", request_id, name,
            )
            arn_prefix = STATE_MACHINE_ARN.replace(":stateMachine:", ":execution:")
            return {
                "executionArn": f"{arn_prefix}:{name}",
                "duplicate": True,
                "created": False,
            }
        raise InvalidRequest(
            f"Execution name {name} already exists with DIFFERENT input. Two "
            f"distinct requests derived the same identity from request_id "
            f"{request_id!r}. Supply distinct request_id values."
        ) from exc


def handler(event, context):
    results = []
    for record in event["Records"]:
        results.append(dispatch(record))
    return {"statusCode": 200, "dispatched": results}
