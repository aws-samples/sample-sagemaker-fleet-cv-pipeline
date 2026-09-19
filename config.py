# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Shared physical names and S3 prefixes for the fleet CV sample.'

import re

# --- SageMaker pipeline ---
PIPELINE_NAME = "fleet-cv-pipeline"

# --- Model Registry ---
ANOMALY_MODEL_GROUP = "fleet-cv-anomaly-model-group"
CLASSIFIER_MODEL_GROUP = "fleet-cv-classifier-model-group"

DEFAULT_MIN_AUROC = 0.55

TRAINING_REPO_NAME = "fleet-cv-custom-training"

# --- Orchestration ---
SEMAPHORE_TABLE_NAME = "fleet-cv-pipeline-semaphore"
TRAINING_QUEUE_NAME = "fleet-cv-training-requests.fifo"
TRAINING_DLQ_NAME = "fleet-cv-training-requests-dlq.fifo"
COMPLETION_TOPIC_NAME = "fleet-cv-pipeline-completions"
STATE_MACHINE_NAME = "fleet-cv-pipeline-orchestrator"
QUEUE_TRIGGER_FUNCTION_NAME = "fleet-cv-queue-trigger"
FAILURE_RULE_NAME = "fleet-cv-orchestrator-failures"

# --- CloudFormation stack names (scripts read outputs from these) ---
PIPELINE_STACK_NAME = "FleetCVPipelineStack"
ORCHESTRATION_STACK_NAME = "FleetCVOrchestrationStack"

# --- Stack output keys ---
OUTPUT_BUCKET_NAME = "CVBucketName"
OUTPUT_EXECUTION_ROLE_ARN = "CVExecutionRoleArn"
OUTPUT_TRAINING_REPO_URI = "CustomTrainingRepoUri"
OUTPUT_PIPELINE_NAME = "CVPipelineName"
OUTPUT_QUEUE_URL = "QueueUrl"
OUTPUT_DLQ_URL = "DLQUrl"
OUTPUT_STATE_MACHINE_ARN = "StateMachineArn"
OUTPUT_TOPIC_ARN = "TopicArn"
# Named here rather than spelled as literals in both the stack and the scripts that
# read them: a rename in one place only would leave cleanup silently unable to find
# the lease table, and "cannot find it" must never be read as "nothing to inspect".
OUTPUT_LEASE_TABLE_NAME = "LeaseTableName"
OUTPUT_LEASE_FUNCTION_NAME = "LeaseFunctionName"
OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME = "QueueTriggerFunctionName"

DATASET_MANIFEST_NAME = "dataset_manifest.json"

# Publication-complete marker, written LAST under a dataset version prefix.
# A consumer that sees it knows every split finished uploading; without it a
# pipeline starting mid-publication could read a mixture of two versions.
DATASET_READY_MARKER = "_READY"

PUBLICATION_INDEX_NAME = "publication_index.json"

# --- S3 key prefixes ---
# Execution-scoped run prefix: static prefixes would let concurrent fleet
# executions overwrite each other's processing outputs.
RUNS_PREFIX = "cv-pipeline/runs"
# Dataset publication prefix. Datasets are published under an immutable
# version directory so a pipeline can never download a half-replaced split.
DATASETS_PREFIX = "cv-pipeline/datasets"


def bucket_name(account: str, region: str) -> str:
    'Pipeline artifact bucket name for *account* in *region*.'
    return f"fleet-cv-{account}-{region}"


def execution_role_name(region: str) -> str:
    """SageMaker execution role name for *region*."""
    return f"FleetCVExecutionRole-{region}"


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def bootstrap_only(raw) -> bool:
    'Interpret the ``bootstrap`` CDK context value as a boolean.'
    if raw is None or isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Unrecognized value for CDK context 'bootstrap': {raw!r}. "
        f"Use one of {sorted(_TRUE_VALUES)} or {sorted(_FALSE_VALUES - {''})}."
    )


# A digest is 'sha256:' plus exactly 64 lowercase hex characters. Validated rather
# than interpolated, because a malformed value would be baked into the pipeline
# definition and only surface when a training job failed to pull.
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def training_image_digest(raw) -> str:
    'Interpret the ``training_image_digest`` CDK context value.'
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if not _DIGEST_PATTERN.match(text):
        raise ValueError(
            f"Invalid value for CDK context 'training_image_digest': {text!r}. "
            f"Expected 'sha256:' followed by 64 lowercase hex characters, as printed "
            f"by scripts/push_ecr_image.sh. A malformed digest would be baked into "
            f"the pipeline definition and only surface when a training job could not "
            f"pull its image."
        )
    return text
