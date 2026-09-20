# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'One place that answers "which account and Region am I operating on?".'
from __future__ import annotations

import os


# In the order a reader is most likely to set them. CDK_DEFAULT_REGION is what
# the CDK CLI exports from its own resolution, so it acts as the fallback.
REGION_VARS = ("AWS_REGION", "AWS_DEFAULT_REGION", "CDK_DEFAULT_REGION")


class AwsEnvError(RuntimeError):
    """Raised when the account/Region cannot be resolved unambiguously."""


def _boto3():
    """Import boto3 on demand, with an actionable error if it is absent."""
    try:
        import boto3  # noqa: PLC0415 -- deliberately lazy, see the note above
        return boto3
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise AwsEnvError(
            "boto3 is required for this operation but is not installed. Either set "
            "AWS_REGION and AWS_DEFAULT_REGION explicitly (which needs no AWS SDK), "
            "or run inside the project virtualenv: "
            "pip install -r requirements.txt"
        ) from exc


def resolve_region(explicit: str = None) -> str:
    'Return the one Region to operate on.'
    if explicit:
        return explicit

    from_env = {var: os.environ[var] for var in REGION_VARS if os.environ.get(var)}
    distinct = set(from_env.values())
    if len(distinct) > 1:
        conflict = ", ".join(f"{k}={v}" for k, v in sorted(from_env.items()))
        raise AwsEnvError(
            f"Conflicting AWS Regions in the environment: {conflict}. Set them to "
            f"the same Region (or unset all but one) and retry."
        )
    if distinct:
        return _publish_region(distinct.pop())

    region = _boto3().session.Session().region_name
    if not region:
        raise AwsEnvError(
            "AWS Region could not be resolved. Set AWS_REGION and "
            "AWS_DEFAULT_REGION to the same value, or configure a region on your "
            "AWS profile."
        )
    return _publish_region(region)


def _publish_region(region: str) -> str:
    'Write the resolved Region back into the environment, then return it.'
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        if os.environ.get(var) != region:
            os.environ[var] = region
    return region


def session(region: str = None):
    """A boto3 session pinned to the resolved Region."""
    return _boto3().session.Session(region_name=resolve_region(region))


def resolve_account(region: str = None, expected: str = None) -> str:
    'Return the account ID the Python process is authenticated against.'
    resolved_region = resolve_region(region)
    expected = expected if expected is not None else os.environ.get("CDK_DEFAULT_ACCOUNT")
    try:
        identity = (
            _boto3().session.Session(region_name=resolved_region)
            .client("sts")
            .get_caller_identity()
        )
    except AwsEnvError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface the real cause
        raise AwsEnvError(
            f"AWS account could not be resolved ({exc}). Configure credentials "
            f"(see README Prerequisites) and retry."
        ) from exc

    account = identity["Account"]
    if expected and expected != account:
        raise AwsEnvError(
            f"Account mismatch: the deployment target is {expected} but these "
            f"credentials belong to {account} (as {identity.get('Arn')}). An "
            f"account ID only controls ARNs and names -- it does not select "
            f"credentials -- so continuing would synthesize resources for one "
            f"account while calling AWS as another. Use the same profile for the "
            f"CDK CLI and for Python, e.g. export AWS_PROFILE=<profile> instead "
            f"of passing --profile to cdk alone."
        )
    return account


def resolve_env(region: str = None) -> tuple:
    """Return ``(account, region)``, both verified. Used by ``app.py``."""
    resolved_region = resolve_region(region)
    return resolve_account(resolved_region), resolved_region
