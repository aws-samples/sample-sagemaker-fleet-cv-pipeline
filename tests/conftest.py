# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Import paths for the test suite, in one place.'
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Order matters only in that the repository root must not shadow the flat directories.
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "model",
    REPO_ROOT / "src" / "orchestration",
    REPO_ROOT / "src" / "data",
):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def pytest_runtest_setup(item):
    'Open AWS permission around the COMPLETE protocol of an integration-marked item.'
    _AWS_ALLOWED["on"] = item.get_closest_marker("integration") is not None


def pytest_runtest_teardown(item, nextitem):
    _AWS_ALLOWED["on"] = False


_AWS_ALLOWED = {"on": False}


class RealAwsCallInTest(AssertionError):
    """A test tried to talk to AWS for real."""


def _install_aws_guard():
    'Deny real AWS traffic from import time, before collection begins.'
    import botocore.httpsession

    real_send = botocore.httpsession.URLLib3Session.send

    def guarded_send(self, request, *args, **kwargs):
        if _AWS_ALLOWED["on"]:
            return real_send(self, request, *args, **kwargs)
        raise RealAwsCallInTest(
            f"A test made a REAL AWS request ({getattr(request, 'method', '?')} "
            f"{getattr(request, 'url', '?')}). Offline tests must use the module's fakes -- a real "
            f"call can MUTATE the deployed stack, which has happened: a parser assertion once "
            f"closed admission on the live deployment. Mark the test @pytest.mark.integration if "
            f"it genuinely needs AWS."
        )

    botocore.httpsession.URLLib3Session.send = guarded_send

    # Earlier, clearer failure for the common construction paths. Not sufficient alone -- see the
    # hook docstring -- but a better message than a request-time refusal.
    import boto3

    def _guard_constructor(real, kind):
        def guarded(*args, **kwargs):
            if _AWS_ALLOWED["on"]:
                return real(*args, **kwargs)
            service = args[0] if args else kwargs.get("service_name", "?")
            raise RealAwsCallInTest(
                f"A test tried to create a real boto3 {kind} for {service!r}. Use the module's "
                f"fakes, or mark the test @pytest.mark.integration."
            )
        return guarded

    boto3.client = _guard_constructor(boto3.client, "client")
    boto3.resource = _guard_constructor(boto3.resource, "resource")


_install_aws_guard()
