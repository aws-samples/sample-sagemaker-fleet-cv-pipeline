# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Tests for the account/Region resolution and bootstrap contracts.'
import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import aws_env  # noqa: E402


@pytest.mark.parametrize("var", aws_env.REGION_VARS)
def test_any_single_region_variable_is_accepted(monkeypatch, var):
    """Readers set whichever variable they know about; all are honoured."""
    monkeypatch.setenv(var, "us-east-2")
    assert aws_env.resolve_region() == "us-east-2"


def test_agreeing_region_variables_are_accepted(monkeypatch):
    for var in aws_env.REGION_VARS:
        monkeypatch.setenv(var, "eu-west-1")
    assert aws_env.resolve_region() == "eu-west-1"


def test_conflicting_regions_raise_rather_than_choosing(monkeypatch):
    'A disagreement must not be silently resolved.'
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    with pytest.raises(aws_env.AwsEnvError, match="Conflicting AWS Regions"):
        aws_env.resolve_region()
    try:
        aws_env.resolve_region()
    except aws_env.AwsEnvError as exc:
        assert "us-east-2" in str(exc) and "us-west-2" in str(exc)


def test_explicit_region_wins_over_environment(monkeypatch):
    """A --region flag overrides the environment, including a conflict."""
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    assert aws_env.resolve_region("eu-central-1") == "eu-central-1"


def test_unresolvable_region_raises_with_guidance(monkeypatch):
    monkeypatch.setattr(
        aws_env, "_boto3",
        lambda: mock.Mock(session=mock.Mock(
            Session=lambda *a, **k: mock.Mock(region_name=None))),
    )
    with pytest.raises(aws_env.AwsEnvError, match="Region could not be resolved"):
        aws_env.resolve_region()


def _stub_identity(monkeypatch, account, arn="arn:aws:iam::x:user/test"):
    client = mock.Mock()
    client.get_caller_identity.return_value = {"Account": account, "Arn": arn}
    monkeypatch.setattr(
        aws_env, "_boto3",
        lambda: mock.Mock(session=mock.Mock(
            Session=lambda *a, **k: mock.Mock(
                client=lambda name: client, region_name="us-east-2"))),
    )


def test_account_comes_from_caller_identity(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    _stub_identity(monkeypatch, "111111111111")
    assert aws_env.resolve_account() == "111111111111"


def test_matching_expected_account_is_accepted(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "111111111111")
    _stub_identity(monkeypatch, "111111111111")
    assert aws_env.resolve_account() == "111111111111"


def test_account_mismatch_raises(monkeypatch):
    'A target account does not select credentials, so it must be checked.'
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "222222222222")
    _stub_identity(monkeypatch, "111111111111")
    with pytest.raises(aws_env.AwsEnvError, match="Account mismatch"):
        aws_env.resolve_account()
    try:
        aws_env.resolve_account()
    except aws_env.AwsEnvError as exc:
        assert "222222222222" in str(exc) and "111111111111" in str(exc)


def test_unusable_credentials_raise_with_guidance(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-2")
    client = mock.Mock()
    client.get_caller_identity.side_effect = RuntimeError("expired token")
    monkeypatch.setattr(
        aws_env, "_boto3",
        lambda: mock.Mock(session=mock.Mock(
            Session=lambda *a, **k: mock.Mock(
                client=lambda name: client, region_name="us-east-2"))),
    )
    with pytest.raises(aws_env.AwsEnvError, match="account could not be resolved"):
        aws_env.resolve_account()


def test_aws_env_import_is_stdlib_only():
    'Importing must not require boto3, nor call AWS, nor synthesize anything.'
    source = (REPO_ROOT / "aws_env.py").read_text()
    assert "\nimport boto3" not in source, "boto3 must not be imported at module level"

    import subprocess
    code = (
        "import sys, os;"
        "sys.modules['boto3'] = None;"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "os.environ['AWS_REGION'] = 'eu-west-1';"
        "os.environ.pop('AWS_DEFAULT_REGION', None);"
        "os.environ.pop('CDK_DEFAULT_REGION', None);"
        "import aws_env;"
        "print(aws_env.resolve_region())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "eu-west-1"


@pytest.fixture(scope="module")
def stack_module():
    'Import the pipeline stack.'
    return importlib.import_module("sagemaker_cv_pipeline_stack")


@pytest.mark.parametrize("raw,expected", [
    (None, False), (True, True), (False, False),
    ("true", True), ("True", True), ("TRUE", True), ("1", True),
    ("yes", True), ("on", True),
    ("false", False), ("False", False), ("0", False), ("no", False),
    ("off", False), ("", False),
])
def test_bootstrap_context_parsing(raw, expected):
    """Runs OFFLINE: config.py imports only the standard library."""
    import config
    assert config.bootstrap_only(raw) is expected


def test_bootstrap_rejects_ambiguous_value():
    import config
    with pytest.raises(ValueError, match="Unrecognized value"):
        config.bootstrap_only("maybe")


def test_config_import_needs_no_aws():
    'config.py must stay free of AWS imports.'
    source = (REPO_ROOT / "config.py").read_text()
    for forbidden in ("import boto3", "import aws_cdk", "from sagemaker"):
        assert forbidden not in source, f"config.py must not contain {forbidden!r}"


@pytest.mark.integration
def test_unresolved_account_is_rejected(stack_module):
    'A stack built without an explicit env must fail loudly.'
    aws_cdk = pytest.importorskip("aws_cdk")
    app = aws_cdk.App()
    with pytest.raises(ValueError, match="Unresolved account"):
        stack_module.SageMakerCVPipelineStack(app, "NoEnvStack")


@pytest.mark.integration
def test_unresolved_region_is_rejected(stack_module):
    with pytest.raises(ValueError, match="Unresolved region"):
        stack_module.SageMakerCVPipelineStack.__dict__[
            "_create_cv_pipeline_definition"
        ](mock.Mock(), account="111111111111", region="${Token[AWS.Region.7]}")


def test_no_aws_clients_at_stack_import():
    'Importing the stack module must not construct an AWS client.'
    import subprocess

    probe = (
        "import boto3, sys\n"
        "built = []\n"
        "real = boto3.client\n"
        "boto3.client = lambda s, *a, **k: (built.append(s), real(s, *a, **k))[1]\n"
        "sys.path.insert(0, %r)\n"
        "import sagemaker_cv_pipeline_stack\n"
        "print('CLIENTS:' + ','.join(built))\n"
    ) % str(REPO_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("CLIENTS:")]
    assert line, f"probe produced no marker; stdout was {result.stdout[-500:]!r}"
    built = [c for c in line[0][len("CLIENTS:"):].split(",") if c]
    assert built == [], (
        f"importing the stack constructed AWS clients {built}. An AWS-active SDK "
        f"import has been hoisted to module scope; see 'NOTE ON IMPORT SAFETY' in "
        f"sagemaker_cv_pipeline_stack.py"
    )


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_an_unset_digest_is_allowed(raw):
    """The first deploy has no image to pin yet; that is the bootstrap order."""
    import config

    assert config.training_image_digest(raw) == ""


def test_a_valid_digest_is_returned_verbatim():
    import config

    digest = "sha256:" + "0123456789abcdef" * 4
    assert config.training_image_digest(digest) == digest
    assert config.training_image_digest(f"  {digest}  ") == digest


@pytest.mark.parametrize(
    "bad",
    [
        "notadigest",
        "sha256:tooshort",
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "sha256:" + "A" * 64,          # uppercase is not what ECR reports
        "sha512:" + "a" * 64,
        "a" * 64,                       # bare hex, no algorithm
        "latest",
    ],
)
def test_a_malformed_digest_is_rejected(bad):
    'A malformed digest would be baked into the pipeline definition.'
    import config

    with pytest.raises(ValueError, match="training_image_digest"):
        config.training_image_digest(bad)


FIXTURE_ACCOUNT = "210987654321"


_ALIASES = {"--file": "-f", "-f": "-f", "--tag": "-t", "-t": "-t"}




@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove every AWS variable so each case starts from a known state."""
    for var in (
        "AWS_REGION", "AWS_DEFAULT_REGION", "CDK_DEFAULT_REGION",
        "CDK_DEFAULT_ACCOUNT", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
