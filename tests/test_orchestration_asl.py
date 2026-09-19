# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Regression tests on the synthesized Step Functions definition.'
import json
from pathlib import Path

import pytest

aws_cdk = pytest.importorskip("aws_cdk")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TEST_ACCOUNT = "111111111111"
TEST_REGION = "us-east-1"

APPROVED_BASELINE_POLICIES = (
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
)
from config import ORCHESTRATION_STACK_NAME  # noqa: E402
from orchestration_stack import OrchestrationStack  # noqa: E402

MAX_CONCURRENCY = 5
MAX_SLOT_WAIT = 480


@pytest.fixture(scope="module")
def synthesized():
    app = aws_cdk.App()
    OrchestrationStack(
        app,
        ORCHESTRATION_STACK_NAME,
        max_concurrency=MAX_CONCURRENCY,
        env=aws_cdk.Environment(account=TEST_ACCOUNT, region=TEST_REGION),
    )
    template = app.synth().get_stack_by_name(ORCHESTRATION_STACK_NAME).template
    machines = [
        res for res in template["Resources"].values()
        if res["Type"] == "AWS::StepFunctions::StateMachine"
    ]
    assert len(machines) == 1
    parts = machines[0]["Properties"]["DefinitionString"]["Fn::Join"][1]
    rendered = "".join(p if isinstance(p, str) else "ARN" for p in parts)
    return json.loads(rendered)["States"], template


@pytest.fixture(scope="module")
def asl_states(synthesized):
    return synthesized[0]


@pytest.fixture(scope="module")
def template(synthesized):
    return synthesized[1]


def test_lease_table_has_no_ttl(template):
    'A TTL would expire the lease of a still-running pipeline.'
    tables = [r for r in template["Resources"].values()
              if r["Type"] == "AWS::DynamoDB::Table"]
    assert tables, "no lease table synthesized"
    for table in tables:
        assert "TimeToLiveSpecification" not in table["Properties"], (
            "the lease table must not expire leases by age"
        )


def test_max_concurrency_reaches_the_lease_manager(template):
    functions = [r for r in template["Resources"].values()
                 if r["Type"] == "AWS::Lambda::Function"]
    envs = [f["Properties"].get("Environment", {}).get("Variables", {})
            for f in functions]
    assert any(e.get("MAX_CONCURRENCY") == str(MAX_CONCURRENCY) for e in envs), (
        f"no Lambda carries MAX_CONCURRENCY={MAX_CONCURRENCY}"
    )


def test_wait_loop_is_bounded(asl_states):
    choices = asl_states["CheckSlotWaitBudget"]["Choices"]
    assert choices[0]["Variable"] == "$.slotWait.count"
    assert choices[0]["NumericLessThan"] == MAX_SLOT_WAIT
    assert asl_states["CheckSlotWaitBudget"]["Default"] == "AcquireTimedOut"
    assert asl_states["AcquireTimedOut"]["Type"] == "Fail"
    assert asl_states["WaitForSlot"]["Next"] == "AcquireLease"


def test_start_token_is_recorded_before_starting(asl_states):
    'An ambiguous start is only resolvable if the token was persisted first.'
    assert asl_states["GenerateClientToken"]["Next"] == "RecordStartIntent"
    assert asl_states["RecordStartIntent"]["Next"] == "StartPipelineExecution"
    payload = asl_states["RecordStartIntent"]["Parameters"]
    assert payload["action"] == "record_start"
    assert payload["clientToken.$"] == "$.clientToken.value"


@pytest.mark.parametrize("state_name", [
    "StartPipelineExecution",
    "DescribePipelineExecution",
    "RecordStartIntent",
    "RecordExecutionArn",
    "StopPipelineExecution",
    "DescribeAfterStop",
])
def test_unconfirmed_outcome_retains_the_lease(asl_states, state_name):
    'Errors while the lease is HELD must NOT release it.'
    catches = asl_states[state_name].get("Catch", [])
    targets = [c["Next"] for c in catches]

    catch_all = [c for c in catches if "States.ALL" in c["ErrorEquals"]]
    assert catch_all, f"{state_name} has no catch-all; an error could escape"

    destination = catch_all[-1]["Next"]
    if destination != "LeaseRetainedForReconciliation":
        hop = asl_states[destination]
        assert hop.get("Next") == "LeaseRetainedForReconciliation", (
            f"{state_name} must retain the lease on an unconfirmed outcome; its "
            f"catch-all reaches {destination}, which does not retain. Targets: "
            f"{targets}"
        )
    for catch in catches:
        if "States.ALL" in catch["ErrorEquals"]:
            continue
        target = asl_states[catch["Next"]]
        assert (target.get("Next") == "LeaseRetainedForReconciliation"
                or target.get("Type") == "Fail"), (
            f"{state_name} routes {catch['ErrorEquals']} to {catch['Next']}, which "
            f"neither retains the lease nor fails"
        )


def test_success_releases_then_notifies(asl_states):
    'Release, notify, THEN branch on what the release did.'
    assert asl_states["ReleaseLease"]["Next"] == "NotifySuccess"
    assert asl_states["NotifySuccess"]["Next"] == "CheckLeaseReleased"

    check = asl_states["CheckLeaseReleased"]
    assert check["Type"] == "Choice"
    released = [
        c for c in check["Choices"]
        if c.get("Variable") == "$.release.released" and c.get("BooleanEquals") is True
    ]
    assert len(released) == 1, check["Choices"]

    for target in (released[0]["Next"], check["Default"]):
        assert asl_states[target]["Type"] == "Succeed", target
    assert released[0]["Next"] != check["Default"]


def test_poll_exhaustion_stops_and_confirms_before_release(asl_states):
    choices = asl_states["CheckPipelineStatus"]["Choices"]
    nonterminal = [c for c in choices if "Stopping" in json.dumps(c)]
    assert nonterminal[-1]["Next"] == "StopPipelineExecution"
    assert asl_states["CheckStoppedStatus"]["Default"] == "ReleaseLeaseOnFailure"
    assert asl_states["CheckStoppedStatus"]["Choices"][-1]["Next"] == (
        "StopBudgetExhausted"
    )


def test_reconciler_is_scheduled_and_event_driven(template):
    'States.ALL does not catch States.Runtime, and an abort skips every Catch.'
    rules = [r for r in template["Resources"].values()
             if r["Type"] == "AWS::Events::Rule"]
    scheduled = [r for r in rules if "ScheduleExpression" in r["Properties"]]
    assert scheduled, "no scheduled reconciliation rule"
    payloads = json.dumps([r["Properties"].get("Targets", []) for r in rules])
    assert "reconcile" in payloads, "no rule invokes the reconcile action"


def test_a_closed_admission_ends_instead_of_waiting(asl_states):
    'Refused-for-teardown is terminal, and must never route to a release.'
    choice = asl_states["CheckLeaseAcquired"]

    closed = [
        c for c in choice["Choices"]
        if c.get("Variable") == "$.lease.reason"
        and c.get("StringEquals") == "admission_closed"
    ]
    assert len(closed) == 1, f"no admission_closed branch: {choice['Choices']}"
    target = closed[0]["Next"]

    assert target != choice["Default"], (
        "a closed admission falls through to the capacity-wait loop, so it waits "
        "out its budget instead of ending"
    )
    assert choice["Default"] == "IncrementSlotWaitCounter"

    state = asl_states[target]
    assert state["Type"] == "Fail", f"{target} is {state['Type']}, not terminal"
    assert state["Error"] == "AdmissionClosed"
    assert "Next" not in state and "Catch" not in state

    for name, other in asl_states.items():
        if "Release" in name:
            assert other.get("Next") != target
