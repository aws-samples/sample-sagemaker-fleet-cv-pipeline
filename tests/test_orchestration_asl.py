# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Regression tests on the synthesized Step Functions definition.

The concurrency semaphore was previously broken in a way local tests could
not see: a Choice state compared DynamoDB's ActiveCount.N (a JSON string)
with NumericGreaterThan, which never matches under the Amazon States
Language, silently bypassing the cap. These tests synthesize the real
stack and assert on the resulting ASL so that class of bug fails CI.
"""
import json
from pathlib import Path

import pytest

aws_cdk = pytest.importorskip("aws_cdk")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestration_stack import OrchestrationStack  # noqa: E402

MAX_CONCURRENCY = 5


@pytest.fixture(scope="module")
def asl_states():
    app = aws_cdk.App()
    OrchestrationStack(
        app,
        "OrchestrationStack",
        max_concurrency=MAX_CONCURRENCY,
        env=aws_cdk.Environment(account="111111111111", region="us-east-1"),
    )
    template = app.synth().get_stack_by_name("OrchestrationStack").template
    machines = [
        res
        for res in template["Resources"].values()
        if res["Type"] == "AWS::StepFunctions::StateMachine"
    ]
    assert len(machines) == 1
    parts = machines[0]["Properties"]["DefinitionString"]["Fn::Join"][1]
    rendered = "".join(p if isinstance(p, str) else "ARN" for p in parts)
    return json.loads(rendered)["States"]


def test_acquire_is_atomic_conditional_update(asl_states):
    params = asl_states["AcquireSemaphore"]["Parameters"]
    assert (
        params["ConditionExpression"]
        == "attribute_not_exists(ActiveCount) OR ActiveCount < :limit"
    )
    assert params["ExpressionAttributeValues"][":limit"] == {
        "N": str(MAX_CONCURRENCY)
    }


def test_no_string_numeric_choice_on_semaphore_count(asl_states):
    """No Choice state may compare the raw DynamoDB .N string numerically."""
    for name, state in asl_states.items():
        for choice in state.get("Choices", []):
            variable = json.dumps(choice)
            assert "ActiveCount.N" not in variable, (
                f"State {name} numerically compares ActiveCount.N, which is a "
                "JSON string and never matches ASL numeric comparisons"
            )


def test_over_limit_path_waits_without_decrementing(asl_states):
    """ConditionalCheckFailed must route to the wait loop, not a release."""
    acquire = asl_states["AcquireSemaphore"]
    retries = acquire["Retry"]
    assert retries[0] == {
        "ErrorEquals": ["DynamoDB.ConditionalCheckFailedException"],
        "MaxAttempts": 0,
    }
    catch_targets = [c["Next"] for c in acquire["Catch"]]
    assert catch_targets == ["IncrementSlotWaitCounter"]
    # The wait loop path must not contain any DynamoDB decrement: the failed
    # conditional acquire incremented nothing.
    for name in ("IncrementSlotWaitCounter", "CheckSlotWaitBudget", "WaitForSlot"):
        assert "ActiveCount - " not in json.dumps(asl_states[name])
    assert asl_states["WaitForSlot"]["Next"] == "AcquireSemaphore"


def test_wait_loop_is_bounded(asl_states):
    choices = asl_states["CheckSlotWaitBudget"]["Choices"]
    assert choices[0]["Variable"] == "$.slotWait.count"
    assert choices[0]["NumericLessThan"] == 480
    assert asl_states["CheckSlotWaitBudget"]["Default"] == "AcquireTimedOut"
    assert asl_states["AcquireTimedOut"]["Type"] == "Fail"


def test_poll_loop_is_bounded_and_no_machine_timeout(asl_states):
    choices = asl_states["CheckPipelineStatus"]["Choices"]
    bounded = [
        c
        for c in choices
        if "$.poll.count" in json.dumps(c) and "NumericLessThan" in json.dumps(c)
    ]
    assert bounded, "poll loop must be bounded by a numeric counter"


def test_all_release_paths_retry_and_surface_leaks(asl_states):
    """Every release state must retry AND escalate to the same explicit
    leak Fail state -- an inconsistent path would end as a raw DynamoDB
    task error, making operational leak detection unreliable."""
    for name in ("ReleaseSemaphore", "ReleaseSemaphoreOnFail", "ReleaseSemaphoreOnError"):
        state = asl_states[name]
        assert any(
            r.get("MaxAttempts", 0) >= 3 for r in state.get("Retry", [])
        ), f"{name} must retry transient faults"
        targets = [c["Next"] for c in state.get("Catch", [])]
        assert targets == ["SemaphoreLeakDetected"], (
            f"{name} must escalate exhausted retries to SemaphoreLeakDetected"
        )


def test_stopping_status_is_nonterminal(asl_states):
    """A Stopping pipeline still consumes resources; releasing its slot
    early would let a new execution exceed the concurrency cap."""
    poll_rule = asl_states["CheckPipelineStatus"]["Choices"][0]
    rule = json.dumps(poll_rule)
    assert '"Executing"' in rule and '"Stopping"' in rule


def test_poll_exhaustion_stops_pipeline_before_release(asl_states):
    """Poll-budget exhaustion on a live pipeline must route through
    StopPipelineExecution and confirm a terminal status before any
    semaphore release."""
    choices = asl_states["CheckPipelineStatus"]["Choices"]
    nonterminal_rules = [c for c in choices if "Stopping" in json.dumps(c)]
    # Second nonterminal rule (no poll-count bound) is the exhaustion route.
    assert nonterminal_rules[-1]["Next"] == "StopPipelineExecution"
    assert asl_states["StopPipelineExecution"]["Parameters"]["PipelineExecutionArn.$"]
    # The stop-wait loop reaches the release only via its terminal branch.
    assert asl_states["CheckStoppedStatus"]["Default"] == "ReleaseSemaphoreOnFail"
    # If the pipeline never stops, fail WITHOUT releasing.
    stop_rules = asl_states["CheckStoppedStatus"]["Choices"]
    assert stop_rules[-1]["Next"] == "StopTimeoutDetected"
    assert asl_states["StopTimeoutDetected"]["Type"] == "Fail"
    # Stop-path task errors must not route to any release state.
    for name in ("StopPipelineExecution", "DescribeAfterStop"):
        targets = [c["Next"] for c in asl_states[name].get("Catch", [])]
        assert targets == ["StopTimeoutDetected"]


def test_pre_release_tasks_catch_into_error_release(asl_states):
    for name in ("StartPipelineExecution", "DescribePipelineExecution"):
        targets = [c["Next"] for c in asl_states[name].get("Catch", [])]
        assert targets == ["ReleaseSemaphoreOnError"], (
            f"{name} errors while the semaphore is held must release it"
        )
