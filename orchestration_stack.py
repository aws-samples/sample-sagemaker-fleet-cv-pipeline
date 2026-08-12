#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Orchestration stack: SQS → Lambda → Step Functions with concurrency control.

Allows submitting N camera model training requests while limiting concurrent
SageMaker pipeline executions to a configurable maximum (default: 5).

Architecture:
    SQS FIFO Queue
        → Lambda (poller, triggered by SQS)
            → Step Functions State Machine
                → Acquire semaphore (DynamoDB atomic counter)
                → StartPipelineExecution (SageMaker)
                → Poll until complete
                → Release semaphore
                → SNS notification

Known limitation (sample scope): the semaphore is a shared arithmetic
counter, and its increment/decrement updates are not idempotent. If a
DynamoDB update commits but the response is lost, the Step Functions
retry re-applies the arithmetic -- acquisition can overcount (permanently
burning a slot until an operator resets ActiveCount) and release can
undercount (briefly admitting an extra execution). Every such failure is
surfaced through explicit Fail states rather than hidden. For
production-strength fleet orchestration, replace the counter with
per-execution lease records keyed by the state-machine execution ID so
each acquire/release is idempotent.
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_events as events,
    aws_events_targets as targets,
    aws_sqs as sqs,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    aws_lambda_event_sources as event_sources,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
)
from constructs import Construct

PIPELINE_NAME = "anomaly-detection-pipeline-e2e-final"


class OrchestrationStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        max_concurrency = kwargs.pop("max_concurrency", 5)
        pipeline_name = kwargs.pop("pipeline_name", PIPELINE_NAME)
        # Slot-wait budget: how long an execution polls for a free semaphore
        # slot before failing as AcquireTimedOut. Size it to your backlog:
        # worst-case wait is (queued executions / max_concurrency) x average
        # pipeline runtime. At the default (480 x 30s = 4 hours), cap 5, and
        # the sample's ~15-minute runtime, roughly 85 queued executions can
        # start before the rest time out. NOTE (sample scope): every waiting
        # execution polls DynamoDB on this interval, so a longer bound scales
        # state-transition cost linearly with backlog size. For sustained
        # fleet-scale backlogs, keep pending work in SQS and admit executions
        # only when capacity exists (bounded dispatcher) instead of raising
        # this bound.
        max_slot_wait_attempts = kwargs.pop("max_slot_wait_attempts", 480)
        super().__init__(scope, construct_id, **kwargs)

        # --- DynamoDB semaphore table ---
        semaphore_table = dynamodb.Table(
            self,
            "SemaphoreTable",
            table_name="sm1k-pipeline-semaphore",
            partition_key=dynamodb.Attribute(
                name="LockName", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- SQS FIFO queue + DLQ ---
        dlq = sqs.Queue(
            self,
            "TrainingDLQ",
            queue_name="sm1k-training-requests-dlq.fifo",
            fifo=True,
            retention_period=Duration.days(14),
        )
        queue = sqs.Queue(
            self,
            "TrainingQueue",
            queue_name="sm1k-training-requests.fifo",
            fifo=True,
            content_based_deduplication=True,
            visibility_timeout=Duration.minutes(15),
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=3, queue=dlq),
        )

        # --- SNS topic for completion notifications ---
        topic = sns.Topic(
            self,
            "CompletionTopic",
            topic_name="sm1k-pipeline-completions",
        )

        # --- Step Functions state machine ---
        # Acquire semaphore: a single atomic conditional increment. The
        # ConditionExpression enforces the cap inside DynamoDB itself, so
        # there is no increment-then-check window to undo (the previous
        # design compared $.semaphore.Attributes.ActiveCount.N in a Choice
        # state, but DynamoDB returns N as a JSON *string* and ASL numeric
        # comparisons do not coerce strings -- the over-limit rule never
        # matched and the cap was silently bypassed).
        acquire_semaphore = tasks.DynamoUpdateItem(
            self,
            "AcquireSemaphore",
            table=semaphore_table,
            key={"LockName": tasks.DynamoAttributeValue.from_string("pipeline-concurrency")},
            update_expression="SET ActiveCount = if_not_exists(ActiveCount, :zero) + :one",
            condition_expression="attribute_not_exists(ActiveCount) OR ActiveCount < :limit",
            expression_attribute_values={
                ":zero": tasks.DynamoAttributeValue.number_from_string("0"),
                ":one": tasks.DynamoAttributeValue.number_from_string("1"),
                ":limit": tasks.DynamoAttributeValue.number_from_string(str(max_concurrency)),
            },
            return_values=tasks.DynamoReturnValues.UPDATED_NEW,
            result_path="$.semaphore",
        )

        # Release semaphore (decrement)
        release_semaphore = tasks.DynamoUpdateItem(
            self,
            "ReleaseSemaphore",
            table=semaphore_table,
            key={"LockName": tasks.DynamoAttributeValue.from_string("pipeline-concurrency")},
            update_expression="SET ActiveCount = ActiveCount - :one",
            expression_attribute_values={
                ":one": tasks.DynamoAttributeValue.number_from_string("1"),
            },
            result_path=sfn.JsonPath.DISCARD,
        )

        # Over-limit path: the conditional acquire failed, so nothing was
        # incremented and there is nothing to release -- wait for a slot and
        # try again. The wait loop is bounded (max_slot_wait_attempts, a
        # stack prop) so a genuinely stuck fleet fails loudly instead of
        # spinning forever.
        init_slot_wait = sfn.Pass(
            self,
            "InitSlotWaitCounter",
            result=sfn.Result.from_object({"count": 0}),
            result_path="$.slotWait",
        )
        increment_slot_wait = sfn.Pass(
            self,
            "IncrementSlotWaitCounter",
            parameters={"count.$": "States.MathAdd($.slotWait.count, 1)"},
            result_path="$.slotWait",
        )
        wait_for_slot = sfn.Wait(
            self,
            "WaitForSlot",
            time=sfn.WaitTime.duration(Duration.seconds(30)),
        )
        acquire_timed_out = sfn.Fail(
            self,
            "AcquireTimedOut",
            cause="No semaphore slot became available within the wait bound",
            error="SemaphoreAcquireTimeout",
        )
        check_slot_wait = sfn.Choice(self, "CheckSlotWaitBudget")
        increment_slot_wait.next(check_slot_wait)
        check_slot_wait.when(
            sfn.Condition.number_less_than("$.slotWait.count", max_slot_wait_attempts),
            wait_for_slot,
        )
        check_slot_wait.otherwise(acquire_timed_out)
        wait_for_slot.next(acquire_semaphore)

        # ConditionalCheckFailedException means the cap is reached -- do not
        # retry it inline (MaxAttempts=0 routes it straight to the Catch and
        # the wait loop). Everything else gets transient-fault retries and,
        # if still failing, also falls into the bounded wait loop: no
        # semaphore was acquired, so waiting and re-acquiring is always safe.
        acquire_semaphore.add_retry(
            errors=["DynamoDB.ConditionalCheckFailedException"],
            max_attempts=0,
        )
        acquire_semaphore.add_retry(
            errors=["States.ALL"],
            max_attempts=3,
            interval=Duration.seconds(2),
            backoff_rate=2.0,
        )
        acquire_semaphore.add_catch(
            increment_slot_wait,
            errors=["States.ALL"],
            result_path="$.acquireError",
        )

        # Scoped ARNs for the SageMaker pipeline this state machine operates on
        pipeline_arns = [
            f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{pipeline_name}",
            f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{pipeline_name}/*",
        ]

        # Generate the idempotency token ONCE per state-machine execution and
        # store it in the state input. Referencing States.UUID() directly in
        # the Task would mint a fresh token on every retry attempt, defeating
        # deduplication; a stored UUID (36 chars, within the API's 32-128
        # constraint) makes retries of StartPipelineExecution true no-ops.
        generate_client_token = sfn.Pass(
            self,
            "GenerateClientToken",
            parameters={"value.$": "States.UUID()"},
            result_path="$.clientToken",
        )

        # Start pipeline execution
        start_pipeline = tasks.CallAwsService(
            self,
            "StartPipelineExecution",
            service="sagemaker",
            action="startPipelineExecution",
            parameters={
                "PipelineName": pipeline_name,
                "PipelineParameters": sfn.JsonPath.list_at("$.PipelineParameters"),
                "ClientRequestToken.$": "$.clientToken.value",
            },
            iam_resources=pipeline_arns,
            result_path="$.execution",
        )

        # Poll loop: wait → describe → check
        wait_poll = sfn.Wait(
            self,
            "WaitBeforePoll",
            time=sfn.WaitTime.duration(Duration.seconds(60)),
        )

        describe_execution = tasks.CallAwsService(
            self,
            "DescribePipelineExecution",
            service="sagemaker",
            action="describePipelineExecution",
            parameters={
                "PipelineExecutionArn": sfn.JsonPath.string_at(
                    "$.execution.PipelineExecutionArn"
                ),
            },
            iam_resources=pipeline_arns,
            result_path="$.status",
        )

        # Bounded poll loop. A state-machine-level timeout cannot be used to
        # cap runaway executions because timeouts skip Catch handlers and
        # would leak the semaphore; instead the loop counts iterations and
        # fails through the normal release path when exhausted.
        # 240 polls x 60s = 4 hours.
        max_poll_attempts = 240
        init_poll_counter = sfn.Pass(
            self,
            "InitPollCounter",
            result=sfn.Result.from_object({"count": 0}),
            result_path="$.poll",
        )
        increment_poll_counter = sfn.Pass(
            self,
            "IncrementPollCounter",
            parameters={"count.$": "States.MathAdd($.poll.count, 1)"},
            result_path="$.poll",
        )

        # Check if pipeline is done. Executing AND Stopping are nonterminal:
        # a Stopping pipeline still consumes resources, so releasing its
        # semaphore slot early would let a new execution start and exceed
        # the cap.
        is_nonterminal = sfn.Condition.or_(
            sfn.Condition.string_equals("$.status.PipelineExecutionStatus", "Executing"),
            sfn.Condition.string_equals("$.status.PipelineExecutionStatus", "Stopping"),
        )
        still_polling = sfn.Condition.and_(
            is_nonterminal,
            sfn.Condition.number_less_than("$.poll.count", max_poll_attempts),
        )
        check_status = sfn.Choice(self, "CheckPipelineStatus")

        # Poll loop wiring
        check_status.when(still_polling, wait_poll)
        wait_poll.next(describe_execution)
        describe_execution.next(increment_poll_counter)
        increment_poll_counter.next(check_status)

        # Poll exhaustion while the pipeline still runs: stop it and wait for
        # a terminal status BEFORE releasing the semaphore. Releasing while
        # the pipeline is live would hand its slot to a new execution and
        # exceed the concurrency cap.
        stop_pipeline = tasks.CallAwsService(
            self,
            "StopPipelineExecution",
            service="sagemaker",
            action="stopPipelineExecution",
            parameters={
                "PipelineExecutionArn": sfn.JsonPath.string_at(
                    "$.execution.PipelineExecutionArn"
                ),
                "ClientRequestToken.$": "States.UUID()",
            },
            iam_resources=pipeline_arns,
            result_path=sfn.JsonPath.DISCARD,
        )
        max_stop_wait_attempts = 30  # 30 x 60s = 30 minutes for the stop to land
        init_stop_counter = sfn.Pass(
            self,
            "InitStopWaitCounter",
            result=sfn.Result.from_object({"count": 0}),
            result_path="$.stopWait",
        )
        increment_stop_counter = sfn.Pass(
            self,
            "IncrementStopWaitCounter",
            parameters={"count.$": "States.MathAdd($.stopWait.count, 1)"},
            result_path="$.stopWait",
        )
        wait_for_stop = sfn.Wait(
            self,
            "WaitForStop",
            time=sfn.WaitTime.duration(Duration.seconds(60)),
        )
        describe_after_stop = tasks.CallAwsService(
            self,
            "DescribeAfterStop",
            service="sagemaker",
            action="describePipelineExecution",
            parameters={
                "PipelineExecutionArn": sfn.JsonPath.string_at(
                    "$.execution.PipelineExecutionArn"
                ),
            },
            iam_resources=pipeline_arns,
            result_path="$.status",
        )
        # If the pipeline will not stop (or its status cannot be read), fail
        # WITHOUT releasing: the slot is genuinely still consumed, and
        # freeing it would break the cap. Operator intervention required.
        stop_timeout = sfn.Fail(
            self,
            "StopTimeoutDetected",
            cause="Pipeline did not reach a terminal status after StopPipelineExecution;"
            " semaphore intentionally NOT released -- resolve the execution and"
            " decrement ActiveCount manually",
            error="PipelineStopTimeout",
        )
        check_stopped = sfn.Choice(self, "CheckStoppedStatus")
        stop_pipeline.next(init_stop_counter)
        init_stop_counter.next(wait_for_stop)
        wait_for_stop.next(describe_after_stop)
        describe_after_stop.next(increment_stop_counter)
        increment_stop_counter.next(check_stopped)
        check_stopped.when(
            sfn.Condition.and_(
                is_nonterminal,
                sfn.Condition.number_less_than("$.stopWait.count", max_stop_wait_attempts),
            ),
            wait_for_stop,
        )
        check_stopped.when(is_nonterminal, stop_timeout)

        # Notify success
        notify_success = tasks.SnsPublish(
            self,
            "NotifySuccess",
            topic=topic,
            message=sfn.TaskInput.from_object(
                {
                    "status": "SUCCEEDED",
                    "camera_id": sfn.JsonPath.string_at("$.camera_id"),
                    "execution_arn": sfn.JsonPath.string_at(
                        "$.execution.PipelineExecutionArn"
                    ),
                }
            ),
            result_path=sfn.JsonPath.DISCARD,
        )

        # Notify failure
        notify_failure = tasks.SnsPublish(
            self,
            "NotifyFailure",
            topic=topic,
            message=sfn.TaskInput.from_object(
                {
                    "status": "FAILED",
                    "camera_id": sfn.JsonPath.string_at("$.camera_id"),
                    "execution_arn": sfn.JsonPath.string_at(
                        "$.execution.PipelineExecutionArn"
                    ),
                    "pipeline_status": sfn.JsonPath.string_at(
                        "$.status.PipelineExecutionStatus"
                    ),
                }
            ),
            result_path=sfn.JsonPath.DISCARD,
        )

        # Wire the state machine
        # Success path: release semaphore → notify
        success_path = release_semaphore.next(notify_success)

        # Failure path: release semaphore copy → notify failure
        release_on_fail = tasks.DynamoUpdateItem(
            self,
            "ReleaseSemaphoreOnFail",
            table=semaphore_table,
            key={"LockName": tasks.DynamoAttributeValue.from_string("pipeline-concurrency")},
            update_expression="SET ActiveCount = ActiveCount - :one",
            expression_attribute_values={
                ":one": tasks.DynamoAttributeValue.number_from_string("1"),
            },
            result_path=sfn.JsonPath.DISCARD,
        )
        failure_path = release_on_fail.next(notify_failure)

        # Error path: release the semaphore when any task between acquire and
        # release throws (SageMaker API errors, SNS failures, DynamoDB
        # throttling in the poll loop). Without this, a single task error
        # terminates the execution with the counter still incremented and
        # permanently burns one slot of fleet concurrency. The release itself
        # gets bounded retries; if it still fails, Fail surfaces the leak
        # loudly instead of dying silently mid-state-machine.
        release_on_error = tasks.DynamoUpdateItem(
            self,
            "ReleaseSemaphoreOnError",
            table=semaphore_table,
            key={"LockName": tasks.DynamoAttributeValue.from_string("pipeline-concurrency")},
            update_expression="SET ActiveCount = ActiveCount - :one",
            expression_attribute_values={
                ":one": tasks.DynamoAttributeValue.number_from_string("1"),
            },
            result_path=sfn.JsonPath.DISCARD,
        )
        release_on_error.add_retry(
            errors=["States.ALL"],
            max_attempts=3,
            interval=Duration.seconds(2),
            backoff_rate=2.0,
        )
        execution_failed = sfn.Fail(
            self,
            "ExecutionFailed",
            cause="Task error after semaphore acquisition; semaphore released",
            error="OrchestrationTaskError",
        )
        release_on_error.next(execution_failed)

        # Transient-fault retries on the AWS API tasks themselves.
        for task in (start_pipeline, describe_execution, notify_success, notify_failure):
            task.add_retry(
                errors=["States.ALL"],
                max_attempts=3,
                interval=Duration.seconds(5),
                backoff_rate=2.0,
            )

        # Tasks that error while the semaphore is HELD funnel to the release.
        # Only pre-release tasks catch here: routing the notify tasks (which
        # run after release) through it would decrement the counter twice.
        for task in (start_pipeline, describe_execution):
            task.add_catch(
                release_on_error,
                errors=["States.ALL"],
                result_path="$.error",
            )

        # If any release exhausts its retries the slot genuinely leaks; fail
        # loudly through the same explicit state on every release path so an
        # operator resets the counter, instead of the execution dying with a
        # raw DynamoDB task error on some paths and SemaphoreReleaseError on
        # others.
        semaphore_leak = sfn.Fail(
            self,
            "SemaphoreLeakDetected",
            cause="Semaphore release failed after retries; reset ActiveCount manually",
            error="SemaphoreReleaseError",
        )
        for task in (release_semaphore, release_on_fail):
            task.add_retry(
                errors=["States.ALL"],
                max_attempts=3,
                interval=Duration.seconds(2),
                backoff_rate=2.0,
            )
        for task in (release_semaphore, release_on_fail, release_on_error):
            task.add_catch(
                semaphore_leak,
                errors=["States.ALL"],
                result_path="$.error",
            )

        # Stop-path task retries. Errors here must NOT route to a release:
        # the pipeline may still be running, so the slot is still consumed.
        for task in (stop_pipeline, describe_after_stop):
            task.add_retry(
                errors=["States.ALL"],
                max_attempts=3,
                interval=Duration.seconds(5),
                backoff_rate=2.0,
            )
            task.add_catch(
                stop_timeout,
                errors=["States.ALL"],
                result_path="$.error",
            )

        # Terminal states
        check_status.when(
            sfn.Condition.string_equals("$.status.PipelineExecutionStatus", "Succeeded"),
            success_path,
        )
        # Poll budget exhausted while the pipeline is still nonterminal
        # (still_polling above already failed, so reaching this rule means
        # the counter is spent): stop the pipeline first, then release.
        check_status.when(is_nonterminal, stop_pipeline)
        check_status.otherwise(failure_path)
        # Once the stopped pipeline reaches a terminal status, release the
        # slot and notify through the normal failure path.
        check_stopped.otherwise(failure_path)

        # Main flow: init wait budget → conditional acquire → mint client
        # token → start → poll loop. The acquire either succeeds atomically
        # or lands in the bounded WaitForSlot loop via its Catch; on success
        # execution proceeds through the token Pass to
        # StartPipelineExecution.
        acquire_semaphore.next(
            generate_client_token.next(
                start_pipeline.next(init_poll_counter.next(wait_poll))
            )
        )
        definition = init_slot_wait.next(acquire_semaphore)

        # NOTE on the state-machine timeout: a timeout terminates the
        # execution immediately without running Catch handlers, so it would
        # still leak the semaphore. The poll loop is instead bounded by
        # MaxPollAttempts below, which fails through the normal Catch path;
        # no top-level timeout is set.
        state_machine = sfn.StateMachine(
            self,
            "PipelineOrchestrator",
            state_machine_name="sm1k-pipeline-orchestrator",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
        )

        # Operational failure notifications. The in-flow SNS tasks only cover
        # the normal success/pipeline-failure paths; executions that end in
        # AcquireTimedOut, ExecutionFailed, StopTimeoutDetected, or
        # SemaphoreLeakDetected terminate without notifying. Rather than
        # inserting a publish before every Fail state (which would risk
        # replacing the original failure with an SNS failure), an EventBridge
        # rule forwards every FAILED/TIMED_OUT/ABORTED execution -- with its
        # original error and cause preserved in the event detail -- to the
        # same completion topic.
        events.Rule(
            self,
            "ExecutionFailureRule",
            rule_name="sm1k-orchestrator-failures",
            event_pattern=events.EventPattern(
                source=["aws.states"],
                detail_type=["Step Functions Execution Status Change"],
                detail={
                    "status": ["FAILED", "TIMED_OUT", "ABORTED"],
                    "stateMachineArn": [state_machine.state_machine_arn],
                },
            ),
            targets=[targets.SnsTopic(topic)],
        )

        # --- Lambda: SQS → Step Functions ---
        trigger_fn = _lambda.Function(
            self,
            "QueueTrigger",
            function_name="sm1k-queue-trigger",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="index.handler",
            timeout=Duration.seconds(30),
            environment={
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
            code=_lambda.Code.from_inline(
                """
import json, os, boto3

sfn = boto3.client("stepfunctions")
STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]

def handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        camera_id = body.get("camera_id", "default")
        # Build pipeline parameters from message. CameraId is always
        # forwarded so registry lineage and warm-start isolation work;
        # an explicit value in the message wins.
        params = [{"Name": "CameraId", "Value": camera_id}]
        for k, v in body.get("parameters", {}).items():
            if k == "CameraId":
                params[0]["Value"] = str(v)
                continue
            params.append({"Name": k, "Value": str(v)})
        sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            input=json.dumps({
                "camera_id": camera_id,
                "PipelineParameters": params,
            }),
        )
    return {"statusCode": 200}
"""
            ),
        )
        state_machine.grant_start_execution(trigger_fn)
        trigger_fn.add_event_source(
            event_sources.SqsEventSource(queue, batch_size=1)
        )

        # --- Outputs ---
        CfnOutput(self, "QueueUrl", value=queue.queue_url)
        CfnOutput(self, "DLQUrl", value=dlq.queue_url)
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        CfnOutput(self, "TopicArn", value=topic.topic_arn)