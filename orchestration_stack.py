#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Orchestration stack: SQS -> Lambda -> Step Functions with concurrency control.'

import sys
from pathlib import Path

from aws_cdk import (
    ArnFormat,
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_events as events,
    aws_events_targets as targets,
    aws_sqs as sqs,
    aws_sns as sns,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_lambda_event_sources as event_sources,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
)
from constructs import Construct

import config

# The compilation job naming contract, shared with the driver that creates the jobs and
# the lease manager that inspects them. Imported here so the IAM scope is derived from
# the same model-type list rather than a third copy of it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "orchestration"))
import neo_jobs  # noqa: E402

from config import (
    RUNS_PREFIX,
    COMPLETION_TOPIC_NAME,
    FAILURE_RULE_NAME,
    OUTPUT_DLQ_URL,
    OUTPUT_LEASE_FUNCTION_NAME,
    OUTPUT_LEASE_TABLE_NAME,
    OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME,
    OUTPUT_QUEUE_URL,
    OUTPUT_STATE_MACHINE_ARN,
    OUTPUT_TOPIC_ARN,
    PIPELINE_NAME,
    QUEUE_TRIGGER_FUNCTION_NAME,
    SEMAPHORE_TABLE_NAME,
    STATE_MACHINE_NAME,
    TRAINING_DLQ_NAME,
    TRAINING_QUEUE_NAME,
)

LEASE_TABLE_NAME = f"{SEMAPHORE_TABLE_NAME}-leases"


class OrchestrationStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        max_concurrency = kwargs.pop("max_concurrency", 5)
        pipeline_name = kwargs.pop("pipeline_name", PIPELINE_NAME)
        max_slot_wait_attempts = kwargs.pop("max_slot_wait_attempts", 480)
        stale_lease_seconds = kwargs.pop("stale_lease_seconds", 6 * 60 * 60)
        super().__init__(scope, construct_id, **kwargs)

        # --- DynamoDB lease table ---
        lease_table = dynamodb.Table(
            self,
            "LeaseTable",
            table_name=LEASE_TABLE_NAME,
            partition_key=dynamodb.Attribute(
                name="LeaseId", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
        )

        # --- SQS FIFO queue + DLQ ---
        dlq = sqs.Queue(
            self,
            "TrainingDLQ",
            queue_name=TRAINING_DLQ_NAME,
            fifo=True,
            retention_period=Duration.days(14),
        )
        queue = sqs.Queue(
            self,
            "TrainingQueue",
            queue_name=TRAINING_QUEUE_NAME,
            fifo=True,
            # Producer-supplied deduplication IDs (see scripts/submit_batch.py):
            # content-based dedup would suppress a deliberate retrain of the same
            # camera with the same parameters inside the 5-minute window.
            content_based_deduplication=False,
            visibility_timeout=Duration.minutes(15),
            dead_letter_queue=sqs.DeadLetterQueue(max_receive_count=3, queue=dlq),
        )

        # --- SNS topic for completion notifications ---
        topic = sns.Topic(
            self,
            "CompletionTopic",
            topic_name=COMPLETION_TOPIC_NAME,
        )

        pipeline_arns = [
            f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{pipeline_name}",
            f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{pipeline_name}/*",
        ]

        lease_fn = _lambda.Function(
            self,
            "LeaseManager",
            function_name=f"{QUEUE_TRIGGER_FUNCTION_NAME}-lease",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lease_handler.handler",
            code=_lambda.Code.from_asset(
                "src/orchestration", exclude=["**/__pycache__/**", "**/*.pyc"],
            ),
            timeout=Duration.minutes(2),
            environment={
                "LEASE_TABLE_NAME": lease_table.table_name,
                "MAX_CONCURRENCY": str(max_concurrency),
                "PIPELINE_NAME": pipeline_name,
                "STALE_LEASE_SECONDS": str(stale_lease_seconds),
                "RUNS_BUCKET": config.bucket_name(
                    Stack.of(self).account, Stack.of(self).region
                ),
                "RUNS_PREFIX": RUNS_PREFIX,
            },
            log_group=logs.LogGroup(
                self,
                "LeaseManagerLogs",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            ),
        )
        lease_table.grant_read_write_data(lease_fn)
        lease_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:DescribePipelineExecution",
                    "sagemaker:ListPipelineExecutions",
                ],
                resources=pipeline_arns,
            )
        )
        lease_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:DescribeCompilationJob",
                ],
                resources=[
                    Stack.of(self).format_arn(
                        service="sagemaker",
                        resource="compilation-job",
                        resource_name=f"neo-{model_type}-*",
                        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
                    )
                    for model_type in neo_jobs.MODEL_TYPES
                ],
            )
        )

        # Read-only, and scoped to the run prefix. The lease manager only ever reads the
        # submission records the compilation driver wrote; it does not read or write
        # anything else in the bucket.
        lease_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[
                    f"arn:{Stack.of(self).partition}:s3:::"
                    f"{config.bucket_name(Stack.of(self).account, Stack.of(self).region)}"
                    f"/{RUNS_PREFIX}/*"
                ],
            )
        )

        lease_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[
                    f"arn:{Stack.of(self).partition}:s3:::"
                    f"{config.bucket_name(Stack.of(self).account, Stack.of(self).region)}"
                ],
                conditions={"StringLike": {"s3:prefix": [f"{RUNS_PREFIX}/*"]}},
            )
        )

        def lease_task(name: str, payload: dict, result_path: str) -> tasks.LambdaInvoke:
            return tasks.LambdaInvoke(
                self,
                name,
                lambda_function=lease_fn,
                payload=sfn.TaskInput.from_object(payload),
                # Unwrap the Lambda envelope so downstream Choice states read the
                # handler's own fields.
                payload_response_only=True,
                result_path=result_path,
            )

        acquire_lease = lease_task(
            "AcquireLease",
            {
                "action": "acquire",
                "executionName.$": "$$.Execution.Name",
                "requestId.$": "$.request_id",
            },
            "$.lease",
        )

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
            cause="No lease became available within the wait bound; nothing was acquired",
            error="LeaseAcquireTimeout",
        )
        check_slot_wait = sfn.Choice(self, "CheckSlotWaitBudget")

        # === Start ============================================================
        generate_client_token = sfn.Pass(
            self,
            "GenerateClientToken",
            parameters={"value.$": "States.UUID()"},
            result_path="$.clientToken",
        )

        record_start = lease_task(
            "RecordStartIntent",
            {
                "action": "record_start",
                "executionName.$": "$$.Execution.Name",
                "clientToken.$": "$.clientToken.value",
                "parameters.$": "$.PipelineParameters",
                "displayName.$": "$$.Execution.Name",
            },
            "$.recordStart",
        )

        start_pipeline = tasks.CallAwsService(
            self,
            "StartPipelineExecution",
            service="sagemaker",
            action="startPipelineExecution",
            parameters={
                "PipelineName": pipeline_name,
                "PipelineParameters": sfn.JsonPath.list_at("$.PipelineParameters"),
                "ClientRequestToken.$": "$.clientToken.value",
                "PipelineExecutionDisplayName.$": "$$.Execution.Name",
            },
            iam_resources=pipeline_arns,
            result_path="$.execution",
        )

        record_arn = lease_task(
            "RecordExecutionArn",
            {
                "action": "record_start",
                "executionName.$": "$$.Execution.Name",
                "clientToken.$": "$.clientToken.value",
                "executionArn.$": "$.execution.PipelineExecutionArn",
            },
            "$.recordArn",
        )

        # === Poll =============================================================
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
        # A state-machine-level timeout cannot be used to bound this: timeouts
        # skip Catch handlers, so the lease would be left behind with no failure
        # state recording why. The loop counts iterations instead. 240 x 60s = 4h.
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
        # Executing AND Stopping are both nonterminal: a Stopping pipeline still
        # consumes resources, so releasing its lease early would let a new
        # execution start and exceed the cap.
        is_nonterminal = sfn.Condition.or_(
            sfn.Condition.string_equals("$.status.PipelineExecutionStatus", "Executing"),
            sfn.Condition.string_equals("$.status.PipelineExecutionStatus", "Stopping"),
        )
        still_polling = sfn.Condition.and_(
            is_nonterminal,
            sfn.Condition.number_less_than("$.poll.count", max_poll_attempts),
        )
        check_status = sfn.Choice(self, "CheckPipelineStatus")

        # === Stop path ========================================================
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
        max_stop_wait_attempts = 30  # 30 x 60s
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

        # === Release ==========================================================
        release_lease = lease_task(
            "ReleaseLease",
            {"action": "release", "executionName.$": "$$.Execution.Name"},
            "$.release",
        )
        release_lease_on_failure = lease_task(
            "ReleaseLeaseOnFailure",
            {"action": "release", "executionName.$": "$$.Execution.Name"},
            "$.release",
        )

        # === Terminal states ==================================================
        notify_success = tasks.SnsPublish(
            self,
            "NotifySuccess",
            topic=topic,
            message=sfn.TaskInput.from_object(
                {
                    "status": "SUCCEEDED",
                    "camera_id": sfn.JsonPath.string_at("$.camera_id"),
                    "request_id": sfn.JsonPath.string_at("$.request_id"),
                    "execution_arn": sfn.JsonPath.string_at(
                        "$.execution.PipelineExecutionArn"
                    ),
                }
            ),
            result_path=sfn.JsonPath.DISCARD,
        )

        pipeline_failed = sfn.Fail(
            self,
            "PipelineFailed",
            cause_path=sfn.JsonPath.format(
                "Pipeline execution {} ended with status {}",
                sfn.JsonPath.string_at("$.execution.PipelineExecutionArn"),
                sfn.JsonPath.string_at("$.status.PipelineExecutionStatus"),
            ),
            error="PipelineExecutionFailed",
        )

        # Lease deliberately RETAINED. Reaching here means the pipeline's state
        # could not be established, so freeing the slot could exceed the cap while
        # the pipeline still runs. The reconciler resolves it against real status.
        lease_retained = sfn.Fail(
            self,
            "LeaseRetainedForReconciliation",
            cause_path=sfn.JsonPath.format(
                "Outcome could not be established ({}); the lease is intentionally "
                "retained and will be resolved by the reconciler against the real "
                "pipeline status. Cause: {}",
                sfn.JsonPath.string_at("$.error.Error"),
                sfn.JsonPath.string_at("$.error.Cause"),
            ),
            error="LeaseRetainedPendingReconciliation",
        )
        stop_budget_exhausted = sfn.Fail(
            self,
            "StopBudgetExhausted",
            cause=(
                "The pipeline was asked to stop and the stop request succeeded, "
                "but the pipeline did not reach a terminal status within the "
                "configured stop-wait budget. The lease is intentionally RETAINED: "
                "the pipeline is demonstrably still consuming resources, so "
                "freeing the slot could exceed the concurrency cap. The "
                "reconciler resolves this lease against the real pipeline status; "
                "for immediate recovery invoke the lease function with "
                '{"action": "reconcile", "ExecutionName": "<this execution name>"}.'
            ),
            error="StopBudgetExhausted",
        )
        release_failed = sfn.Fail(
            self,
            "LeaseReleaseFailed",
            cause_path=sfn.JsonPath.format(
                "Releasing the lease failed after retries: {}",
                sfn.JsonPath.string_at("$.error.Cause"),
            ),
            error="LeaseReleaseError",
        )

        # === Retries and catches ==============================================
        for task in (acquire_lease, record_start, record_arn, release_lease,
                     release_lease_on_failure, describe_execution, notify_success):
            task.add_retry(
                errors=["States.ALL"],
                max_attempts=3,
                interval=Duration.seconds(5),
                backoff_rate=2.0,
            )

        start_pipeline.add_retry(
            errors=["SageMaker.ThrottlingException"],
            max_attempts=3,
            interval=Duration.seconds(5),
            backoff_rate=2.0,
        )

        record_rejection = lease_task(
            "RecordStartRejected",
            {
                "action": "record_start_rejected",
                "ExecutionName.$": "$$.Execution.Name",
                "Error.$": "$.error.Error",
            },
            "$.rejection",
        )
        record_rejection.next(lease_retained)
        # If the marking call itself fails, fall back to the retain path: an
        # unmarked lease is recovered by the operator, which is the safe direction.
        record_rejection.add_catch(
            lease_retained, errors=["States.ALL"], result_path="$.markError"
        )

        record_ambiguity = lease_task(
            "RecordAmbiguousStart",
            {
                "action": "record_ambiguous_attempt",
                "ExecutionName.$": "$$.Execution.Name",
            },
            "$.ambiguity",
        )
        record_ambiguity.next(lease_retained)
        # If the marker cannot be written, still retain: an unmarked lease is
        # recovered by an operator, which is the safe direction.
        record_ambiguity.add_catch(
            lease_retained, errors=["States.ALL"], result_path="$.markError"
        )

        AUTHORIZATION_REFUSALS = [
            "SageMaker.AccessDeniedException",
            "SageMaker.NotAuthorizedException",
            "SageMaker.UnrecognizedClientException",
            "SageMaker.ExpiredTokenException",
            "SageMaker.IncompleteSignatureException",
            "SageMaker.MalformedHttpRequestException",
            "SageMaker.OptInRequiredException",
            "SageMaker.RequestEntityTooLargeException",
            "SageMaker.UnknownOperationException",
            "States.Permissions",
        ]
        DEFINITIVE_START_REJECTIONS = [
            "SageMaker.ValidationException",
            "SageMaker.ValidationErrorException",
            "SageMaker.ResourceNotFoundException",
            "SageMaker.ResourceLimitExceededException",
            "SageMaker.ThrottlingException",
            *AUTHORIZATION_REFUSALS,
        ]
        start_pipeline.add_catch(
            record_rejection,
            errors=DEFINITIVE_START_REJECTIONS,
            result_path="$.error",
        )

        for task in (record_start, record_arn, describe_execution):
            task.add_catch(
                lease_retained, errors=["States.ALL"], result_path="$.error"
            )

        start_pipeline.add_catch(
            record_ambiguity, errors=["States.ALL"], result_path="$.error"
        )

        # A failed acquire acquired nothing, so waiting and retrying is always
        # safe -- there is no lease to leak.
        acquire_lease.add_catch(
            increment_slot_wait, errors=["States.ALL"], result_path="$.acquireError"
        )
        for task in (release_lease, release_lease_on_failure):
            task.add_catch(release_failed, errors=["States.ALL"], result_path="$.error")

        # Stop-path errors must not release: the pipeline may still be running.
        for task in (stop_pipeline, describe_after_stop):
            task.add_retry(
                errors=["States.ALL"],
                max_attempts=3,
                interval=Duration.seconds(5),
                backoff_rate=2.0,
            )
            task.add_catch(
                lease_retained, errors=["States.ALL"], result_path="$.error"
            )

        # === Wiring ===========================================================
        check_acquired = sfn.Choice(self, "CheckLeaseAcquired")

        definition = init_slot_wait.next(acquire_lease)
        acquire_lease.next(check_acquired)

        check_acquired.when(
            sfn.Condition.boolean_equals("$.lease.acquired", True),
            generate_client_token
            .next(record_start)
            .next(start_pipeline)
            .next(record_arn)
            .next(init_poll_counter)
            .next(wait_poll),
        )
        admission_closed = sfn.Fail(
            self,
            "AdmissionClosed",
            cause=(
                "Admission is closed for teardown or maintenance; nothing was "
                "acquired and no pipeline was started. Reopen admission and "
                "resubmit with a new batch id."
            ),
            error="AdmissionClosed",
        )
        check_acquired.when(
            sfn.Condition.string_equals("$.lease.reason", "admission_closed"),
            admission_closed,
        )
        check_acquired.otherwise(increment_slot_wait)

        increment_slot_wait.next(check_slot_wait)
        check_slot_wait.when(
            sfn.Condition.number_less_than("$.slotWait.count", max_slot_wait_attempts),
            wait_for_slot,
        )
        check_slot_wait.otherwise(acquire_timed_out)
        wait_for_slot.next(acquire_lease)

        wait_poll.next(describe_execution)
        describe_execution.next(increment_poll_counter)
        increment_poll_counter.next(check_status)

        check_status.when(still_polling, wait_poll)
        succeeded = sfn.Succeed(self, "Succeeded")
        lease_retained_after_success = sfn.Succeed(
            self,
            "SucceededWithLeaseRetained",
            comment=(
                "The pipeline succeeded and was notified, but its concurrency slot is "
                "still held because a Neo compilation job it created has not finished. "
                "Reconciliation releases the slot once that child is terminal."
            ),
        )
        check_release = sfn.Choice(self, "CheckLeaseReleased")
        check_release.when(
            sfn.Condition.boolean_equals("$.release.released", True),
            succeeded,
        )
        check_release.otherwise(lease_retained_after_success)

        check_status.when(
            sfn.Condition.string_equals("$.status.PipelineExecutionStatus", "Succeeded"),
            release_lease.next(notify_success).next(check_release),
        )
        # Poll budget exhausted while still nonterminal: stop it and confirm a
        # terminal status BEFORE releasing.
        check_status.when(is_nonterminal, stop_pipeline)
        check_status.otherwise(release_lease_on_failure.next(pipeline_failed))

        stop_pipeline.next(init_stop_counter)
        init_stop_counter.next(wait_for_stop)
        wait_for_stop.next(describe_after_stop)
        describe_after_stop.next(increment_stop_counter)
        check_stopped = sfn.Choice(self, "CheckStoppedStatus")
        increment_stop_counter.next(check_stopped)
        check_stopped.when(
            sfn.Condition.and_(
                is_nonterminal,
                sfn.Condition.number_less_than("$.stopWait.count", max_stop_wait_attempts),
            ),
            wait_for_stop,
        )
        # Will not stop: retain the lease rather than free a slot whose pipeline
        # is demonstrably still consuming resources. Uses the dedicated Fail state
        # because $.error is absent on this route.
        check_stopped.when(is_nonterminal, stop_budget_exhausted)
        check_stopped.otherwise(release_lease_on_failure)

        state_machine = sfn.StateMachine(
            self,
            "PipelineOrchestrator",
            state_machine_name=STATE_MACHINE_NAME,
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            # Retained execution history for the operational questions this
            # design creates: which leases were retained, and why.
            logs=sfn.LogOptions(
                destination=logs.LogGroup(
                    self,
                    "OrchestratorLogs",
                    retention=logs.RetentionDays.ONE_MONTH,
                    removal_policy=RemovalPolicy.DESTROY,
                ),
                level=sfn.LogLevel.ERROR,
                include_execution_data=False,
            ),
            tracing_enabled=True,
        )

        events.Rule(
            self,
            "ExecutionFailureRule",
            rule_name=FAILURE_RULE_NAME,
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

        reconcile_rule = events.Rule(
            self,
            "LeaseReconcileSchedule",
            rule_name=f"{FAILURE_RULE_NAME}-reconcile",
            schedule=events.Schedule.rate(Duration.hours(1)),
        )
        reconcile_rule.add_target(
            targets.LambdaFunction(
                lease_fn,
                event=events.RuleTargetInput.from_object({"action": "reconcile"}),
            )
        )
        events.Rule(
            self,
            "LeaseReconcileOnAbnormalEnd",
            rule_name=f"{FAILURE_RULE_NAME}-reconcile-now",
            event_pattern=events.EventPattern(
                source=["aws.states"],
                detail_type=["Step Functions Execution Status Change"],
                detail={
                    "status": ["FAILED", "TIMED_OUT", "ABORTED"],
                    "stateMachineArn": [state_machine.state_machine_arn],
                },
            ),
            targets=[
                targets.LambdaFunction(
                    lease_fn,
                    event=events.RuleTargetInput.from_object({
                        "action": "reconcile",
                        "ExecutionArn": events.EventField.from_path(
                            "$.detail.executionArn"
                        ),
                    }),
                )
            ],
        )

        # --- Lambda: SQS -> Step Functions ---
        trigger_fn = _lambda.Function(
            self,
            "QueueTrigger",
            function_name=QUEUE_TRIGGER_FUNCTION_NAME,
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="queue_trigger.handler",
            code=_lambda.Code.from_asset(
                "src/orchestration", exclude=["**/__pycache__/**", "**/*.pyc"],
            ),
            timeout=Duration.seconds(30),
            environment={
                "STATE_MACHINE_ARN": state_machine.state_machine_arn,
            },
            log_group=logs.LogGroup(
                self,
                "QueueTriggerLogs",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            ),
        )
        state_machine.grant_start_execution(trigger_fn)
        trigger_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["states:DescribeExecution"],
                resources=[
                    f"arn:aws:states:{self.region}:{self.account}:execution:"
                    f"{STATE_MACHINE_NAME}:*"
                ],
            )
        )
        trigger_fn.add_event_source(
            event_sources.SqsEventSource(queue, batch_size=1)
        )

        # --- Outputs ---
        CfnOutput(self, OUTPUT_QUEUE_URL, value=queue.queue_url)
        CfnOutput(self, OUTPUT_DLQ_URL, value=dlq.queue_url)
        CfnOutput(self, OUTPUT_STATE_MACHINE_ARN, value=state_machine.state_machine_arn)
        CfnOutput(self, OUTPUT_TOPIC_ARN, value=topic.topic_arn)
        CfnOutput(self, OUTPUT_LEASE_TABLE_NAME, value=lease_table.table_name)
        # The recovery runbook in README.md reads these. It previously told readers
        # to use $LEASE_FUNCTION and $TRIGGER_FUNCTION without any way to obtain
        # them, so its commands could not be run as written.
        CfnOutput(self, OUTPUT_LEASE_FUNCTION_NAME, value=lease_fn.function_name)
        CfnOutput(self, OUTPUT_QUEUE_TRIGGER_FUNCTION_NAME,
                  value=trigger_fn.function_name)
