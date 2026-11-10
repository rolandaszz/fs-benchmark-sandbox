"""Durable Temporal LIFO rollback saga for mitigating VFS online store SLA breaches."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

TASK_QUEUE_NAME = "vfs-sre-mitigation-queue"


@dataclass
class SagaInput:
    """Parameters passed into EngineMitigationSagaWorkflow."""

    target_engine: str
    fallback_engine: str = "pgvector"
    p99_latency_ms: float = 0.0
    simulate_quarantine_failure: bool = False


@dataclass
class SagaExecutionResult:
    """Structured outcome returned by EngineMitigationSagaWorkflow."""

    status: str
    executed_steps: list[str] = field(default_factory=list)
    compensated_steps: list[str] = field(default_factory=list)


class SagaStepError(RuntimeError):
    """Raised when an activity in the mitigation saga fails and triggers LIFO compensation."""


@activity.defn(name="shift_proxy_routing_weight")
async def shift_proxy_routing_weight(engine: str) -> str:
    return f"shifted_traffic_to:{engine}"


@activity.defn(name="restore_proxy_routing_weight")
async def restore_proxy_routing_weight(engine: str) -> str:
    return f"restored_traffic_to:{engine}"


@activity.defn(name="snapshot_engine_diagnostics")
async def snapshot_engine_diagnostics(engine: str) -> str:
    return f"captured_diagnostics:{engine}"


@activity.defn(name="purge_partial_diagnostics")
async def purge_partial_diagnostics(engine: str) -> str:
    return f"purged_diagnostics:{engine}"


@activity.defn(name="quarantine_degraded_engine")
async def quarantine_degraded_engine(inp: SagaInput) -> str:
    if inp.simulate_quarantine_failure:
        raise SagaStepError(f"quarantine lock rejected for {inp.target_engine}")
    return f"quarantined:{inp.target_engine}"


@workflow.defn(name="EngineMitigationSagaWorkflow")
class EngineMitigationSagaWorkflow:
    """Executes traffic drain, telemetry snapshot, and engine quarantine with LIFO compensations."""

    @workflow.run
    async def run(self, inp: SagaInput) -> SagaExecutionResult:
        executed: list[str] = []
        compensations: list[tuple[Callable[[str], Coroutine[Any, Any, str]], str]] = []
        compensated: list[str] = []
        retry_policy_single = RetryPolicy(maximum_attempts=1)

        try:
            step1 = await workflow.execute_activity(
                shift_proxy_routing_weight,
                inp.fallback_engine,
                start_to_close_timeout=timedelta(seconds=5),
                retry_policy=retry_policy_single,
            )
            executed.append(step1)
            compensations.append((restore_proxy_routing_weight, inp.target_engine))

            step2 = await workflow.execute_activity(
                snapshot_engine_diagnostics,
                inp.target_engine,
                start_to_close_timeout=timedelta(seconds=5),
                retry_policy=retry_policy_single,
            )
            executed.append(step2)
            compensations.append((purge_partial_diagnostics, inp.target_engine))

            step3 = await workflow.execute_activity(
                quarantine_degraded_engine,
                inp,
                start_to_close_timeout=timedelta(seconds=5),
                retry_policy=retry_policy_single,
            )
            executed.append(step3)
            return SagaExecutionResult(status="mitigated", executed_steps=executed)
        except Exception:
            for comp_activity_fn, arg in reversed(compensations):
                comp_res = await workflow.execute_activity(
                    comp_activity_fn,
                    arg,
                    start_to_close_timeout=timedelta(seconds=5),
                    retry_policy=retry_policy_single,
                )
                compensated.append(comp_res)
            return SagaExecutionResult(
                status="compensated_rollback",
                executed_steps=executed,
                compensated_steps=compensated,
            )


ALL_SAGA_ACTIVITIES = [
    shift_proxy_routing_weight,
    restore_proxy_routing_weight,
    snapshot_engine_diagnostics,
    purge_partial_diagnostics,
    quarantine_degraded_engine,
]


async def execute_mitigation_saga(inp: SagaInput) -> SagaExecutionResult:
    """Run EngineMitigationSagaWorkflow in Temporal time-skipping memory mode or Docker cluster mode."""
    cluster_mode = os.getenv("TEMPORAL_CLUSTER_MODE", "in_memory").strip().lower()
    workflow_id = f"vfs-mitigation-{inp.target_engine}-{uuid.uuid4().hex[:8]}"

    if cluster_mode == "docker":
        target_host = os.getenv("TEMPORAL_TARGET_HOST", "localhost:7233")
        client = await Client.connect(target_host)
        async with Worker(
            client,
            task_queue=TASK_QUEUE_NAME,
            workflows=[EngineMitigationSagaWorkflow],
            activities=ALL_SAGA_ACTIVITIES,
        ):
            return await client.execute_workflow(
                EngineMitigationSagaWorkflow.run,
                inp,
                id=workflow_id,
                task_queue=TASK_QUEUE_NAME,
            )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE_NAME,
            workflows=[EngineMitigationSagaWorkflow],
            activities=ALL_SAGA_ACTIVITIES,
        ):
            return await env.client.execute_workflow(
                EngineMitigationSagaWorkflow.run,
                inp,
                id=workflow_id,
                task_queue=TASK_QUEUE_NAME,
            )
