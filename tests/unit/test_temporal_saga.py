"""Unit tests verifying Temporal in-memory saga execution and LIFO compensation order."""

from __future__ import annotations

import pytest

from harness.temporal_saga import SagaInput, execute_mitigation_saga


@pytest.mark.asyncio
async def test_temporal_saga_happy_path_in_memory() -> None:
    result = await execute_mitigation_saga(
        SagaInput(
            target_engine="qdrant",
            fallback_engine="pgvector",
            p99_latency_ms=48.2,
            simulate_quarantine_failure=False,
        )
    )
    assert result.status == "mitigated"
    assert result.executed_steps == [
        "shifted_traffic_to:pgvector",
        "captured_diagnostics:qdrant",
        "quarantined:qdrant",
    ]
    assert result.compensated_steps == []


@pytest.mark.asyncio
async def test_temporal_saga_lifo_compensation_on_failure() -> None:
    result = await execute_mitigation_saga(
        SagaInput(
            target_engine="vespa",
            fallback_engine="pgvector",
            p99_latency_ms=72.0,
            simulate_quarantine_failure=True,
        )
    )
    assert result.status == "compensated_rollback"
    assert result.executed_steps == [
        "shifted_traffic_to:pgvector",
        "captured_diagnostics:vespa",
    ]
    assert result.compensated_steps == [
        "purged_diagnostics:vespa",
        "restored_traffic_to:vespa",
    ]
