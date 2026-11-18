"""Unit tests for sequential multi-engine benchmark orchestration and winner synthesis."""

from __future__ import annotations

import pytest

from harness.orchestrator import ALL_ENGINES, BenchmarkOrchestrator, OrchestratorRunReport
from harness.traffic_agent import TrafficSummary


@pytest.mark.asyncio
async def test_multi_engine_suite_aggregates_rankings_and_sla_compliance(monkeypatch: pytest.MonkeyPatch) -> None:
    rps_by_engine = {"pgvector": 180.0, "qdrant": 310.0, "vespa": 218.0, "milvus": 224.0}
    p99_by_engine = {"pgvector": 14.2, "qdrant": 6.8, "vespa": 18.5, "milvus": 11.4}

    async def fake_single_run(
        self: BenchmarkOrchestrator,
        seed_count: int = 200,
        request_count: int = 200,
        skip_seed: bool = False,
    ) -> OrchestratorRunReport:
        eng = self.engine
        return OrchestratorRunReport(
            engine=eng,
            seeded_items=seed_count,
            traffic=TrafficSummary(
                total_requests=request_count,
                succeeded=request_count,
                failed=0,
                elapsed_seconds=1.0,
                achieved_rps=rps_by_engine[eng],
            ),
            verdict={
                "is_sla_breached": False,
                "action": "PROCEED",
                "stability_score": 5,
                "source": "deterministic_fallback",
            },
            saga_result=None,
            telemetry={
                "engine": eng,
                "p50_latency_ms": 2.5,
                "p95_latency_ms": 5.0,
                "p99_latency_ms": p99_by_engine[eng],
                "rps": rps_by_engine[eng],
                "error_rate": 0.0,
                "cpu_throttled_ratio": 0.0,
            },
        )

    monkeypatch.setattr(BenchmarkOrchestrator, "execute_single_target_run", fake_single_run)

    conductor = BenchmarkOrchestrator(concurrency=8)
    suite = await conductor.execute_multi_engine_suite(
        engines=ALL_ENGINES,
        seed_count=100,
        request_count=200,
        skip_seed=True,
        cooldown_seconds=0.0,
    )

    assert [r.engine for r in suite.runs] == ["pgvector", "qdrant", "vespa", "milvus"]
    assert suite.fastest_rps_engine == "qdrant"
    assert suite.lowest_p99_engine == "qdrant"
    assert suite.all_sla_compliant is True
