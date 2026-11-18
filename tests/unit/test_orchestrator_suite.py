"""Unit tests for sequential multi-engine benchmark orchestration, report generation, and chaos drills."""

from __future__ import annotations

from pathlib import Path
import pytest

from harness.orchestrator import ALL_ENGINES, BenchmarkOrchestrator, OrchestratorRunReport
from harness.report_generator import render_markdown_report, write_suite_artifacts
from harness.sre_evaluator import TelemetrySnapshot
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
        chaos_delay_ms: int = 0,
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


@pytest.mark.asyncio
async def test_markdown_report_generation_and_artifact_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_single_run(
        self: BenchmarkOrchestrator,
        seed_count: int = 200,
        request_count: int = 200,
        skip_seed: bool = False,
        chaos_delay_ms: int = 0,
    ) -> OrchestratorRunReport:
        return OrchestratorRunReport(
            engine=self.engine,
            seeded_items=seed_count,
            traffic=TrafficSummary(
                total_requests=request_count,
                succeeded=request_count,
                failed=0,
                elapsed_seconds=1.0,
                achieved_rps=250.0,
            ),
            verdict={
                "is_sla_breached": False,
                "action": "PROCEED",
                "stability_score": 5,
                "source": "deterministic_fallback",
            },
            saga_result=None,
            telemetry={
                "engine": self.engine,
                "p50_latency_ms": 2.0,
                "p95_latency_ms": 4.5,
                "p99_latency_ms": 9.2,
                "rps": 250.0,
                "error_rate": 0.0,
            },
        )

    monkeypatch.setattr(BenchmarkOrchestrator, "execute_single_target_run", fake_single_run)
    conductor = BenchmarkOrchestrator(concurrency=8)
    suite = await conductor.execute_multi_engine_suite(engines=("qdrant", "milvus"), cooldown_seconds=0.0)

    md_out = tmp_path / "benchmark_report.md"
    json_out = tmp_path / "benchmark_report.json"
    write_suite_artifacts(suite, markdown_path=md_out, json_path=json_out)

    content = md_out.read_text(encoding="utf-8")
    assert "# Vector Feature Store (VFS) Multi-Engine Benchmark Report" in content
    assert "| `qdrant` |" in content
    assert "| `milvus` |" in content
    assert json_out.exists()


@pytest.mark.asyncio
async def test_chaos_delay_triggers_temporal_saga_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    conductor = BenchmarkOrchestrator(engine="milvus", concurrency=4, dry_run_chaos=True)

    async def fake_workload(total_requests: int = 200, catalog_size: int = 200) -> TrafficSummary:
        return TrafficSummary(
            total_requests=total_requests,
            succeeded=total_requests,
            failed=0,
            elapsed_seconds=1.0,
            achieved_rps=200.0,
        )

    async def fake_snapshot(engine: str = "milvus") -> TelemetrySnapshot:
        return TelemetrySnapshot(
            engine=engine,
            p50_latency_ms=2.0,
            p95_latency_ms=4.5,
            p99_latency_ms=10.0,
            rps=200.0,
            error_rate=0.0,
        )

    monkeypatch.setattr(conductor.traffic_agent, "run_workload_mix", fake_workload)
    monkeypatch.setattr(conductor.evaluator, "query_prometheus_snapshot", fake_snapshot)

    report = await conductor.execute_single_target_run(
        seed_count=100,
        request_count=100,
        skip_seed=True,
        chaos_delay_ms=45,
    )

    assert report.verdict["is_sla_breached"] is True
    assert report.verdict["action"] == "TRIGGER_TEMPORAL_SAGA"
    assert report.saga_result is not None
    assert "shifted_traffic_to:qdrant" in report.saga_result.executed_steps
