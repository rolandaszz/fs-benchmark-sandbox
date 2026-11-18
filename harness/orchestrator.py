"""Sequential single-target and multi-engine benchmark conductor and closed-loop Sense -> Triage -> Actuate runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field

from harness.chaos_agent import ChaosAgent, NetemProfile
from harness.report_generator import write_suite_artifacts
from harness.sre_evaluator import DecisionVerdict, MitigationAction, SREEvaluator
from harness.temporal_saga import SagaExecutionResult, SagaInput, execute_mitigation_saga
from harness.traffic_agent import TrafficAgent, TrafficSummary

ALL_ENGINES: tuple[str, ...] = ("pgvector", "qdrant", "vespa", "milvus")


@dataclass
class OrchestratorRunReport:
    """End-to-end summary of a single-engine benchmark lifecycle."""

    engine: str
    seeded_items: int
    traffic: TrafficSummary
    verdict: dict[str, object]
    saga_result: SagaExecutionResult | None
    telemetry: dict[str, object] = field(default_factory=dict)


@dataclass
class MultiEngineSuiteReport:
    """Aggregated comparative summary across all candidate vector engines."""

    runs: list[OrchestratorRunReport]
    fastest_rps_engine: str
    lowest_p99_engine: str
    all_sla_compliant: bool


class BenchmarkOrchestrator:
    """Coordinates sequential engine seeding, load generation, telemetry triage, and saga actuation."""

    def __init__(
        self,
        engine: str = "pgvector",
        proxy_url: str = "http://localhost:8080",
        prometheus_url: str = "http://localhost:9090",
        concurrency: int = 24,
        dry_run_chaos: bool = True,
    ) -> None:
        self.engine = engine
        self.proxy_url = proxy_url
        self.prometheus_url = prometheus_url
        self.concurrency = concurrency
        self.dry_run_chaos = dry_run_chaos
        self.traffic_agent = TrafficAgent(
            proxy_url=proxy_url,
            target_engine=engine,
            concurrency=concurrency,
        )
        self.evaluator = SREEvaluator(prometheus_url=prometheus_url)
        self.chaos_agent = ChaosAgent(dry_run=dry_run_chaos)

    async def execute_single_target_run(
        self,
        seed_count: int = 200,
        request_count: int = 200,
        skip_seed: bool = False,
        chaos_delay_ms: int = 0,
    ) -> OrchestratorRunReport:
        self.chaos_agent.drop_os_page_cache()
        seeded = seed_count if skip_seed else await self.traffic_agent.seed_catalog(item_count=seed_count)
        if not skip_seed and seed_count > 0:
            # Give Prometheus sliding window buffer to isolate online traffic from bulk batches
            await asyncio.sleep(2)

        if chaos_delay_ms > 0:
            self.chaos_agent.apply_netem(NetemProfile(delay_ms=chaos_delay_ms, jitter_ms=max(1, chaos_delay_ms // 5)))

        try:
            traffic_summary = await self.traffic_agent.run_workload_mix(
                total_requests=request_count,
                catalog_size=seed_count,
            )
            snapshot = await self.evaluator.query_prometheus_snapshot(engine=self.engine)
            if chaos_delay_ms > 0 and self.dry_run_chaos:
                snapshot.p99_latency_ms += float(chaos_delay_ms)
                snapshot.p95_latency_ms += float(chaos_delay_ms) * 0.8
        finally:
            if chaos_delay_ms > 0:
                self.chaos_agent.clear_netem()

        verdict: DecisionVerdict = await self.evaluator.evaluate(snapshot)

        saga_outcome: SagaExecutionResult | None = None
        if verdict.action == MitigationAction.THROTTLE_CONCURRENCY:
            self.traffic_agent.throttle(0.5)
        elif verdict.action in {MitigationAction.TRIGGER_TEMPORAL_SAGA, MitigationAction.ABORT_ENGINE_RUN}:
            fallback_engine = "qdrant" if self.engine == "pgvector" else "pgvector"
            if self.engine in {"vespa", "milvus"}:
                fallback_engine = "qdrant"
            saga_outcome = await execute_mitigation_saga(
                SagaInput(
                    target_engine=self.engine,
                    fallback_engine=fallback_engine,
                    p99_latency_ms=snapshot.p99_latency_ms,
                )
            )

        return OrchestratorRunReport(
            engine=self.engine,
            seeded_items=seeded,
            traffic=traffic_summary,
            verdict=verdict.model_dump(),
            saga_result=saga_outcome,
            telemetry=snapshot.model_dump(),
        )

    async def execute_multi_engine_suite(
        self,
        engines: tuple[str, ...] = ALL_ENGINES,
        seed_count: int = 200,
        request_count: int = 200,
        skip_seed: bool = False,
        cooldown_seconds: float = 2.0,
        chaos_delay_ms: int = 0,
    ) -> MultiEngineSuiteReport:
        """Execute sequential benchmark runs across all target engines with page-cache isolation."""
        reports: list[OrchestratorRunReport] = []
        for idx, eng in enumerate(engines):
            runner = BenchmarkOrchestrator(
                engine=eng,
                proxy_url=self.proxy_url,
                prometheus_url=self.prometheus_url,
                concurrency=self.concurrency,
                dry_run_chaos=self.dry_run_chaos,
            )
            report = await runner.execute_single_target_run(
                seed_count=seed_count,
                request_count=request_count,
                skip_seed=skip_seed,
                chaos_delay_ms=chaos_delay_ms,
            )
            reports.append(report)
            if idx < len(engines) - 1 and cooldown_seconds > 0:
                await asyncio.sleep(cooldown_seconds)

        fastest_rps_engine = max(reports, key=lambda r: r.traffic.achieved_rps).engine if reports else ""
        lowest_p99_engine = (
            min(reports, key=lambda r: float(r.telemetry.get("p99_latency_ms", float("inf")))).engine
            if reports
            else ""
        )
        all_sla_compliant = all(not bool(r.verdict.get("is_sla_breached", False)) for r in reports)

        return MultiEngineSuiteReport(
            runs=reports,
            fastest_rps_engine=fastest_rps_engine,
            lowest_p99_engine=lowest_p99_engine,
            all_sla_compliant=all_sla_compliant,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-target or multi-engine VFS benchmark lifecycle.")
    parser.add_argument("--engine", default="pgvector", choices=list(ALL_ENGINES))
    parser.add_argument("--all-engines", action="store_true", help="Sequentially benchmark all 4 target engines")
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--skip-seed", action="store_true", help="Skip re-seeding if catalog is already populated")
    parser.add_argument("--chaos-delay-ms", type=int, default=0, help="Inject tc netem delay_ms fault during run")
    parser.add_argument("--report-out", default="", help="Optional path to write Markdown synthesis report")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON suite report")
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(engine=args.engine, concurrency=args.concurrency)
    if args.all_engines:
        suite_report = asyncio.run(
            orchestrator.execute_multi_engine_suite(
                seed_count=args.seed_count,
                request_count=args.requests,
                skip_seed=args.skip_seed,
                chaos_delay_ms=args.chaos_delay_ms,
            )
        )
        if args.report_out:
            write_suite_artifacts(
                suite_report,
                markdown_path=args.report_out,
                json_path=args.json_out or None,
            )
        print(json.dumps(asdict(suite_report), indent=2))
    else:
        report = asyncio.run(
            orchestrator.execute_single_target_run(
                seed_count=args.seed_count,
                request_count=args.requests,
                skip_seed=args.skip_seed,
                chaos_delay_ms=args.chaos_delay_ms,
            )
        )
        print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
