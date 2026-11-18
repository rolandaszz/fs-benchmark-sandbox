"""Sequential single-target benchmark conductor and closed-loop Sense -> Triage -> Actuate runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass

from harness.chaos_agent import ChaosAgent
from harness.sre_evaluator import DecisionVerdict, MitigationAction, SREEvaluator
from harness.temporal_saga import SagaExecutionResult, SagaInput, execute_mitigation_saga
from harness.traffic_agent import TrafficAgent, TrafficSummary


@dataclass
class OrchestratorRunReport:
    """End-to-end summary of a single-engine benchmark lifecycle."""

    engine: str
    seeded_items: int
    traffic: TrafficSummary
    verdict: dict[str, object]
    saga_result: SagaExecutionResult | None


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
    ) -> OrchestratorRunReport:
        self.chaos_agent.drop_os_page_cache()
        seeded = seed_count if skip_seed else await self.traffic_agent.seed_catalog(item_count=seed_count)
        if not skip_seed and seed_count > 0:
            # Give Prometheus sliding window buffer to isolate online traffic from bulk batches
            await asyncio.sleep(2)
        traffic_summary = await self.traffic_agent.run_workload_mix(
            total_requests=request_count,
            catalog_size=seed_count,
        )

        snapshot = await self.evaluator.query_prometheus_snapshot(engine=self.engine)
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
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-target VFS benchmark lifecycle.")
    parser.add_argument("--engine", default="pgvector", choices=["pgvector", "qdrant", "vespa", "milvus"])
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--skip-seed", action="store_true", help="Skip re-seeding if catalog is already populated")
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(engine=args.engine, concurrency=args.concurrency)
    report = asyncio.run(
        orchestrator.execute_single_target_run(
            seed_count=args.seed_count,
            request_count=args.requests,
            skip_seed=args.skip_seed,
        )
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
