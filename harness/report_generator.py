"""Executive Markdown and JSON report synthesizer for multi-engine VFS benchmark suites."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from harness.sre_evaluator import SLA_P99_THRESHOLD_MS

if TYPE_CHECKING:
    from harness.orchestrator import MultiEngineSuiteReport


def render_markdown_report(suite: MultiEngineSuiteReport) -> str:
    """Render a GitHub-Flavored Markdown comparative matrix and SRE triage summary."""
    compliance_badge = "PASS (All Within SLA)" if suite.all_sla_compliant else "BREACH DETECTED (Mitigation Actuated)"
    lines: list[str] = [
        "# Vector Feature Store (VFS) Multi-Engine Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- **Fastest Throughput Engine**: `{suite.fastest_rps_engine}`",
        f"- **Lowest Tail-Latency (p99) Engine**: `{suite.lowest_p99_engine}`",
        f"- **SLA Target**: `p99 <= {SLA_P99_THRESHOLD_MS:.1f} ms`",
        f"- **Suite SLA Status**: **{compliance_badge}**",
        "",
        "## Comparative Performance & SRE Triage Matrix",
        "",
        "| Engine | Achieved RPS | p50 (ms) | p95 (ms) | p99 (ms) | Error Rate | Stability (1-5) | SRE Action | Saga Mitigation |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :--- | :--- |",
    ]

    for run in suite.runs:
        tel = run.telemetry
        p50 = float(tel.get("p50_latency_ms", 0.0))
        p95 = float(tel.get("p95_latency_ms", 0.0))
        p99 = float(tel.get("p99_latency_ms", 0.0))
        err_rate = float(tel.get("error_rate", 0.0)) * 100.0
        score = run.verdict.get("stability_score", "-")
        action = str(run.verdict.get("action", "-"))
        saga_str = "None"
        if run.saga_result is not None and run.saga_result.executed_steps:
            saga_str = " -> ".join(run.saga_result.executed_steps)

        lines.append(
            f"| `{run.engine}` | {run.traffic.achieved_rps:.2f} | {p50:.2f} | {p95:.2f} | "
            f"{p99:.2f} | {err_rate:.2f}% | {score} | `{action}` | `{saga_str}` |"
        )

    lines.extend(
        [
            "",
            "## Workload & Architectural Notes",
            "",
            "- **Workload Mix**: 80% filtered ANN cosine search (`price_eur <= 50.0`, `seller_country`), 15% real-time batch upsert, 5% point lookup.",
            "- **Isolation**: Sequential target execution with OS page-cache drop (`sync && drop_caches`) and cooldown window between engines.",
            "- **Out-of-Band Control Plane**: Prometheus 1s histogram scrape evaluated via OpenAI Decisions API (`gpt-6-luna`) contract with Temporal LIFO Saga mitigation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_suite_artifacts(
    suite: MultiEngineSuiteReport,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    """Persist rendered Markdown report and optional JSON suite output to disk."""
    md_file = Path(markdown_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.write_text(render_markdown_report(suite), encoding="utf-8")

    if json_path is not None:
        js_file = Path(json_path)
        js_file.parent.mkdir(parents=True, exist_ok=True)
        js_file.write_text(json.dumps(asdict(suite), indent=2) + "\n", encoding="utf-8")
