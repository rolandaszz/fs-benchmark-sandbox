"""Out-of-band SRE evaluator integrating Prometheus p99 telemetry with the OpenAI Decisions API."""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel, Field

SLA_P99_THRESHOLD_MS: float = 25.0
DEFAULT_DECISIONS_ENDPOINT = "https://api.openai.com/v1/decisions"
DEFAULT_DECISIONS_MODEL = "gpt-6-luna"


class MitigationAction(StrEnum):
    """Closed enumeration of SRE mitigation actions emitted by the decision layer."""

    PROCEED = "PROCEED"
    THROTTLE_CONCURRENCY = "THROTTLE_CONCURRENCY"
    TRIGGER_TEMPORAL_SAGA = "TRIGGER_TEMPORAL_SAGA"
    ABORT_ENGINE_RUN = "ABORT_ENGINE_RUN"


class TelemetrySnapshot(BaseModel):
    """Point-in-time golden signal snapshot sampled from Prometheus."""

    engine: str
    p99_latency_ms: float = Field(ge=0.0)
    rps: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    cpu_throttled_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class DecisionVerdict(BaseModel):
    """Typed non-autoregressive decision primitives returned by gpt-6-luna."""

    is_sla_breached: bool
    action: MitigationAction
    stability_score: int = Field(ge=1, le=5)
    source: str = "deterministic_fallback"


def build_decisions_api_payload(
    snapshot: TelemetrySnapshot,
    model: str = DEFAULT_DECISIONS_MODEL,
) -> dict[str, Any]:
    """Construct the structured OpenAI Decisions API request body with predicate, choice, and score."""
    return {
        "model": model,
        "context": {
            "service": "vfs-online-store",
            "sla_p99_limit_ms": SLA_P99_THRESHOLD_MS,
            "telemetry": snapshot.model_dump(),
        },
        "decisions": {
            "is_sla_breached": {
                "type": "predicate",
                "description": "True when p99_latency_ms exceeds 25.0ms or error_rate exceeds 0.01.",
            },
            "action": {
                "type": "choice",
                "options": [member.value for member in MitigationAction],
                "description": "Bounded SRE control-plane action for the active vector engine.",
            },
            "stability_score": {
                "type": "score",
                "min": 1,
                "max": 5,
                "description": "Integer operational health score from 1 (critical breach) to 5 (nominal).",
            },
        },
    }


def evaluate_locally(snapshot: TelemetrySnapshot) -> DecisionVerdict:
    """Deterministic fallback policy matching the Decisions API contract when offline or unauthenticated."""
    breached = snapshot.p99_latency_ms > SLA_P99_THRESHOLD_MS or snapshot.error_rate > 0.01
    if not breached:
        score = 5 if snapshot.p99_latency_ms <= 15.0 else 4
        return DecisionVerdict(
            is_sla_breached=False,
            action=MitigationAction.PROCEED,
            stability_score=score,
            source="deterministic_fallback",
        )

    if snapshot.p99_latency_ms >= 100.0 or snapshot.error_rate >= 0.10:
        return DecisionVerdict(
            is_sla_breached=True,
            action=MitigationAction.ABORT_ENGINE_RUN,
            stability_score=1,
            source="deterministic_fallback",
        )

    if snapshot.p99_latency_ms > 40.0 or snapshot.error_rate >= 0.03:
        return DecisionVerdict(
            is_sla_breached=True,
            action=MitigationAction.TRIGGER_TEMPORAL_SAGA,
            stability_score=2,
            source="deterministic_fallback",
        )

    return DecisionVerdict(
        is_sla_breached=True,
        action=MitigationAction.THROTTLE_CONCURRENCY,
        stability_score=3,
        source="deterministic_fallback",
    )


class SREEvaluator:
    """Asynchronous out-of-band telemetry sampler and Decisions API client."""

    def __init__(
        self,
        prometheus_url: str | None = None,
        api_key: str | None = None,
        decisions_endpoint: str | None = None,
        model: str = DEFAULT_DECISIONS_MODEL,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.prometheus_url = (prometheus_url or os.getenv("PROMETHEUS_ENDPOINT", "http://localhost:9090")).rstrip("/")
        raw_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.api_key = "" if raw_key in {"", "your_openai_api_key_here"} else raw_key
        self.decisions_endpoint = decisions_endpoint or os.getenv(
            "OPENAI_DECISIONS_ENDPOINT",
            DEFAULT_DECISIONS_ENDPOINT,
        )
        self.model = model
        self._external_client = http_client

    async def query_prometheus_snapshot(self, engine: str = "pgvector") -> TelemetrySnapshot:
        """Query 30-second sliding p99 latency, RPS, and error ratio from Prometheus."""
        p99_query = (
            f'histogram_quantile(0.99, sum(rate(vfs_request_duration_seconds_bucket{{engine="{engine}"}}[30s])) by (le))'
        )
        rps_query = f'sum(rate(vfs_request_duration_seconds_count{{engine="{engine}"}}[30s]))'
        err_query = (
            f'sum(rate(vfs_request_duration_seconds_count{{engine="{engine}",status="error"}}[30s])) / '
            f'clamp_min(sum(rate(vfs_request_duration_seconds_count{{engine="{engine}"}}[30s])), 1e-9)'
        )

        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=2.0)
        try:
            p99_sec = await self._scalar_promql(client, p99_query)
            rps = await self._scalar_promql(client, rps_query)
            err_rate = await self._scalar_promql(client, err_query)
        finally:
            if owns_client:
                await client.aclose()

        return TelemetrySnapshot(
            engine=engine,
            p99_latency_ms=max(0.0, p99_sec * 1000.0),
            rps=max(0.0, rps),
            error_rate=min(1.0, max(0.0, err_rate)),
        )

    async def _scalar_promql(self, client: httpx.AsyncClient, expr: str) -> float:
        resp = await client.get(f"{self.prometheus_url}/api/v1/query", params={"query": expr})
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("data", {}).get("result", [])
        if not results:
            return 0.0
        raw_val = results[0].get("value", [0, "0"])[1]
        try:
            parsed = float(raw_val)
        except (TypeError, ValueError):
            return 0.0
        if parsed != parsed:  # NaN check
            return 0.0
        return parsed

    async def evaluate(self, snapshot: TelemetrySnapshot) -> DecisionVerdict:
        """Evaluate telemetry via POST /v1/decisions on gpt-6-luna with deterministic fallback."""
        if not self.api_key:
            return evaluate_locally(snapshot)

        payload = build_decisions_api_payload(snapshot, model=self.model)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(timeout=0.5)
        try:
            resp = await client.post(self.decisions_endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            decisions = body.get("decisions", body)
            return DecisionVerdict(
                is_sla_breached=bool(decisions["is_sla_breached"]),
                action=MitigationAction(decisions["action"]),
                stability_score=int(decisions["stability_score"]),
                source=self.model,
            )
        except (httpx.HTTPError, KeyError, ValueError):
            return evaluate_locally(snapshot)
        finally:
            if owns_client:
                await client.aclose()
