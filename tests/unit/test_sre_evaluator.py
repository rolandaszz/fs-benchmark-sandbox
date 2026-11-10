"""Unit tests for the OpenAI Responses API payload builder and SRE triage verdicts."""

from __future__ import annotations

import json

import httpx
import pytest

from harness.sre_evaluator import (
    MitigationAction,
    SREEvaluator,
    TelemetrySnapshot,
    build_responses_api_payload,
    evaluate_locally,
)


def test_responses_api_payload_contains_structured_json_schema() -> None:
    snap = TelemetrySnapshot(engine="pgvector", p99_latency_ms=18.4, rps=4200.0, error_rate=0.0)
    payload = build_responses_api_payload(snap)

    assert payload["model"] == "gpt-4o-mini"
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    props = fmt["schema"]["properties"]
    assert props["is_sla_breached"]["type"] == "boolean"
    assert props["action"]["type"] == "string"
    assert set(props["action"]["enum"]) == {m.value for m in MitigationAction}
    assert props["stability_score"]["type"] == "integer"
    assert props["stability_score"]["minimum"] == 1
    assert props["stability_score"]["maximum"] == 5


def test_local_evaluation_respects_25ms_sla_boundary() -> None:
    nominal = TelemetrySnapshot(engine="pgvector", p99_latency_ms=21.0, rps=5000.0, error_rate=0.0)
    verdict_ok = evaluate_locally(nominal)
    assert verdict_ok.is_sla_breached is False
    assert verdict_ok.action == MitigationAction.PROCEED
    assert verdict_ok.stability_score >= 4

    mild_breach = TelemetrySnapshot(engine="pgvector", p99_latency_ms=31.5, rps=5000.0, error_rate=0.0)
    verdict_throttle = evaluate_locally(mild_breach)
    assert verdict_throttle.is_sla_breached is True
    assert verdict_throttle.action == MitigationAction.THROTTLE_CONCURRENCY

    severe_breach = TelemetrySnapshot(engine="pgvector", p99_latency_ms=55.0, rps=5000.0, error_rate=0.04)
    verdict_saga = evaluate_locally(severe_breach)
    assert verdict_saga.is_sla_breached is True
    assert verdict_saga.action == MitigationAction.TRIGGER_TEMPORAL_SAGA
    assert verdict_saga.stability_score == 2


@pytest.mark.asyncio
async def test_sre_evaluator_parses_openai_responses_verdict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        verdict_json = json.dumps(
            {
                "is_sla_breached": True,
                "action": "TRIGGER_TEMPORAL_SAGA",
                "stability_score": 2,
            }
        )
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": verdict_json,
                            }
                        ],
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        evaluator = SREEvaluator(
            api_key="sk-test-valid-key",
            openai_endpoint="https://api.openai.com/v1/responses",
            http_client=mock_client,
        )
        snap = TelemetrySnapshot(engine="pgvector", p99_latency_ms=42.0, rps=3100.0, error_rate=0.02)
        verdict = await evaluator.evaluate(snap)

    assert verdict.is_sla_breached is True
    assert verdict.action == MitigationAction.TRIGGER_TEMPORAL_SAGA
    assert verdict.stability_score == 2
    assert verdict.source == "gpt-4o-mini"
