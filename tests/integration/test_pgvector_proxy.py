"""End-to-end integration test against live Docker Compose pgvector, Go h2c proxy, and Prometheus."""

from __future__ import annotations

import os

import httpx
import pytest

from harness.traffic_agent import TrafficAgent

RUN_INTEGRATION = os.getenv("RUN_VFS_INTEGRATION", "0") == "1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_VFS_INTEGRATION=1 with docker compose stack running")
@pytest.mark.asyncio
async def test_end_to_end_pgvector_proxy_and_prometheus_scrape() -> None:
    agent = TrafficAgent(proxy_url="http://localhost:8080", target_engine="pgvector", concurrency=8)
    seeded = await agent.seed_catalog(item_count=40, batch_size=20)
    assert seeded == 40

    summary = await agent.run_workload_mix(total_requests=40, catalog_size=40)
    assert summary.failed == 0
    assert summary.succeeded == 40

    async with httpx.AsyncClient(timeout=3.0) as client:
        targets_resp = await client.get("http://localhost:9090/api/v1/targets")
        targets_resp.raise_for_status()
        active_targets = targets_resp.json()["data"]["activeTargets"]
        assert any(t["labels"].get("job") == "vfs-connect-proxy" and t["health"] == "up" for t in active_targets)
