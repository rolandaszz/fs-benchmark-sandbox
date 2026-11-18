from __future__ import annotations

import pytest

from harness import traffic_agent
from harness.traffic_agent import TrafficAgent


@pytest.mark.asyncio
async def test_traffic_agent_uses_prior_knowledge_h2c(monkeypatch: pytest.MonkeyPatch) -> None:
    client_options: list[dict[str, object]] = []

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            client_options.append(kwargs)

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(traffic_agent.httpx, "AsyncClient", FakeAsyncClient)
    agent = TrafficAgent()

    await agent.seed_catalog(item_count=0)
    await agent.run_workload_mix(total_requests=0)

    assert len(client_options) == 2
    assert all(options["http1"] is False for options in client_options)
    assert all(options["http2"] is True for options in client_options)
