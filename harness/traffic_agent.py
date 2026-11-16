"""Asynchronous ConnectRPC HTTP/2 workload generator implementing the 80/15/5 VFS query mix."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

EMBEDDING_DIM = 768
CATEGORIES = ("women_shoes", "vintage_jackets", "mens_knitwear", "kids_outerwear", "luxury_bags")
COUNTRIES = ("FR", "ES", "LT", "DE", "PL", "IT")


def deterministic_unit_vector(seed: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """Generate a deterministic L2-normalized 768-d float vector."""
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [round(v / norm, 6) for v in raw]


@dataclass
class TrafficSummary:
    """Aggregated statistics from a synthetic benchmark burst."""

    total_requests: int
    succeeded: int
    failed: int
    elapsed_seconds: float
    achieved_rps: float


class TrafficAgent:
    """Drives ConnectRPC JSON-over-HTTP/2 requests against the Go L7 reverse proxy."""

    def __init__(
        self,
        proxy_url: str = "http://localhost:8080",
        target_engine: str = "pgvector",
        concurrency: int = 32,
    ) -> None:
        self.proxy_url = proxy_url.rstrip("/")
        self.target_engine = target_engine
        self.concurrency = max(1, concurrency)

    def throttle(self, factor: float = 0.5) -> int:
        """Reduce active worker concurrency when instructed by the SRE evaluator."""
        self.concurrency = max(1, int(self.concurrency * factor))
        return self.concurrency

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Connect-Protocol-Version": "1",
            "X-Target-Engine": self.target_engine,
        }

    async def seed_catalog(self, item_count: int = 250, batch_size: int = 50) -> int:
        """Populate the active engine through UpsertItemFeatures."""
        inserted = 0
        async with httpx.AsyncClient(http2=True, base_url=self.proxy_url, timeout=10.0) as client:
            for offset in range(0, item_count, batch_size):
                batch_items: list[dict[str, Any]] = []
                for idx in range(offset, min(offset + batch_size, item_count)):
                    batch_items.append(
                        {
                            "itemId": f"item-{idx:06d}",
                            "embedding": deterministic_unit_vector(idx),
                            "brandId": f"brand-{idx % 40:03d}",
                            "category": CATEGORIES[idx % len(CATEGORIES)],
                            "priceEur": round(5.0 + (idx % 95) * 1.25, 2),
                            "sellerCountry": COUNTRIES[idx % len(COUNTRIES)],
                            "updatedAtUnixMs": int(time.time() * 1000),
                        }
                    )
                resp = await client.post(
                    "/vfs.v1.FeatureStoreService/UpsertItemFeatures",
                    json={"items": batch_items},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                inserted += int(resp.json().get("upsertedCount", len(batch_items)))
        return inserted

    async def run_workload_mix(self, total_requests: int = 200, catalog_size: int = 250) -> TrafficSummary:
        """Execute 80% filtered ANN search, 15% real-time upsert, and 5% point lookup."""
        semaphore = asyncio.Semaphore(self.concurrency)
        succeeded = 0
        failed = 0
        start = time.perf_counter()

        async with httpx.AsyncClient(http2=True, base_url=self.proxy_url, timeout=5.0) as client:

            async def _one_request(req_idx: int) -> bool:
                async with semaphore:
                    bucket = req_idx % 20
                    try:
                        if bucket < 16:  # 80% Filtered ANN Retrieval
                            resp = await client.post(
                                "/vfs.v1.FeatureStoreService/SearchSimilarItems",
                                json={
                                    "queryEmbedding": deterministic_unit_vector(req_idx),
                                    "topK": 10,
                                    "maxPriceEur": 50.0,
                                    "sellerCountries": ["FR", "ES"],
                                },
                                headers=self._headers(),
                            )
                        elif bucket < 19:  # 15% Real-Time Ingestion
                            item_idx = req_idx % max(1, catalog_size)
                            resp = await client.post(
                                "/vfs.v1.FeatureStoreService/UpsertItemFeatures",
                                json={
                                    "items": [
                                        {
                                            "itemId": f"item-{item_idx:06d}",
                                            "embedding": deterministic_unit_vector(req_idx + 10_000),
                                            "brandId": f"brand-{item_idx % 40:03d}",
                                            "category": CATEGORIES[item_idx % len(CATEGORIES)],
                                            "priceEur": round(8.0 + (req_idx % 40), 2),
                                            "sellerCountry": "FR",
                                            "updatedAtUnixMs": int(time.time() * 1000),
                                        }
                                    ]
                                },
                                headers=self._headers(),
                            )
                        else:  # 5% Point Lookup
                            item_idx = req_idx % max(1, catalog_size)
                            resp = await client.post(
                                "/vfs.v1.FeatureStoreService/GetItemFeatures",
                                json={"itemId": f"item-{item_idx:06d}"},
                                headers=self._headers(),
                            )
                        return resp.status_code == 200
                    except httpx.HTTPError:
                        return False

            outcomes = await asyncio.gather(*(_one_request(i) for i in range(total_requests)))

        elapsed = max(1e-6, time.perf_counter() - start)
        for ok in outcomes:
            if ok:
                succeeded += 1
            else:
                failed += 1

        return TrafficSummary(
            total_requests=total_requests,
            succeeded=succeeded,
            failed=failed,
            elapsed_seconds=elapsed,
            achieved_rps=total_requests / elapsed,
        )
