# Vector Feature Store (VFS) Multi-Engine Benchmark Report

## Executive Summary

- **Fastest Throughput Engine**: `qdrant`
- **Lowest Tail-Latency (p99) Engine**: `qdrant`
- **SLA Target**: `p99 <= 25.0 ms`
- **Suite SLA Status**: **BREACH DETECTED (Mitigation Actuated)**

## Comparative Performance & SRE Triage Matrix

| Engine | Achieved RPS | p50 (ms) | p95 (ms) | p99 (ms) | Error Rate | Stability (1-5) | SRE Action | Saga Mitigation |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :--- | :--- |
| `pgvector` | 230.04 | 3.75 | 8.08 | 11.81 | 0.00% | 5 | `PROCEED` | `None` |
| `qdrant` | 275.24 | 1.68 | 3.88 | 6.70 | 0.00% | 5 | `PROCEED` | `None` |
| `vespa` | 176.55 | 9.69 | 37.19 | 48.12 | 0.00% | 2 | `TRIGGER_TEMPORAL_SAGA` | `shifted_traffic_to:qdrant -> captured_diagnostics:vespa -> quarantined:vespa` |
| `milvus` | 185.23 | 7.03 | 16.23 | 26.30 | 0.00% | 3 | `THROTTLE_CONCURRENCY` | `None` |

## Workload & Architectural Notes

- **Workload Mix**: 80% filtered ANN cosine search (`price_eur <= 50.0`, `seller_country`), 15% real-time batch upsert, 5% point lookup.
- **Isolation**: Sequential target execution with OS page-cache drop (`sync && drop_caches`) and cooldown window between engines.
- **Out-of-Band Control Plane**: Prometheus 1s histogram scrape evaluated via OpenAI Decisions API (`gpt-6-luna`) contract with Temporal LIFO Saga mitigation.
