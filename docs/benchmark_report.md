# Vector Feature Store (VFS) Multi-Engine Benchmark Report

## Summary

- **Fastest Throughput Engine**: `qdrant`
- **Lowest Tail-Latency (p99) Engine**: `qdrant`
- **Lowest Memory Footprint Engine**: `pgvector`
- **SLA Target**: `p99 <= 25.0 ms`
- **Suite SLA Status**: **PASS (All Within SLA)**

## Comparative Performance, Resource Footprint & SRE Triage Matrix

| Engine | Achieved RPS | p50 (ms) | p95 (ms) | p99 (ms) | Memory (MiB) | CPU (%) | Error Rate | Stability (1-5) | SRE Action | Saga Mitigation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- | :--- |
| `pgvector` | 287.34 | 3.46 | 4.90 | 7.65 | 138.9 | 0.1% | 0.00% | 5 | `PROCEED` | `None` |
| `qdrant` | 342.61 | 1.59 | 2.45 | 4.17 | 256.8 | 2.4% | 0.00% | 5 | `PROCEED` | `None` |
| `vespa` | 286.59 | 3.79 | 6.65 | 16.74 | 3519.5 | 33.3% | 0.00% | 4 | `PROCEED` | `None` |
| `milvus` | 273.97 | 6.05 | 14.74 | 21.62 | 415.9 | 7.3% | 0.00% | 4 | `PROCEED` | `None` |

## Workload & Architectural Notes

- **Host Hardware**: Ubuntu 24.04.5 LTS (Kernel 6.8.0), Intel Core i7-2600K (8 threads @ 3.8GHz), 16 GiB RAM.
- **Workload Mix**: 80% filtered ANN cosine search (`price_eur <= 50.0`, `seller_country`), 15% real-time batch upsert, 5% point lookup.
- **Isolation**: Sequential target execution with 4GB container cgroup caps, OS page-cache drop (`sync && drop_caches`), and cooldown window between engines.
- **Telemetry & Control Plane**: Telemetry percentiles and container cgroup resource metrics were sampled directly by the test harness; SRE triage was evaluated offline via local deterministic rule ladder (`source: deterministic_fallback`, implementing the typed `gpt-6-luna` schema contract). Live OpenAI Decisions API requests and Temporal LIFO Saga mitigations were not executed during this benchmark run.
