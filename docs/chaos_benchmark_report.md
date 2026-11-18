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
| `pgvector` | 101.32 | 3.95 | 8.86 | 9.77 | 138.7 | 0.0% | 0.00% | 5 | `PROCEED` | `None` |
| `qdrant` | 111.51 | 1.83 | 4.26 | 4.85 | 257.1 | 0.3% | 0.00% | 5 | `PROCEED` | `None` |
| `vespa` | 108.47 | 4.15 | 9.16 | 10.31 | 3520.5 | 36.2% | 0.00% | 5 | `PROCEED` | `None` |
| `milvus` | 105.98 | 6.80 | 11.62 | 16.49 | 414.8 | 7.0% | 0.00% | 4 | `PROCEED` | `None` |

## Workload & Architectural Notes

- **Host Hardware**: Ubuntu 24.04.5 LTS (Kernel 6.8.0), Intel Core i7-2600K (8 threads @ 3.8GHz), 16 GiB RAM.
- **Workload Mix**: 80% filtered ANN cosine search (`price_eur <= 50.0`, `seller_country`), 15% real-time batch upsert, 5% point lookup.
- **Isolation**: Sequential target execution with 4GB container cgroup caps, OS page-cache drop (`sync && drop_caches`), and cooldown window between engines.
- **Telemetry & Control Plane**: Telemetry percentiles and container cgroup resource metrics were sampled directly by the test harness; SRE triage was evaluated offline via local deterministic rule ladder (`source: deterministic_fallback`, implementing the typed `gpt-6-luna` schema contract). Live OpenAI Decisions API requests and Temporal LIFO Saga mitigations were not executed during this benchmark run.
