# fs-benchmark-sandbox

> High-throughput vector db benchmarking harness with out-of-band Decisions API triage.

`fs-benchmark-sandbox` is an SRE and platform evaluation harness designed to benchmark vector search and feature-store retrieval engines (PostgreSQL `pgvector`, Qdrant, Vespa, and Milvus) under simulated marketplace workloads. It couples a compiled Go L7 edge gateway with an out-of-band automated triage harness powered by the OpenAI Decisions API (`gpt-6-luna`) and Linux host-level chaos injection.

---

### Component Matrix

To maintain strict benchmarking validity on a developer workstation without sacrificing systems fidelity, distributed production patterns are adapted into an isolated, sequential local harness:

| Component | Role | Concept → Sandbox Reality | Comprehension (1-5) |
| :--- | :--- | :--- | :---: |
| **Vector Engines** | System Under Test | Concurrent multi-DB cluster → **Sequential 1-container lifecycle** (`pgvector`, Qdrant, Vespa, Milvus; 4GB cap + cache flush) | `_` |
| **Serving Proxy** | L7 Edge Gateway | Envoy/Istio mesh or Python gateway → **Go ConnectRPC binary (`h2c`)** routing via `X-Target-Engine` on `:8080` | `_` |
| **Observability** | SLA Telemetry | Full Grafana/OTel stack (15s scrape) → **Standalone Prometheus** (`1s` scrape of `vfs_request_duration_seconds`) | `_` |
| **Chaos Agent** | Fault Injection | K8s Chaos Mesh daemonset → **Host Linux `tc netem` & `cgroups v2`** controlled from Python | `_` |
| **SRE Evaluator** | Tail-Latency Triage | Inline generative LLM (1.5–4s lag) → **Out-of-band OpenAI Decisions API (`gpt-6-luna`)** (`predicate`, `choice`, `score`) | `_` |
| **Workflow Engine** | Incident Mitigation | Multi-container Temporal cluster → **`temporalio` in-memory test server** (optional Docker profile) running LIFO Sagas | `_` |
| **Traffic Agent** | Load Generation | Distributed k6/Locust cluster → **Deterministic Python `asyncio` + `httpx[http2]` pool** (80% ANN, 15% write, 5% point lookup) | `_` |

---

### Workload & Data Contract

* **Dataset:** 50,000 synthetic items with 768-dimensional float embeddings (simulating multimodal marketplace representations) and typed metadata (`brand_id`, `category`, `price_eur`, `seller_country`).
* **Query Mix:**
  * **80% Filtered ANN Retrieval:** Vector cosine similarity where `price_eur <= 50.0` and `seller_country IN ('FR', 'ES')`.
  * **15% Real-Time Ingestion:** Concurrent partial updates simulating stock adjustments and inventory status.
  * **5% Pure Scalar / Lexical Lookup:** Point retrieval by `item_id`.
* **SLA Target:** Maintain **p99 latency < 25.0ms** under sustained high-concurrency QPS load and induced network/cgroup degradation.

---

### Quickstart

#### Prerequisites
* Linux host (Ubuntu 22.04+ or Debian 12 recommended)
* Docker & Docker Compose v2 (`cgroups v2` enabled)
* Python 3.12+ and Go 1.23+ (`Makefile` automatically falls back to a containerized Go 1.23 toolchain if `go`/`protoc` are not installed on the host)

#### 1. Start Infrastructure & Target 1 (`pgvector` + `proxy` + `prometheus`)
```bash
# Build and launch pgvector, ConnectRPC proxy (:8080), and Prometheus (:9090)
make up

# Verify active Prometheus scrape target health
curl -sf http://localhost:9090/api/v1/targets | grep -q '"health":"up"' && echo "Telemetry online"
```

#### 2. Compile the Go Serving Proxy Binary (Standalone / Local Check)
```bash
# Compile proto/feature_store.proto and build ./bin/proxy
make build
```

#### 3. Run Benchmark Harness
```bash
# Execute sequential benchmark with automated out-of-band SRE triage
python3 -m harness.orchestrator --engine pgvector --seed-count 200 --requests 200
```

#### 4. Run Unit & Contract Tests
```bash
# Run Go proxy h2c tests and Python SRE/Temporal/Chaos unit tests
make test-go
make test-unit
```

---

### Repository Structure

```text
.
├── proto/
│   └── feature_store.proto         # ConnectRPC schema (768-dim vectors + metadata)
├── solution/
│   ├── proxy/                      # Go L7 h2c reverse proxy shim, Dockerfile & tests
│   └── engine_adapters/
│       └── pgvector/               # Target 1 DDL & HNSW cosine index configuration
├── harness/
│   ├── orchestrator.py             # Sequential container lifecycle & cache purger
│   ├── traffic_agent.py            # High-throughput AsyncIO synthetic load generator
│   ├── chaos_agent.py              # Linux tc netem & cgroup controller
│   ├── sre_evaluator.py            # OpenAI Decisions API triage client
│   └── temporal_saga.py            # Durable LIFO mitigation workflow definitions
├── deploy/
│   └── k8s/                        # Declarative K8s Deployment, Service, ServiceMonitor & Kustomize
├── tests/
│   ├── unit/                       # Telemetry parsing, Chaos & Temporal Saga unit tests
│   └── integration/                # End-to-end container benchmark passes
├── docker-compose.yaml             # Single-target engine, proxy & Prometheus definitions
├── prometheus.yml                  # 1s scrape config with sub-millisecond histogram buckets
└── Makefile                        # Build, lifecycle, and benchmark automation targets
```

---

### License

Apache License 2.0. See [LICENSE](LICENSE) for details.
