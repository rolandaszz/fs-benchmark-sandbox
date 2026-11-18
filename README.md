# fs-benchmark-sandbox

> High-throughput vector db benchmarking harness with out-of-band Decisions API triage.

`fs-benchmark-sandbox` is an SRE and platform evaluation harness designed to benchmark vector search and feature-store retrieval engines (PostgreSQL `pgvector`, Qdrant, Vespa, and Milvus) under simulated marketplace workloads. It couples a compiled Go L7 edge gateway with an out-of-band automated triage harness powered by the OpenAI Decisions API (`gpt-6-luna`) and Linux host-level chaos injection.

---

### Component Matrix

To maintain strict benchmarking validity on a developer workstation without sacrificing systems fidelity, distributed production patterns are adapted into an isolated, sequential local harness:

| Component | Role | Concept → Sandbox | Comprehension (1-5) |
| :--- | :--- | :--- | :---: |
| **Vector Engines** | System Under Test | Concurrent multi-DB cluster → **Sequential 1-container lifecycle** (`pgvector`, Qdrant, Vespa, Milvus; 4GB cap + cache flush) | `_` |
| **Serving Proxy** | L7 Edge Gateway | Network/Service mesh → **Go ConnectRPC binary (`h2c`)** routing via `X-Target-Engine` on `:8080` | `_` |
| **Observability** | SLA Telemetry | Full Grafana/OTel stack → **Standalone Prometheus** (`1s` scrape of `vfs_request_duration_seconds`) | `_` |
| **Chaos Agent** | Fault Injection | K8s Chaos Mesh daemonset → **Host Linux `tc netem` & `cgroups v2`** controlled from Python | `_` |
| **SRE Evaluator** | Tail-Latency Triage | Inline generative LLM (1.5–4s lag) → **Decisions API contract (`gpt-6-luna`)** with offline deterministic rule ladder fallback | `_` |
| **Workflow Engine** | Incident Mitigation | Multi-container Temporal cluster → **`temporalio` in-memory test server** (LIFO Sagas triggered during fault/chaos drills) | `_` |
| **Traffic Agent** | Load Generation | Distributed k6/Locust cluster → **Deterministic Python `asyncio` + `httpx[http2]` pool** (80% ANN, 15% write, 5% point lookup) | `_` |

> [!NOTE]
> **Triage & Control Plane Execution**: While the harness implements the production schema contract for the OpenAI Decisions API (`gpt-6-luna`) and Temporal LIFO Saga workflows, benchmark runs execute offline against the local deterministic rule ladder (`source: deterministic_fallback`) without external API dependencies. Live Decisions API calls are only made if `OPENAI_API_KEY` is provided, and Temporal Sagas are actuated during critical SLA breaches or active chaos injection drills.

---

### Workload & Data Contract

* **Dataset:** 50,000 synthetic items with 768-dimensional float embeddings (simulating multimodal marketplace representations) and typed metadata (`brand_id`, `category`, `price_eur`, `seller_country`).
* **Query Mix:**
  * **80% Filtered ANN Retrieval:** Vector cosine similarity where `price_eur <= 50.0` and `seller_country IN ('FR', 'ES')`.
  * **15% Real-Time Ingestion:** Concurrent partial updates simulating stock adjustments and inventory status.
  * **5% Pure Scalar / Lexical Lookup:** Point retrieval by `item_id`.
* **SLA Target:** Maintain **p99 latency < 25.0ms** under sustained high-concurrency QPS load and induced network/cgroup degradation.
* **Host Hardware:** Ubuntu 24.04.5 LTS (Kernel 6.8.0), Intel Core i7-2600K (8 threads @ 3.8GHz), 16 GiB RAM.

---

### Quickstart

#### Prerequisites
* Linux host (Ubuntu 22.04+ or Debian 12 recommended)
* Docker & Docker Compose v2 (`cgroups v2` enabled)
* Python 3.12+ and Go 1.23+ (`Makefile` automatically falls back to a containerized Go 1.23 toolchain if `go`/`protoc` are not installed on the host)

#### 1. Start Infrastructure (`pgvector`, `Qdrant`, `Vespa`, `Milvus`, `proxy` & `prometheus`)
```bash
# Build and launch database engines, ConnectRPC proxy (:8080), and Prometheus (:9090)
make up

# Verify active Prometheus scrape target health
curl -sf http://localhost:9090/api/v1/targets | grep -q '"health":"up"' && echo "Telemetry online"
```

#### 2. Compile the Go Serving Proxy Binary (Standalone / Local Check)
```bash
# Lint/generate Protobuf via Buf and compile ./bin/proxy
make build
```

#### 3. Run Benchmark Harness
```bash
# Benchmark pgvector (Target 1)
python3 -m harness.orchestrator --engine pgvector --seed-count 5000 --requests 2000 --skip-seed --concurrency 8

# Benchmark Qdrant (Target 2)
python3 -m harness.orchestrator --engine qdrant --seed-count 5000 --requests 2000 --skip-seed --concurrency 8

# Benchmark Vespa (Target 3)
python3 -m harness.orchestrator --engine vespa --seed-count 5000 --requests 2000 --skip-seed --concurrency 8

# Benchmark Milvus (Target 4)
python3 -m harness.orchestrator --engine milvus --seed-count 5000 --requests 2000 --skip-seed --concurrency 8

# Run Full Multi-Engine Suite & Generate Markdown and JSON Reports (Phase 5)
python3 -m harness.orchestrator --all-engines --seed-count 5000 --requests 1000 --skip-seed --concurrency 8 --report-out docs/benchmark_report.md --json-out docs/benchmark_report.json

# Run Fault-Injection Chaos Drill (safe dry-run simulation triggering Temporal LIFO Saga failover)
python3 -m harness.orchestrator --engine milvus --seed-count 5000 --requests 400 --skip-seed --concurrency 8 --chaos-delay-ms 45

# Run Live Kernel-Level Chaos with sudo (executes real Linux tc netem delay on loopback & drops page-cache)
sudo $(pwd)/.venv/bin/python -m harness.orchestrator --engine milvus --seed-count 5000 --requests 400 --skip-seed --concurrency 8 --chaos-delay-ms 45 --live-chaos
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
├── buf.yaml                        # Buf v2 module and lint configuration
├── buf.gen.yaml                    # Buf code generation template for ConnectRPC
├── pyproject.toml                  # PEP 621 Python project metadata & tool configurations
├── uv.lock                         # Deterministic pinned Python lockfile
├── proto/
│   └── vfs/
│       └── v1/
│           └── feature_store.proto # ConnectRPC schema (768-dim vectors + metadata)
├── solution/
│   ├── proxy/                      # Go L7 h2c proxy (proxy.go, qdrant.go, vespa.go, milvus.go), Dockerfile & tests
│   └── engine_adapters/
│       ├── pgvector/               # Target 1 DDL & HNSW cosine index configuration
│       ├── vespa/                  # Target 3 application package (services.xml & item.sd schema)
│       └── milvus/                 # Target 4 standalone configuration (milvus.yaml)
├── harness/
│   ├── orchestrator.py             # Sequential container lifecycle & cache purger
│   ├── traffic_agent.py            # High-throughput AsyncIO synthetic load generator
│   ├── chaos_agent.py              # Linux tc netem & cgroup controller
│   ├── sre_evaluator.py            # OpenAI Decisions API triage client
│   ├── report_generator.py         # Markdown & JSON benchmark matrix synthesizer
│   └── temporal_saga.py            # Durable LIFO mitigation workflow definitions
├── docs/
│   ├── benchmark_report.md         # Synthesized 4-engine comparative evaluation report
│   └── benchmark_report.json       # Structured comparative telemetry & ranking payload
├── deploy/
│   └── k8s/                        # Declarative K8s Deployment, Service, ServiceMonitor & Kustomize
├── tests/
│   ├── unit/                       # Telemetry parsing, Chaos & Temporal Saga unit tests
│   └── integration/                # End-to-end container benchmark passes
├── docker-compose.yaml             # Engine targets, proxy & Prometheus definitions
├── prometheus.yaml                 # 1s scrape config with sub-millisecond histogram buckets
└── Makefile                        # Build, lifecycle, and benchmark automation targets
```

---

### License

Apache License 2.0. See [LICENSE](LICENSE) for details.
