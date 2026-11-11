SHELL := /bin/bash
.DEFAULT_GOAL := help

PROXY_BIN         := bin/proxy
TOOLCHAIN_IMAGE   := vfs-go-toolchain:1.23-buf
GOMOD_CACHE_VOL   := vfs_gomod_cache
VENV_DIR          := .venv
PYTEST            := $(VENV_DIR)/bin/pytest

HAS_GO            := $(shell command -v go 2>/dev/null)
HAS_BUF           := $(shell command -v buf 2>/dev/null)
HAS_PROTOC_GO     := $(shell command -v protoc-gen-go 2>/dev/null)
HAS_PROTOC_CONN   := $(shell command -v protoc-gen-connect-go 2>/dev/null)

.PHONY: help toolchain-image proto build up down test-unit test-go venv clean

help: ## Display available automation commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

toolchain-image: ## Build cached local Go 1.23 + buf toolchain stage when host tools are absent
	@if ! docker image inspect $(TOOLCHAIN_IMAGE) >/dev/null 2>&1; then \
		docker build --target toolchain -t $(TOOLCHAIN_IMAGE) -f solution/proxy/Dockerfile .; \
	fi
	@docker volume inspect $(GOMOD_CACHE_VOL) >/dev/null 2>&1 || docker volume create $(GOMOD_CACHE_VOL) >/dev/null

proto: ## Lint and compile Protobuf schema via Buf (buf.yaml + buf.gen.yaml)
	@mkdir -p gen
	@if [ -n "$(HAS_BUF)" ] && [ -n "$(HAS_PROTOC_GO)" ] && [ -n "$(HAS_PROTOC_CONN)" ]; then \
		buf lint && buf generate; \
	else \
		$(MAKE) --no-print-directory toolchain-image; \
		docker run --rm \
			--user "$$(id -u):$$(id -g)" \
			-e HOME=/tmp \
			-v "$(CURDIR):/workspace" \
			-w /workspace \
			$(TOOLCHAIN_IMAGE) \
			bash -c "buf lint && buf generate"; \
	fi

build: ## Compile Go ConnectRPC L7 proxy into bin/proxy
	@mkdir -p bin
	@if [ -n "$(HAS_GO)" ]; then \
		CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o $(PROXY_BIN) ./solution/proxy; \
	else \
		$(MAKE) --no-print-directory toolchain-image; \
		docker run --rm \
			--user "$$(id -u):$$(id -g)" \
			-e HOME=/tmp \
			-e GOCACHE=/tmp/gocache \
			-v "$(GOMOD_CACHE_VOL):/go/pkg/mod" \
			-v "$(CURDIR):/workspace" \
			-w /workspace \
			$(TOOLCHAIN_IMAGE) \
			CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o $(PROXY_BIN) ./solution/proxy; \
	fi

test-go: ## Execute Go proxy unit tests
	@if [ -n "$(HAS_GO)" ]; then \
		go test -v -race ./solution/proxy/...; \
	else \
		$(MAKE) --no-print-directory toolchain-image; \
		docker run --rm \
			--user "$$(id -u):$$(id -g)" \
			-e HOME=/tmp \
			-e GOCACHE=/tmp/gocache \
			-v "$(GOMOD_CACHE_VOL):/go/pkg/mod" \
			-v "$(CURDIR):/workspace" \
			-w /workspace \
			$(TOOLCHAIN_IMAGE) \
			go test -v ./solution/proxy/...; \
	fi

up: ## Start single-target pgvector, ConnectRPC proxy, and Prometheus via Docker Compose
	docker compose up -d --build

down: ## Tear down Docker Compose containers and networks cleanly
	docker compose --profile temporal down --remove-orphans

venv: ## Sync local Python virtual environment from deterministic uv.lock
	@if [ ! -x "$(PYTEST)" ]; then \
		UV_PROJECT_ENVIRONMENT=$(VENV_DIR) uv sync --frozen; \
	fi

test-unit: venv ## Execute Python unit tests in tests/unit/
	$(VENV_DIR)/bin/python -m pytest -v

clean: ## Purge Go binaries, test caches, and coverage artifacts idempotently
	rm -rf bin/ coverage.out .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
