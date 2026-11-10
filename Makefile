SHELL := /bin/bash
.DEFAULT_GOAL := help

MODULE_NAME       := github.com/fs-benchmark/fs-benchmark-sandbox
PROXY_BIN         := bin/proxy
PROTO_FILE        := proto/feature_store.proto
TOOLCHAIN_IMAGE   := vfs-go-toolchain:1.23
GOMOD_CACHE_VOL   := vfs_gomod_cache
PYTHON            ?= python3
VENV_DIR          := .venv
PYTEST            := $(VENV_DIR)/bin/pytest

HAS_GO            := $(shell command -v go 2>/dev/null)
HAS_PROTOC        := $(shell command -v protoc 2>/dev/null)
HAS_PROTOC_GO     := $(shell command -v protoc-gen-go 2>/dev/null)
HAS_PROTOC_CONN   := $(shell command -v protoc-gen-connect-go 2>/dev/null)

.PHONY: help toolchain-image proto build up down test-unit test-go venv clean

help: ## Display available automation commands
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

toolchain-image: ## Build cached local Go 1.23 + protoc toolchain stage when host tools are absent
	@if ! docker image inspect $(TOOLCHAIN_IMAGE) >/dev/null 2>&1; then \
		docker build --target toolchain -t $(TOOLCHAIN_IMAGE) -f solution/proxy/Dockerfile .; \
	fi
	@docker volume inspect $(GOMOD_CACHE_VOL) >/dev/null 2>&1 || docker volume create $(GOMOD_CACHE_VOL) >/dev/null

proto: ## Compile Protobuf schema via protoc-gen-go and protoc-gen-connect-go
	@mkdir -p gen
	@if [ -n "$(HAS_PROTOC)" ] && [ -n "$(HAS_PROTOC_GO)" ] && [ -n "$(HAS_PROTOC_CONN)" ]; then \
		protoc -I proto \
			--go_out=. --go_opt=module=$(MODULE_NAME) \
			--connect-go_out=. --connect-go_opt=module=$(MODULE_NAME) \
			$(PROTO_FILE); \
	else \
		$(MAKE) --no-print-directory toolchain-image; \
		docker run --rm \
			--user "$$(id -u):$$(id -g)" \
			-e HOME=/tmp \
			-v "$(CURDIR):/workspace" \
			-w /workspace \
			$(TOOLCHAIN_IMAGE) \
			protoc -I proto \
				--go_out=. --go_opt=module=$(MODULE_NAME) \
				--connect-go_out=. --connect-go_opt=module=$(MODULE_NAME) \
				$(PROTO_FILE); \
	fi

build: proto ## Compile Go ConnectRPC L7 proxy into bin/proxy
	@mkdir -p bin
	@if [ -n "$(HAS_GO)" ]; then \
		go mod tidy && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o $(PROXY_BIN) ./solution/proxy; \
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
			bash -c "go mod tidy && CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o $(PROXY_BIN) ./solution/proxy"; \
	fi

test-go: proto ## Execute Go proxy unit tests
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

venv: ## Initialize local Python virtual environment and install dependencies
	@if [ ! -x "$(PYTEST)" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv $(VENV_DIR) && uv pip install --python $(VENV_DIR)/bin/python -r requirements.txt; \
		else \
			$(PYTHON) -m venv $(VENV_DIR) && $(VENV_DIR)/bin/pip install --upgrade pip && $(VENV_DIR)/bin/pip install -r requirements.txt; \
		fi; \
	fi

test-unit: venv ## Execute Python unit tests in tests/unit/
	PYTHONPATH="$(CURDIR)" $(VENV_DIR)/bin/python -m pytest -v tests/unit/

clean: ## Purge Go binaries, test caches, and coverage artifacts idempotently
	rm -rf bin/ coverage.out .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
