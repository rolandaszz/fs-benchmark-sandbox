FROM golang:1.23-bookworm AS toolchain

RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV GOBIN=/usr/local/bin
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.5 && \
    go install connectrpc.com/connect/cmd/protoc-gen-connect-go@v1.18.1 && \
    mkdir -p /go/pkg/mod /tmp/gocache && chmod -R 0777 /go /tmp/gocache

FROM toolchain AS builder

WORKDIR /workspace

COPY go.mod ./
COPY proto/ ./proto/
COPY solution/proxy/ ./solution/proxy/

RUN mkdir -p gen && \
    protoc -I proto \
      --go_out=. --go_opt=module=github.com/fs-benchmark/fs-benchmark-sandbox \
      --connect-go_out=. --connect-go_opt=module=github.com/fs-benchmark/fs-benchmark-sandbox \
      proto/feature_store.proto && \
    go mod tidy && \
    CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/proxy ./solution/proxy

FROM alpine:3.21

RUN apk add --no-cache ca-certificates wget && \
    adduser -D -u 10001 vfsproxy

USER vfsproxy
WORKDIR /app
COPY --from=builder /out/proxy /app/proxy

EXPOSE 8080
ENTRYPOINT ["/app/proxy"]
