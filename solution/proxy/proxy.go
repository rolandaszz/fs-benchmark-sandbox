package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"connectrpc.com/connect"
	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
	"github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1/vfsv1connect"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

const (
	EmbeddingDimension = 768
	TargetEngineHeader = "X-Target-Engine"
)

// VFSLatencyBuckets defines the mandated Prometheus histogram bounds in seconds,
// providing fine-grained resolution around the 25.0ms (0.025s) VFS p99 SLA.
var VFSLatencyBuckets = []float64{
	0.0005, 0.001, 0.0025, 0.005, 0.010, 0.015,
	0.020, 0.025, 0.050, 0.100, 0.250, 0.500,
}

// EngineBackend abstracts a single active online feature/vector store engine.
type EngineBackend interface {
	Name() string
	GetItem(ctx context.Context, itemID string) (*vfsv1.ItemFeatures, bool, error)
	BatchGetItems(ctx context.Context, itemIDs []string) ([]*vfsv1.ItemFeatures, []string, error)
	UpsertItems(ctx context.Context, items []*vfsv1.ItemFeatures) (uint32, error)
	SearchSimilar(ctx context.Context, queryVec []float32, topK uint32, maxPrice float32, countries []string) ([]*vfsv1.ScoredItemFeatures, error)
	Ping(ctx context.Context) error
	Close()
}

// PgVectorBackend implements EngineBackend against PostgreSQL 17 + pgvector HNSW.
type PgVectorBackend struct {
	pool *pgxpool.Pool
}

func NewPgVectorBackend(ctx context.Context, dsn string) (*PgVectorBackend, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse pgvector dsn: %w", err)
	}
	cfg.MaxConns = 64
	cfg.MinConns = 4
	cfg.MaxConnLifetime = 30 * time.Minute
	cfg.MaxConnIdleTime = 5 * time.Minute
	if cfg.ConnConfig.RuntimeParams == nil {
		cfg.ConnConfig.RuntimeParams = make(map[string]string)
	}
	cfg.ConnConfig.RuntimeParams["hnsw.iterative_scan"] = "relaxed_order"
	cfg.ConnConfig.RuntimeParams["random_page_cost"] = "1.1"

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("create pgxpool: %w", err)
	}
	return &PgVectorBackend{pool: pool}, nil
}

func (b *PgVectorBackend) Name() string { return "pgvector" }

func (b *PgVectorBackend) Ping(ctx context.Context) error {
	return b.pool.Ping(ctx)
}

func (b *PgVectorBackend) Close() {
	b.pool.Close()
}

func (b *PgVectorBackend) GetItem(ctx context.Context, itemID string) (*vfsv1.ItemFeatures, bool, error) {
	const query = `
		SELECT item_id, embedding::text, brand_id, category, price_eur, seller_country, updated_at
		FROM items
		WHERE item_id = $1
	`
	var (
		id, vecText, brandID, category, country string
		price                                   float32
		updatedAt                               time.Time
	)
	err := b.pool.QueryRow(ctx, query, itemID).Scan(&id, &vecText, &brandID, &category, &price, &country, &updatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	vec, err := parseVectorLiteral(vecText)
	if err != nil {
		return nil, false, err
	}
	return &vfsv1.ItemFeatures{
		ItemId:          id,
		Embedding:       vec,
		BrandId:         brandID,
		Category:        category,
		PriceEur:        price,
		SellerCountry:   country,
		UpdatedAtUnixMs: updatedAt.UnixMilli(),
	}, true, nil
}

func (b *PgVectorBackend) BatchGetItems(ctx context.Context, itemIDs []string) ([]*vfsv1.ItemFeatures, []string, error) {
	if len(itemIDs) == 0 {
		return nil, nil, nil
	}
	const query = `
		SELECT item_id, embedding::text, brand_id, category, price_eur, seller_country, updated_at
		FROM items
		WHERE item_id = ANY($1)
	`
	rows, err := b.pool.Query(ctx, query, itemIDs)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()

	foundMap := make(map[string]*vfsv1.ItemFeatures, len(itemIDs))
	for rows.Next() {
		var (
			id, vecText, brandID, category, country string
			price                                   float32
			updatedAt                               time.Time
		)
		if err := rows.Scan(&id, &vecText, &brandID, &category, &price, &country, &updatedAt); err != nil {
			return nil, nil, err
		}
		vec, err := parseVectorLiteral(vecText)
		if err != nil {
			return nil, nil, err
		}
		foundMap[id] = &vfsv1.ItemFeatures{
			ItemId:          id,
			Embedding:       vec,
			BrandId:         brandID,
			Category:        category,
			PriceEur:        price,
			SellerCountry:   country,
			UpdatedAtUnixMs: updatedAt.UnixMilli(),
		}
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}

	items := make([]*vfsv1.ItemFeatures, 0, len(foundMap))
	var missing []string
	for _, reqID := range itemIDs {
		if item, ok := foundMap[reqID]; ok {
			items = append(items, item)
		} else {
			missing = append(missing, reqID)
		}
	}
	return items, missing, nil
}

func (b *PgVectorBackend) UpsertItems(ctx context.Context, items []*vfsv1.ItemFeatures) (uint32, error) {
	if len(items) == 0 {
		return 0, nil
	}
	const stmt = `
		INSERT INTO items (item_id, embedding, brand_id, category, price_eur, seller_country, updated_at)
		VALUES ($1, $2::vector(768), $3, $4, $5, $6, $7)
		ON CONFLICT (item_id) DO UPDATE SET
			embedding = EXCLUDED.embedding,
			brand_id = EXCLUDED.brand_id,
			category = EXCLUDED.category,
			price_eur = EXCLUDED.price_eur,
			seller_country = EXCLUDED.seller_country,
			updated_at = EXCLUDED.updated_at
	`
	batch := &pgx.Batch{}
	now := time.Now().UTC()
	for _, item := range items {
		ts := now
		if item.GetUpdatedAtUnixMs() > 0 {
			ts = time.UnixMilli(item.GetUpdatedAtUnixMs()).UTC()
		}
		batch.Queue(
			stmt,
			item.GetItemId(),
			formatVectorLiteral(item.GetEmbedding()),
			item.GetBrandId(),
			item.GetCategory(),
			item.GetPriceEur(),
			item.GetSellerCountry(),
			ts,
		)
	}
	br := b.pool.SendBatch(ctx, batch)
	defer br.Close()

	var count uint32
	for range items {
		if _, err := br.Exec(); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

func (b *PgVectorBackend) SearchSimilar(
	ctx context.Context,
	queryVec []float32,
	topK uint32,
	maxPrice float32,
	countries []string,
) ([]*vfsv1.ScoredItemFeatures, error) {
	if topK == 0 {
		topK = 10
	}
	vecLit := formatVectorLiteral(queryVec)
	var (
		rows pgx.Rows
		err  error
	)
	switch {
	case maxPrice > 0 && len(countries) > 0:
		const q = `
			SELECT item_id, brand_id, category, price_eur, seller_country, updated_at,
			       1.0 - (embedding <=> $1::vector(768)) AS cosine_similarity
			FROM items
			WHERE price_eur <= $2 AND seller_country = ANY($3)
			ORDER BY embedding <=> $1::vector(768)
			LIMIT $4
		`
		rows, err = b.pool.Query(ctx, q, vecLit, maxPrice, countries, topK)
	case maxPrice > 0:
		const q = `
			SELECT item_id, brand_id, category, price_eur, seller_country, updated_at,
			       1.0 - (embedding <=> $1::vector(768)) AS cosine_similarity
			FROM items
			WHERE price_eur <= $2
			ORDER BY embedding <=> $1::vector(768)
			LIMIT $3
		`
		rows, err = b.pool.Query(ctx, q, vecLit, maxPrice, topK)
	case len(countries) > 0:
		const q = `
			SELECT item_id, brand_id, category, price_eur, seller_country, updated_at,
			       1.0 - (embedding <=> $1::vector(768)) AS cosine_similarity
			FROM items
			WHERE seller_country = ANY($2)
			ORDER BY embedding <=> $1::vector(768)
			LIMIT $3
		`
		rows, err = b.pool.Query(ctx, q, vecLit, countries, topK)
	default:
		const q = `
			SELECT item_id, brand_id, category, price_eur, seller_country, updated_at,
			       1.0 - (embedding <=> $1::vector(768)) AS cosine_similarity
			FROM items
			ORDER BY embedding <=> $1::vector(768)
			LIMIT $2
		`
		rows, err = b.pool.Query(ctx, q, vecLit, topK)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var hits []*vfsv1.ScoredItemFeatures
	for rows.Next() {
		var (
			id, brandID, category, country string
			price, sim                     float32
			updatedAt                      time.Time
		)
		if err := rows.Scan(&id, &brandID, &category, &price, &country, &updatedAt, &sim); err != nil {
			return nil, err
		}
		hits = append(hits, &vfsv1.ScoredItemFeatures{
			Item: &vfsv1.ItemFeatures{
				ItemId:          id,
				BrandId:         brandID,
				Category:        category,
				PriceEur:        price,
				SellerCountry:   country,
				UpdatedAtUnixMs: updatedAt.UnixMilli(),
			},
			CosineSimilarity: sim,
		})
	}
	return hits, rows.Err()
}

func formatVectorLiteral(vec []float32) string {
	var sb strings.Builder
	sb.Grow(len(vec) * 10)
	sb.WriteByte('[')
	for i, v := range vec {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(strconv.FormatFloat(float64(v), 'f', -1, 32))
	}
	sb.WriteByte(']')
	return sb.String()
}

func parseVectorLiteral(lit string) ([]float32, error) {
	trimmed := strings.TrimSpace(lit)
	trimmed = strings.TrimPrefix(trimmed, "[")
	trimmed = strings.TrimSuffix(trimmed, "]")
	if trimmed == "" {
		return nil, nil
	}
	out := make([]float32, 0, 768)
	start := 0
	for i := 0; i <= len(trimmed); i++ {
		if i == len(trimmed) || trimmed[i] == ',' {
			token := strings.TrimSpace(trimmed[start:i])
			if token != "" {
				val, err := strconv.ParseFloat(token, 32)
				if err != nil {
					return nil, fmt.Errorf("parse vector element %d: %w", len(out), err)
				}
				out = append(out, float32(val))
			}
			start = i + 1
		}
	}
	return out, nil
}

var knownTargetEngines = map[string]struct{}{
	"pgvector": {},
	"qdrant":   {},
	"vespa":    {},
	"milvus":   {},
}

// ProxyTelemetry encapsulates Prometheus metrics for the ConnectRPC L7 shim.
type ProxyTelemetry struct {
	Registry        *prometheus.Registry
	RequestDuration *prometheus.HistogramVec
	InflightGauge   prometheus.Gauge
}

func NewProxyTelemetry(defaultEngine string) *ProxyTelemetry {
	reg := prometheus.NewRegistry()
	reg.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)

	reqDur := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vfs_request_duration_seconds",
			Help:    "End-to-end latency distribution of VFS ConnectRPC proxy requests in seconds.",
			Buckets: VFSLatencyBuckets,
		},
		[]string{"engine", "rpc", "status"},
	)
	inflight := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "vfs_active_inflight_requests",
		Help: "Current number of in-flight ConnectRPC requests on the L7 proxy.",
	})

	reg.MustRegister(reqDur, inflight)
	// Pre-initialize baseline series so Prometheus sees explicit histogram buckets immediately on startup.
	reqDur.WithLabelValues(defaultEngine, "GetItemFeatures", "ok")
	return &ProxyTelemetry{
		Registry:        reg,
		RequestDuration: reqDur,
		InflightGauge:   inflight,
	}
}

// FeatureStoreProxy implements vfsv1connect.FeatureStoreServiceHandler and routes via X-Target-Engine.
type FeatureStoreProxy struct {
	vfsv1connect.UnimplementedFeatureStoreServiceHandler
	defaultEngine string
	backends      map[string]EngineBackend
	telemetry     *ProxyTelemetry
}

func NewFeatureStoreProxy(defaultEngine string, backends map[string]EngineBackend, telemetry *ProxyTelemetry) *FeatureStoreProxy {
	return &FeatureStoreProxy{
		defaultEngine: defaultEngine,
		backends:      backends,
		telemetry:     telemetry,
	}
}

func (p *FeatureStoreProxy) resolveBackend(headers http.Header) (EngineBackend, string, error) {
	target := strings.ToLower(strings.TrimSpace(headers.Get(TargetEngineHeader)))
	if target == "" {
		target = p.defaultEngine
	}
	if _, known := knownTargetEngines[target]; !known {
		return nil, "unknown", connect.NewError(
			connect.CodeInvalidArgument,
			fmt.Errorf("unsupported %s header value %q", TargetEngineHeader, target),
		)
	}
	backend, active := p.backends[target]
	if !active {
		return nil, target, connect.NewError(
			connect.CodeUnavailable,
			fmt.Errorf("target engine %q is inactive in single-tenant benchmark mode", target),
		)
	}
	return backend, target, nil
}

func (p *FeatureStoreProxy) observe(engine, rpc string, start time.Time, err error) {
	if _, known := knownTargetEngines[engine]; !known {
		engine = "unknown"
	}
	status := "ok"
	if err != nil {
		status = "error"
	}
	p.telemetry.RequestDuration.WithLabelValues(engine, rpc, status).Observe(time.Since(start).Seconds())
}

func (p *FeatureStoreProxy) GetItemFeatures(
	ctx context.Context,
	req *connect.Request[vfsv1.GetItemFeaturesRequest],
) (*connect.Response[vfsv1.GetItemFeaturesResponse], error) {
	start := time.Now()
	p.telemetry.InflightGauge.Inc()
	defer p.telemetry.InflightGauge.Dec()

	backend, engine, err := p.resolveBackend(req.Header())
	if err != nil {
		p.observe(engine, "GetItemFeatures", start, err)
		return nil, err
	}
	if strings.TrimSpace(req.Msg.GetItemId()) == "" {
		err := connect.NewError(connect.CodeInvalidArgument, errors.New("item_id must not be empty"))
		p.observe(engine, "GetItemFeatures", start, err)
		return nil, err
	}

	item, found, err := backend.GetItem(ctx, req.Msg.GetItemId())
	if err != nil {
		cErr := connect.NewError(connect.CodeInternal, fmt.Errorf("get item from %s: %w", engine, err))
		p.observe(engine, "GetItemFeatures", start, cErr)
		return nil, cErr
	}
	p.observe(engine, "GetItemFeatures", start, nil)
	resp := connect.NewResponse(&vfsv1.GetItemFeaturesResponse{
		Item:  item,
		Found: found,
	})
	resp.Header().Set(TargetEngineHeader, engine)
	return resp, nil
}

func (p *FeatureStoreProxy) BatchGetItemFeatures(
	ctx context.Context,
	req *connect.Request[vfsv1.BatchGetItemFeaturesRequest],
) (*connect.Response[vfsv1.BatchGetItemFeaturesResponse], error) {
	start := time.Now()
	p.telemetry.InflightGauge.Inc()
	defer p.telemetry.InflightGauge.Dec()

	backend, engine, err := p.resolveBackend(req.Header())
	if err != nil {
		p.observe(engine, "BatchGetItemFeatures", start, err)
		return nil, err
	}
	if len(req.Msg.GetItemIds()) == 0 {
		err := connect.NewError(connect.CodeInvalidArgument, errors.New("item_ids must not be empty"))
		p.observe(engine, "BatchGetItemFeatures", start, err)
		return nil, err
	}

	items, missing, err := backend.BatchGetItems(ctx, req.Msg.GetItemIds())
	if err != nil {
		cErr := connect.NewError(connect.CodeInternal, fmt.Errorf("batch get from %s: %w", engine, err))
		p.observe(engine, "BatchGetItemFeatures", start, cErr)
		return nil, cErr
	}
	p.observe(engine, "BatchGetItemFeatures", start, nil)
	resp := connect.NewResponse(&vfsv1.BatchGetItemFeaturesResponse{
		Items:          items,
		MissingItemIds: missing,
	})
	resp.Header().Set(TargetEngineHeader, engine)
	return resp, nil
}

func (p *FeatureStoreProxy) UpsertItemFeatures(
	ctx context.Context,
	req *connect.Request[vfsv1.UpsertItemFeaturesRequest],
) (*connect.Response[vfsv1.UpsertItemFeaturesResponse], error) {
	start := time.Now()
	p.telemetry.InflightGauge.Inc()
	defer p.telemetry.InflightGauge.Dec()

	backend, engine, err := p.resolveBackend(req.Header())
	if err != nil {
		p.observe(engine, "UpsertItemFeatures", start, err)
		return nil, err
	}
	if len(req.Msg.GetItems()) == 0 {
		err := connect.NewError(connect.CodeInvalidArgument, errors.New("items must not be empty"))
		p.observe(engine, "UpsertItemFeatures", start, err)
		return nil, err
	}
	for _, item := range req.Msg.GetItems() {
		if strings.TrimSpace(item.GetItemId()) == "" {
			err := connect.NewError(connect.CodeInvalidArgument, errors.New("every item must have a non-empty item_id"))
			p.observe(engine, "UpsertItemFeatures", start, err)
			return nil, err
		}
		if len(item.GetEmbedding()) != EmbeddingDimension {
			err := connect.NewError(
				connect.CodeInvalidArgument,
				fmt.Errorf("item %q embedding length %d != required %d", item.GetItemId(), len(item.GetEmbedding()), EmbeddingDimension),
			)
			p.observe(engine, "UpsertItemFeatures", start, err)
			return nil, err
		}
	}

	upserted, err := backend.UpsertItems(ctx, req.Msg.GetItems())
	if err != nil {
		cErr := connect.NewError(connect.CodeInternal, fmt.Errorf("upsert items in %s: %w", engine, err))
		p.observe(engine, "UpsertItemFeatures", start, cErr)
		return nil, cErr
	}
	p.observe(engine, "UpsertItemFeatures", start, nil)
	resp := connect.NewResponse(&vfsv1.UpsertItemFeaturesResponse{UpsertedCount: upserted})
	resp.Header().Set(TargetEngineHeader, engine)
	return resp, nil
}

func (p *FeatureStoreProxy) SearchSimilarItems(
	ctx context.Context,
	req *connect.Request[vfsv1.SearchSimilarItemsRequest],
) (*connect.Response[vfsv1.SearchSimilarItemsResponse], error) {
	start := time.Now()
	p.telemetry.InflightGauge.Inc()
	defer p.telemetry.InflightGauge.Dec()

	backend, engine, err := p.resolveBackend(req.Header())
	if err != nil {
		p.observe(engine, "SearchSimilarItems", start, err)
		return nil, err
	}
	if len(req.Msg.GetQueryEmbedding()) != EmbeddingDimension {
		err := connect.NewError(
			connect.CodeInvalidArgument,
			fmt.Errorf("query_embedding length %d != required %d", len(req.Msg.GetQueryEmbedding()), EmbeddingDimension),
		)
		p.observe(engine, "SearchSimilarItems", start, err)
		return nil, err
	}

	hits, err := backend.SearchSimilar(
		ctx,
		req.Msg.GetQueryEmbedding(),
		req.Msg.GetTopK(),
		req.Msg.GetMaxPriceEur(),
		req.Msg.GetSellerCountries(),
	)
	if err != nil {
		cErr := connect.NewError(connect.CodeInternal, fmt.Errorf("search similar in %s: %w", engine, err))
		p.observe(engine, "SearchSimilarItems", start, cErr)
		return nil, cErr
	}
	p.observe(engine, "SearchSimilarItems", start, nil)
	resp := connect.NewResponse(&vfsv1.SearchSimilarItemsResponse{Hits: hits})
	resp.Header().Set(TargetEngineHeader, engine)
	return resp, nil
}

// NewProxyHTTPHandler wires the ConnectRPC handler, /metrics, /livez, and /healthz onto an h2c HTTP/2 mux.
func NewProxyHTTPHandler(proxy *FeatureStoreProxy) http.Handler {
	mux := http.NewServeMux()
	path, connectHandler := vfsv1connect.NewFeatureStoreServiceHandler(proxy)
	mux.Handle(path, connectHandler)
	mux.Handle("/metrics", promhttp.HandlerFor(proxy.telemetry.Registry, promhttp.HandlerOpts{}))
	mux.HandleFunc("/livez", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"alive"}`))
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 1500*time.Millisecond)
		defer cancel()

		active := make([]string, 0, len(proxy.backends))
		for name, b := range proxy.backends {
			if err := b.Ping(ctx); err != nil {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusServiceUnavailable)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"status": "degraded",
					"engine": name,
					"error":  err.Error(),
				})
				return
			}
			active = append(active, name)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":         "ok",
			"default_engine": proxy.defaultEngine,
			"active_engines": active,
		})
	})

	h2s := &http2.Server{
		MaxConcurrentStreams: 2048,
		IdleTimeout:          60 * time.Second,
	}
	return h2c.NewHandler(mux, h2s)
}

func getenvDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	listenAddr := getenvDefault("PROXY_LISTEN_ADDR", ":8080")
	defaultEngine := strings.ToLower(getenvDefault("DEFAULT_TARGET_ENGINE", "pgvector"))
	pgDSN := strings.TrimSpace(os.Getenv("PGVECTOR_DSN"))
	if pgDSN == "" {
		slog.Error("PGVECTOR_DSN environment variable is required")
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pgBackend, err := NewPgVectorBackend(ctx, pgDSN)
	if err != nil {
		slog.Error("failed to initialize pgvector backend", "error", err)
		os.Exit(1)
	}
	defer pgBackend.Close()

	backends := map[string]EngineBackend{
		"pgvector": pgBackend,
	}

	qdrantURL := strings.TrimSpace(os.Getenv("QDRANT_URL"))
	qdrantGRPCURL := strings.TrimSpace(os.Getenv("QDRANT_GRPC_URL"))
	qdrantCollection := getenvDefault("QDRANT_COLLECTION", "items")
	if qdrantURL != "" {
		qdrantBackend, err := NewQdrantBackend(ctx, qdrantURL, qdrantGRPCURL, qdrantCollection)
		if err != nil {
			slog.Error("failed to initialize qdrant backend", "error", err)
			os.Exit(1)
		}
		defer qdrantBackend.Close()
		backends["qdrant"] = qdrantBackend
		slog.Info("qdrant backend initialized", "rest_url", qdrantURL, "grpc_url", qdrantGRPCURL, "collection", qdrantCollection)
	}

	vespaURL := strings.TrimSpace(os.Getenv("VESPA_URL"))
	vespaConfigURL := getenvDefault("VESPA_CONFIG_URL", "http://localhost:19071")
	if vespaURL != "" {
		vespaBackend, err := NewVespaBackend(ctx, vespaURL, vespaConfigURL, "vfs", "item")
		if err != nil {
			slog.Error("failed to initialize vespa backend", "error", err)
			os.Exit(1)
		}
		defer vespaBackend.Close()
		backends["vespa"] = vespaBackend
		slog.Info("vespa backend initialized", "url", vespaURL, "config_url", vespaConfigURL)
	}

	milvusURL := strings.TrimSpace(os.Getenv("MILVUS_URL"))
	milvusCollection := getenvDefault("MILVUS_COLLECTION", "items")
	if milvusURL != "" {
		milvusBackend, err := NewMilvusBackend(ctx, milvusURL, milvusCollection)
		if err != nil {
			slog.Error("failed to initialize milvus backend", "error", err)
			os.Exit(1)
		}
		defer milvusBackend.Close()
		backends["milvus"] = milvusBackend
		slog.Info("milvus backend initialized", "url", milvusURL, "collection", milvusCollection)
	}

	telemetry := NewProxyTelemetry(defaultEngine)
	proxy := NewFeatureStoreProxy(defaultEngine, backends, telemetry)

	srv := &http.Server{
		Addr:              listenAddr,
		Handler:           NewProxyHTTPHandler(proxy),
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		slog.Info("starting ConnectRPC h2c proxy", "addr", listenAddr, "default_engine", defaultEngine)
		if serveErr := srv.ListenAndServe(); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			errCh <- serveErr
		}
		close(errCh)
	}()

	select {
	case <-ctx.Done():
		slog.Info("shutting down ConnectRPC proxy")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			slog.Error("graceful shutdown error", "error", err)
		}
	case serveErr := <-errCh:
		if serveErr != nil {
			slog.Error("proxy server failed", "error", serveErr)
			os.Exit(1)
		}
	}
}
