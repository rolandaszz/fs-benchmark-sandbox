package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"connectrpc.com/connect"
	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
	"github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1/vfsv1connect"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/protobuf/encoding/protowire"
)

type inMemoryBackend struct {
	mu    sync.RWMutex
	items map[string]*vfsv1.ItemFeatures
}

func newInMemoryBackend() *inMemoryBackend {
	return &inMemoryBackend{items: make(map[string]*vfsv1.ItemFeatures)}
}

func (m *inMemoryBackend) Name() string                 { return "pgvector" }
func (m *inMemoryBackend) Ping(_ context.Context) error { return nil }
func (m *inMemoryBackend) Close()                       {}

func (m *inMemoryBackend) GetItem(_ context.Context, itemID string) (*vfsv1.ItemFeatures, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	it, ok := m.items[itemID]
	return it, ok, nil
}

func (m *inMemoryBackend) BatchGetItems(_ context.Context, itemIDs []string) ([]*vfsv1.ItemFeatures, []string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var found []*vfsv1.ItemFeatures
	var missing []string
	for _, id := range itemIDs {
		if it, ok := m.items[id]; ok {
			found = append(found, it)
		} else {
			missing = append(missing, id)
		}
	}
	return found, missing, nil
}

func (m *inMemoryBackend) UpsertItems(_ context.Context, items []*vfsv1.ItemFeatures) (uint32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, it := range items {
		m.items[it.GetItemId()] = it
	}
	return uint32(len(items)), nil
}

func cosineSim(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

func (m *inMemoryBackend) SearchSimilar(
	_ context.Context,
	queryVec []float32,
	topK uint32,
	maxPrice float32,
	countries []string,
) ([]*vfsv1.ScoredItemFeatures, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var hits []*vfsv1.ScoredItemFeatures
	for _, it := range m.items {
		if maxPrice > 0 && it.GetPriceEur() > maxPrice {
			continue
		}
		if len(countries) > 0 && !slices.Contains(countries, it.GetSellerCountry()) {
			continue
		}
		hits = append(hits, &vfsv1.ScoredItemFeatures{
			Item:             it,
			CosineSimilarity: cosineSim(queryVec, it.GetEmbedding()),
		})
	}
	if topK > 0 && uint32(len(hits)) > topK {
		hits = hits[:topK]
	}
	return hits, nil
}

func makeTestEmbedding(seed float32) []float32 {
	vec := make([]float32, EmbeddingDimension)
	for i := range vec {
		vec[i] = seed + float32(i)*0.0001
	}
	return vec
}

func newH2CClient() *http.Client {
	return &http.Client{
		Transport: &http2.Transport{
			AllowHTTP: true,
			DialTLSContext: func(ctx context.Context, network, addr string, _ *tls.Config) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, network, addr)
			},
		},
	}
}

func TestConnectRPCProxyH2CRoutingAndTelemetry(t *testing.T) {
	mem := newInMemoryBackend()
	telemetry := NewProxyTelemetry("pgvector")
	proxy := NewFeatureStoreProxy("pgvector", map[string]EngineBackend{"pgvector": mem}, telemetry)

	srv := httptest.NewServer(NewProxyHTTPHandler(proxy))
	defer srv.Close()

	h2cClient := newH2CClient()
	rpcClient := vfsv1connect.NewFeatureStoreServiceClient(h2cClient, srv.URL, connect.WithGRPC())

	ctx := context.Background()
	item := &vfsv1.ItemFeatures{
		ItemId:          "item-lt-001",
		Embedding:       makeTestEmbedding(0.25),
		BrandId:         "brand-nike",
		Category:        "women_shoes",
		PriceEur:        34.50,
		SellerCountry:   "FR",
		UpdatedAtUnixMs: 1762530000000,
	}

	upsertReq := connect.NewRequest(&vfsv1.UpsertItemFeaturesRequest{Items: []*vfsv1.ItemFeatures{item}})
	upsertReq.Header().Set(TargetEngineHeader, "pgvector")
	upsertResp, err := rpcClient.UpsertItemFeatures(ctx, upsertReq)
	if err != nil {
		t.Fatalf("UpsertItemFeatures failed over h2c: %v", err)
	}
	if upsertResp.Msg.GetUpsertedCount() != 1 {
		t.Fatalf("expected upserted_count=1, got %d", upsertResp.Msg.GetUpsertedCount())
	}

	getReq := connect.NewRequest(&vfsv1.GetItemFeaturesRequest{ItemId: "item-lt-001"})
	getReq.Header().Set(TargetEngineHeader, "pgvector")
	getResp, err := rpcClient.GetItemFeatures(ctx, getReq)
	if err != nil {
		t.Fatalf("GetItemFeatures failed: %v", err)
	}
	if !getResp.Msg.GetFound() || getResp.Msg.GetItem().GetBrandId() != "brand-nike" {
		t.Fatalf("unexpected GetItemFeatures response: %+v", getResp.Msg)
	}

	batchReq := connect.NewRequest(&vfsv1.BatchGetItemFeaturesRequest{ItemIds: []string{"item-lt-001", "missing-id"}})
	batchResp, err := rpcClient.BatchGetItemFeatures(ctx, batchReq)
	if err != nil {
		t.Fatalf("BatchGetItemFeatures failed: %v", err)
	}
	if len(batchResp.Msg.GetItems()) != 1 || len(batchResp.Msg.GetMissingItemIds()) != 1 {
		t.Fatalf("unexpected BatchGetItemFeatures counts: %+v", batchResp.Msg)
	}

	searchReq := connect.NewRequest(&vfsv1.SearchSimilarItemsRequest{
		QueryEmbedding:  makeTestEmbedding(0.25),
		TopK:            5,
		MaxPriceEur:     50.0,
		SellerCountries: []string{"FR", "ES"},
	})
	searchResp, err := rpcClient.SearchSimilarItems(ctx, searchReq)
	if err != nil {
		t.Fatalf("SearchSimilarItems failed: %v", err)
	}
	if len(searchResp.Msg.GetHits()) != 1 {
		t.Fatalf("expected 1 ANN hit, got %d", len(searchResp.Msg.GetHits()))
	}

	searchCountryReq := connect.NewRequest(&vfsv1.SearchSimilarItemsRequest{
		QueryEmbedding:  makeTestEmbedding(0.25),
		TopK:            5,
		MaxPriceEur:     0.0,
		SellerCountries: []string{"FR"},
	})
	searchCountryResp, err := rpcClient.SearchSimilarItems(ctx, searchCountryReq)
	if err != nil {
		t.Fatalf("SearchSimilarItems country-only failed: %v", err)
	}
	if len(searchCountryResp.Msg.GetHits()) != 1 {
		t.Fatalf("expected 1 hit for country-only filter FR, got %d", len(searchCountryResp.Msg.GetHits()))
	}

	inactiveReq := connect.NewRequest(&vfsv1.GetItemFeaturesRequest{ItemId: "item-lt-001"})
	inactiveReq.Header().Set(TargetEngineHeader, "vespa")
	_, err = rpcClient.GetItemFeatures(ctx, inactiveReq)
	if connect.CodeOf(err) != connect.CodeUnavailable {
		t.Fatalf("expected CodeUnavailable for inactive engine vespa, got %v", err)
	}

	unknownReq := connect.NewRequest(&vfsv1.GetItemFeaturesRequest{ItemId: "item-lt-001"})
	unknownReq.Header().Set(TargetEngineHeader, "untrusted-engine-header-999")
	_, err = rpcClient.GetItemFeatures(ctx, unknownReq)
	if connect.CodeOf(err) != connect.CodeInvalidArgument {
		t.Fatalf("expected CodeInvalidArgument for unknown engine header, got %v", err)
	}

	metricsResp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics failed: %v", err)
	}
	defer metricsResp.Body.Close()
	bodyBytes, err := io.ReadAll(metricsResp.Body)
	if err != nil {
		t.Fatalf("read /metrics failed: %v", err)
	}
	metricsText := string(bodyBytes)
	if !strings.Contains(metricsText, `vfs_request_duration_seconds_bucket{engine="pgvector",rpc="GetItemFeatures",status="ok",le="0.025"}`) {
		t.Fatalf("expected 0.025s SLA bucket in /metrics output:\n%s", metricsText)
	}
	if !strings.Contains(metricsText, `vfs_request_duration_seconds_count{engine="unknown",rpc="GetItemFeatures",status="error"} 1`) {
		t.Fatalf("expected unknown engine error count in /metrics output:\n%s", metricsText)
	}
	if strings.Contains(metricsText, "untrusted-engine-header-999") {
		t.Fatalf("untrusted engine header leaked into Prometheus metrics:\n%s", metricsText)
	}
}

func TestLivezEndpoint(t *testing.T) {
	backend := newInMemoryBackend()
	telemetry := NewProxyTelemetry("pgvector")
	proxy := NewFeatureStoreProxy("pgvector", map[string]EngineBackend{"pgvector": backend}, telemetry)

	srv := httptest.NewServer(NewProxyHTTPHandler(proxy))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/livez")
	if err != nil {
		t.Fatalf("GET /livez failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK from /livez, got %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read /livez body failed: %v", err)
	}
	if !strings.Contains(string(body), `"alive"`) {
		t.Fatalf("unexpected /livez response body: %s", string(body))
	}
}

func TestItemIDToUUID(t *testing.T) {
	id1 := "item-000001"
	u1 := itemIDToUUID(id1)
	u2 := itemIDToUUID(id1)
	if u1 != u2 {
		t.Fatalf("itemIDToUUID is not deterministic: %s != %s", u1, u2)
	}
	if len(u1) != 36 {
		t.Fatalf("expected 36 characters, got %d (%s)", len(u1), u1)
	}
	// Verify UUID format xxxxxxxx-xxxx-5xxx-yxxx-xxxxxxxxxxxx
	if u1[8] != '-' || u1[13] != '-' || u1[18] != '-' || u1[23] != '-' {
		t.Fatalf("malformed UUID hyphen layout: %s", u1)
	}
	if u1[14] != '5' {
		t.Fatalf("expected version 5 UUID, got char %c at index 14 (%s)", u1[14], u1)
	}
	idDifferent := "item-000002"
	uDiff := itemIDToUUID(idDifferent)
	if u1 == uDiff {
		t.Fatalf("distinct item IDs produced identical UUID: %s", u1)
	}
}

func TestQdrantBackendMock(t *testing.T) {
	mux := http.NewServeMux()

	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"title":"qdrant"}`))
	})
	mux.HandleFunc("/collections/", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result":{"status":"green"},"status":"ok"}`))
	})

	healthHandler := connect.NewUnaryHandler(
		"/qdrant.Qdrant/HealthCheck",
		func(ctx context.Context, req *connect.Request[rawBytes]) (*connect.Response[rawBytes], error) {
			res := rawBytes{}
			return connect.NewResponse(&res), nil
		},
		connect.WithCodec(rawProtoCodec{}),
	)
	upsertHandler := connect.NewUnaryHandler(
		"/qdrant.Points/Upsert",
		func(ctx context.Context, req *connect.Request[rawBytes]) (*connect.Response[rawBytes], error) {
			res := rawBytes{}
			return connect.NewResponse(&res), nil
		},
		connect.WithCodec(rawProtoCodec{}),
	)
	getHandler := connect.NewUnaryHandler(
		"/qdrant.Points/Get",
		func(ctx context.Context, req *connect.Request[rawBytes]) (*connect.Response[rawBytes], error) {
			testItem := &vfsv1.ItemFeatures{
				ItemId:          "item-qd-001",
				Embedding:       makeTestEmbedding(0.12),
				BrandId:         "brand-01",
				Category:        "jackets",
				PriceEur:        45.0,
				SellerCountry:   "FR",
				UpdatedAtUnixMs: 1700000000000,
			}
			var pt []byte
			var pid []byte
			pid = appendStringField(pid, 2, itemIDToUUID(testItem.GetItemId()))
			pt = appendBytesField(pt, 1, pid)
			pt = appendPayloadEntry(pt, 2, "item_id", testItem.GetItemId())
			pt = appendPayloadEntry(pt, 2, "brand_id", testItem.GetBrandId())
			pt = appendPayloadEntry(pt, 2, "category", testItem.GetCategory())
			pt = appendPayloadDouble(pt, 2, "price_eur", float64(testItem.GetPriceEur()))
			pt = appendPayloadEntry(pt, 2, "seller_country", testItem.GetSellerCountry())
			pt = appendPayloadInt(pt, 2, "updated_at_unix_ms", testItem.GetUpdatedAtUnixMs())

			var vecMsg []byte
			vecMsg = appendFloatSlice(vecMsg, 1, testItem.GetEmbedding())
			var vo []byte
			vo = appendBytesField(vo, 1, vecMsg)
			pt = appendBytesField(pt, 4, vo)

			var getResp []byte
			getResp = appendBytesField(getResp, 1, pt)
			res := rawBytes(getResp)
			return connect.NewResponse(&res), nil
		},
		connect.WithCodec(rawProtoCodec{}),
	)
	searchHandler := connect.NewUnaryHandler(
		"/qdrant.Points/Search",
		func(ctx context.Context, req *connect.Request[rawBytes]) (*connect.Response[rawBytes], error) {
			testItem := &vfsv1.ItemFeatures{
				ItemId:          "item-qd-001",
				BrandId:         "brand-01",
				Category:        "jackets",
				PriceEur:        45.0,
				SellerCountry:   "FR",
				UpdatedAtUnixMs: 1700000000000,
			}
			var hit []byte
			hit = appendPayloadEntry(hit, 2, "item_id", testItem.GetItemId())
			hit = appendPayloadEntry(hit, 2, "brand_id", testItem.GetBrandId())
			hit = appendPayloadEntry(hit, 2, "category", testItem.GetCategory())
			hit = appendPayloadDouble(hit, 2, "price_eur", float64(testItem.GetPriceEur()))
			hit = appendPayloadEntry(hit, 2, "seller_country", testItem.GetSellerCountry())
			hit = appendPayloadInt(hit, 2, "updated_at_unix_ms", testItem.GetUpdatedAtUnixMs())
			hit = protowire.AppendTag(hit, 3, protowire.Fixed32Type)
			hit = protowire.AppendFixed32(hit, math.Float32bits(0.95))

			var searchResp []byte
			searchResp = appendBytesField(searchResp, 1, hit)
			res := rawBytes(searchResp)
			return connect.NewResponse(&res), nil
		},
		connect.WithCodec(rawProtoCodec{}),
	)

	mux.Handle("/qdrant.Qdrant/HealthCheck", healthHandler)
	mux.Handle("/qdrant.Points/Upsert", upsertHandler)
	mux.Handle("/qdrant.Points/Get", getHandler)
	mux.Handle("/qdrant.Points/Search", searchHandler)

	h2s := &http2.Server{}
	mockQdrant := httptest.NewServer(h2c.NewHandler(mux, h2s))
	defer mockQdrant.Close()

	ctx := context.Background()
	backend, err := NewQdrantBackend(ctx, mockQdrant.URL, mockQdrant.URL, "items")
	if err != nil {
		t.Fatalf("NewQdrantBackend failed: %v", err)
	}
	defer backend.Close()

	if backend.Name() != "qdrant" {
		t.Fatalf("expected Name() == 'qdrant', got %s", backend.Name())
	}
	if err := backend.Ping(ctx); err != nil {
		t.Fatalf("backend.Ping failed: %v", err)
	}

	testItem := &vfsv1.ItemFeatures{
		ItemId:          "item-qd-001",
		Embedding:       makeTestEmbedding(0.12),
		BrandId:         "brand-01",
		Category:        "jackets",
		PriceEur:        45.0,
		SellerCountry:   "FR",
		UpdatedAtUnixMs: 1700000000000,
	}

	upserted, err := backend.UpsertItems(ctx, []*vfsv1.ItemFeatures{testItem})
	if err != nil {
		t.Fatalf("UpsertItems failed: %v", err)
	}
	if upserted != 1 {
		t.Fatalf("expected upserted count 1, got %d", upserted)
	}

	got, found, err := backend.GetItem(ctx, "item-qd-001")
	if err != nil {
		t.Fatalf("GetItem failed: %v", err)
	}
	if !found || got.GetItemId() != "item-qd-001" {
		t.Fatalf("GetItem failed to retrieve item, found=%v, item=%v", found, got)
	}

	hits, err := backend.SearchSimilar(ctx, makeTestEmbedding(0.12), 10, 50.0, []string{"FR"})
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	if len(hits) != 1 || hits[0].GetItem().GetItemId() != "item-qd-001" {
		t.Fatalf("unexpected SearchSimilar hits: %v", hits)
	}
}

func TestVespaBackendMock(t *testing.T) {
	type vespaDocFields struct {
		ItemID          string  `json:"item_id"`
		BrandID         string  `json:"brand_id"`
		Category        string  `json:"category"`
		PriceEur        float32 `json:"price_eur"`
		SellerCountry   string  `json:"seller_country"`
		UpdatedAtUnixMs int64   `json:"updated_at_unix_ms"`
		Embedding       struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}

	docs := make(map[string]vespaDocFields)
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		// Health endpoint
		if r.URL.Path == "/state/v1/health" {
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]any{"status": map[string]string{"code": "up"}})
			return
		}

		// Config server deploy endpoint
		if strings.Contains(r.URL.Path, "/prepareandactivate") {
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]string{"message": "activated"})
			return
		}

		// Document API: /document/v1/vfs/item/docid/{id}
		if strings.HasPrefix(r.URL.Path, "/document/v1/vfs/item/docid/") {
			id := strings.TrimPrefix(r.URL.Path, "/document/v1/vfs/item/docid/")
			switch r.Method {
			case http.MethodGet:
				doc, exists := docs[id]
				if !exists {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				w.WriteHeader(http.StatusOK)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"fields": doc,
				})
				return
			case http.MethodPost:
				var body struct {
					Fields vespaDocFields `json:"fields"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				docs[id] = body.Fields
				w.WriteHeader(http.StatusOK)
				_ = json.NewEncoder(w).Encode(map[string]string{"id": id})
				return
			}
		}

		// Search API: /search/
		if r.URL.Path == "/search/" && r.Method == http.MethodPost {
			var children []map[string]any
			for _, doc := range docs {
				children = append(children, map[string]any{
					"relevance": 0.98,
					"fields": map[string]any{
						"item_id":             doc.ItemID,
						"brand_id":            doc.BrandID,
						"category":            doc.Category,
						"price_eur":           doc.PriceEur,
						"seller_country":      doc.SellerCountry,
						"updated_at_unix_ms":  doc.UpdatedAtUnixMs,
					},
				})
			}
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"root": map[string]any{
					"children": children,
				},
			})
			return
		}

		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	backend, err := NewVespaBackend(ctx, server.URL, server.URL, "vfs", "item")
	if err != nil {
		t.Fatalf("failed to create VespaBackend: %v", err)
	}
	defer backend.Close()

	if backend.Name() != "vespa" {
		t.Fatalf("expected backend name 'vespa', got %q", backend.Name())
	}

	if err := backend.Ping(ctx); err != nil {
		t.Fatalf("Ping failed: %v", err)
	}

	testItem := &vfsv1.ItemFeatures{
		ItemId:          "item-vp-001",
		Embedding:       makeTestEmbedding(0.42),
		BrandId:         "brand-42",
		Category:        "shoes",
		PriceEur:        39.99,
		SellerCountry:   "FR",
		UpdatedAtUnixMs: 1700000000000,
	}

	upserted, err := backend.UpsertItems(ctx, []*vfsv1.ItemFeatures{testItem})
	if err != nil {
		t.Fatalf("UpsertItems failed: %v", err)
	}
	if upserted != 1 {
		t.Fatalf("expected upserted 1, got %d", upserted)
	}

	got, found, err := backend.GetItem(ctx, "item-vp-001")
	if err != nil {
		t.Fatalf("GetItem failed: %v", err)
	}
	if !found || got.GetItemId() != "item-vp-001" {
		t.Fatalf("GetItem failed, found=%v, item=%v", found, got)
	}

	items, missing, err := backend.BatchGetItems(ctx, []string{"item-vp-001", "missing-doc"})
	if err != nil {
		t.Fatalf("BatchGetItems failed: %v", err)
	}
	if len(items) != 1 || len(missing) != 1 {
		t.Fatalf("unexpected BatchGetItems result: items=%d, missing=%d", len(items), len(missing))
	}

	hits, err := backend.SearchSimilar(ctx, makeTestEmbedding(0.42), 10, 50.0, []string{"FR"})
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	if len(hits) != 1 || hits[0].GetItem().GetItemId() != "item-vp-001" {
		t.Fatalf("unexpected SearchSimilar hits: %v", hits)
	}
	if hits[0].GetCosineSimilarity() != 0.98 {
		t.Fatalf("expected CosineSimilarity 0.98, got %f", hits[0].GetCosineSimilarity())
	}
}

func TestMilvusBackendMock(t *testing.T) {
	docs := make(map[string]milvusEntityDoc)
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v2/vectordb/collections/has":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"code": 0,
				"data": map[string]any{"has": true},
			})
			return
		case "/v2/vectordb/collections/create", "/v2/vectordb/collections/load":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"code": 0,
			})
			return
		case "/v2/vectordb/entities/upsert":
			var req milvusUpsertRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			for _, doc := range req.Data {
				docs[doc.ItemID] = doc
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"code": 0,
				"data": map[string]any{"upsertCount": len(req.Data)},
			})
			return
		case "/v2/vectordb/entities/get":
			var req milvusGetRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			var found []milvusEntityDoc
			for _, id := range req.ID {
				if d, ok := docs[id]; ok {
					found = append(found, d)
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"code": 0,
				"data": found,
			})
			return
		case "/v2/vectordb/entities/search":
			var hits []milvusEntityDoc
			for _, d := range docs {
				d.Distance = 0.95
				hits = append(hits, d)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"code": 0,
				"data": hits,
			})
			return
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	backend, err := NewMilvusBackend(ctx, server.URL, "items")
	if err != nil {
		t.Fatalf("failed to create MilvusBackend: %v", err)
	}
	defer backend.Close()

	if backend.Name() != "milvus" {
		t.Fatalf("expected backend name 'milvus', got %q", backend.Name())
	}

	if err := backend.Ping(ctx); err != nil {
		t.Fatalf("Ping failed: %v", err)
	}

	testItem := &vfsv1.ItemFeatures{
		ItemId:          "item-mv-001",
		Embedding:       makeTestEmbedding(0.55),
		BrandId:         "brand-mv-1",
		Category:        "bags",
		PriceEur:        49.50,
		SellerCountry:   "FR",
		UpdatedAtUnixMs: 1700000000000,
	}

	upserted, err := backend.UpsertItems(ctx, []*vfsv1.ItemFeatures{testItem})
	if err != nil {
		t.Fatalf("UpsertItems failed: %v", err)
	}
	if upserted != 1 {
		t.Fatalf("expected upserted 1, got %d", upserted)
	}

	got, found, err := backend.GetItem(ctx, "item-mv-001")
	if err != nil {
		t.Fatalf("GetItem failed: %v", err)
	}
	if !found || got.GetItemId() != "item-mv-001" {
		t.Fatalf("GetItem failed, found=%v, item=%v", found, got)
	}

	items, missing, err := backend.BatchGetItems(ctx, []string{"item-mv-001", "missing-id"})
	if err != nil {
		t.Fatalf("BatchGetItems failed: %v", err)
	}
	if len(items) != 1 || len(missing) != 1 {
		t.Fatalf("unexpected BatchGetItems result: items=%d, missing=%d", len(items), len(missing))
	}

	hits, err := backend.SearchSimilar(ctx, makeTestEmbedding(0.55), 10, 50.0, []string{"FR"})
	if err != nil {
		t.Fatalf("SearchSimilar failed: %v", err)
	}
	if len(hits) != 1 || hits[0].GetItem().GetItemId() != "item-mv-001" {
		t.Fatalf("unexpected SearchSimilar hits: %v", hits)
	}
	if hits[0].GetCosineSimilarity() != 0.95 {
		t.Fatalf("expected CosineSimilarity 0.95, got %f", hits[0].GetCosineSimilarity())
	}
}
