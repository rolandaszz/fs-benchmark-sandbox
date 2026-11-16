package main

import (
	"context"
	"crypto/tls"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"connectrpc.com/connect"
	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
	"github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1/vfsv1connect"
	"golang.org/x/net/http2"
)

type inMemoryBackend struct {
	mu    sync.RWMutex
	items map[string]*vfsv1.ItemFeatures
}

func newInMemoryBackend() *inMemoryBackend {
	return &inMemoryBackend{items: make(map[string]*vfsv1.ItemFeatures)}
}

func (m *inMemoryBackend) Name() string                   { return "pgvector" }
func (m *inMemoryBackend) Ping(_ context.Context) error   { return nil }
func (m *inMemoryBackend) Close()                         {}

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
