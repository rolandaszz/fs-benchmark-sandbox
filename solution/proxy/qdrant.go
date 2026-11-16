package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
)

// QdrantBackend implements EngineBackend against Qdrant's REST API.
type QdrantBackend struct {
	baseURL    string
	collection string
	client     *http.Client
}

type qdrantItemPayload struct {
	ItemID          string  `json:"item_id"`
	BrandID         string  `json:"brand_id"`
	Category        string  `json:"category"`
	PriceEur        float32 `json:"price_eur"`
	SellerCountry   string  `json:"seller_country"`
	UpdatedAtUnixMs int64   `json:"updated_at_unix_ms"`
}

type qdrantPoint struct {
	ID      string            `json:"id"`
	Vector  []float32         `json:"vector,omitempty"`
	Payload qdrantItemPayload `json:"payload"`
	Score   float32           `json:"score,omitempty"`
}

type qdrantGenericResponse struct {
	Status string  `json:"status"`
	Time   float64 `json:"time,omitempty"`
}

type qdrantPointsResponse struct {
	Result []qdrantPoint `json:"result"`
	Status string        `json:"status"`
}

type qdrantSearchResponse struct {
	Result []qdrantPoint `json:"result"`
	Status string        `json:"status"`
}

// itemIDToUUID deterministically hashes an item ID into a valid UUIDv5 format required by Qdrant.
func itemIDToUUID(itemID string) string {
	h := sha1.New()
	h.Write([]byte("vfs.v1.item:" + itemID))
	b := h.Sum(nil)[:16]
	b[6] = (b[6] & 0x0f) | 0x50 // version 5
	b[8] = (b[8] & 0x3f) | 0x80 // RFC 4122 variant
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// NewQdrantBackend initializes and verifies the Qdrant collection and payload indexes.
func NewQdrantBackend(ctx context.Context, baseURL, collection string) (*QdrantBackend, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	if collection == "" {
		collection = "items"
	}
	backend := &QdrantBackend{
		baseURL:    baseURL,
		collection: collection,
		client: &http.Client{
			Timeout: 5 * time.Second,
			Transport: &http.Transport{
				DisableKeepAlives: true, // Prevents actix-web 40ms delayed ACK on reused HTTP/1.1 sockets
			},
		},
	}

	if err := backend.EnsureCollection(ctx); err != nil {
		return nil, fmt.Errorf("ensure qdrant collection %q: %w", collection, err)
	}
	return backend, nil
}

func (b *QdrantBackend) Name() string {
	return "qdrant"
}

func (b *QdrantBackend) Close() {
	b.client.CloseIdleConnections()
}

func (b *QdrantBackend) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.baseURL+"/readyz", nil)
	if err != nil {
		return err
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant readyz status %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

// EnsureCollection verifies collection existence or creates it with HNSW cosine config and indexes.
func (b *QdrantBackend) EnsureCollection(ctx context.Context) error {
	collURL := fmt.Sprintf("%s/collections/%s", b.baseURL, b.collection)

	checkReq, err := http.NewRequestWithContext(ctx, http.MethodGet, collURL, nil)
	if err != nil {
		return err
	}
	resp, err := b.client.Do(checkReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		createBody := map[string]any{
			"vectors": map[string]any{
				"size":     EmbeddingDimension,
				"distance": "Cosine",
			},
			"hnsw_config": map[string]any{
				"m":                   16,
				"ef_construct":        64,
				"full_scan_threshold": 100,
			},
			"optimizer_config": map[string]any{
				"indexing_threshold": 100,
			},
		}
		raw, err := json.Marshal(createBody)
		if err != nil {
			return err
		}
		createReq, err := http.NewRequestWithContext(ctx, http.MethodPut, collURL, bytes.NewReader(raw))
		if err != nil {
			return err
		}
		createReq.Header.Set("Content-Type", "application/json")
		createResp, err := b.client.Do(createReq)
		if err != nil {
			return err
		}
		defer createResp.Body.Close()
		if createResp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(createResp.Body)
			return fmt.Errorf("create collection status %d: %s", createResp.StatusCode, string(body))
		}
	} else if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("check collection status %d: %s", resp.StatusCode, string(body))
	}

	// Ensure payload indexes
	indexes := []struct {
		name   string
		schema string
	}{
		{name: "seller_country", schema: "keyword"},
		{name: "price_eur", schema: "float"},
	}

	for _, idx := range indexes {
		idxURL := fmt.Sprintf("%s/collections/%s/index?wait=true", b.baseURL, b.collection)
		idxBody := map[string]any{
			"field_name":   idx.name,
			"field_schema": idx.schema,
		}
		raw, err := json.Marshal(idxBody)
		if err != nil {
			return err
		}
		idxReq, err := http.NewRequestWithContext(ctx, http.MethodPut, idxURL, bytes.NewReader(raw))
		if err != nil {
			return err
		}
		idxReq.Header.Set("Content-Type", "application/json")
		idxResp, err := b.client.Do(idxReq)
		if err != nil {
			return err
		}
		idxResp.Body.Close()
		// Qdrant returns 200 on success or if index already exists
	}

	return nil
}

func (b *QdrantBackend) UpsertItems(ctx context.Context, items []*vfsv1.ItemFeatures) (uint32, error) {
	if len(items) == 0 {
		return 0, nil
	}

	points := make([]qdrantPoint, 0, len(items))
	nowMs := time.Now().UTC().UnixMilli()
	for _, item := range items {
		ts := item.GetUpdatedAtUnixMs()
		if ts <= 0 {
			ts = nowMs
		}
		points = append(points, qdrantPoint{
			ID:     itemIDToUUID(item.GetItemId()),
			Vector: item.GetEmbedding(),
			Payload: qdrantItemPayload{
				ItemID:          item.GetItemId(),
				BrandID:         item.GetBrandId(),
				Category:        item.GetCategory(),
				PriceEur:        item.GetPriceEur(),
				SellerCountry:   item.GetSellerCountry(),
				UpdatedAtUnixMs: ts,
			},
		})
	}

	body := map[string]any{
		"points": points,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return 0, fmt.Errorf("marshal points: %w", err)
	}

	upsertURL := fmt.Sprintf("%s/collections/%s/points?wait=true", b.baseURL, b.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, upsertURL, bytes.NewReader(raw))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("upsert points status %d: %s", resp.StatusCode, string(respBody))
	}

	return uint32(len(items)), nil
}

func (b *QdrantBackend) GetItem(ctx context.Context, itemID string) (*vfsv1.ItemFeatures, bool, error) {
	items, _, err := b.BatchGetItems(ctx, []string{itemID})
	if err != nil {
		return nil, false, err
	}
	if len(items) == 1 {
		return items[0], true, nil
	}
	return nil, false, nil
}

func (b *QdrantBackend) BatchGetItems(ctx context.Context, itemIDs []string) ([]*vfsv1.ItemFeatures, []string, error) {
	if len(itemIDs) == 0 {
		return nil, nil, nil
	}

	uuids := make([]string, len(itemIDs))
	uuidToItemID := make(map[string]string, len(itemIDs))
	for i, id := range itemIDs {
		u := itemIDToUUID(id)
		uuids[i] = u
		uuidToItemID[u] = id
	}

	body := map[string]any{
		"ids":          uuids,
		"with_payload": true,
		"with_vector":  true,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal get points req: %w", err)
	}

	getURL := fmt.Sprintf("%s/collections/%s/points", b.baseURL, b.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, getURL, bytes.NewReader(raw))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("get points status %d: %s", resp.StatusCode, string(respBody))
	}

	var res qdrantPointsResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, nil, fmt.Errorf("decode get points response: %w", err)
	}

	foundMap := make(map[string]*vfsv1.ItemFeatures, len(res.Result))
	for _, p := range res.Result {
		origID := p.Payload.ItemID
		if origID == "" {
			origID = uuidToItemID[p.ID]
		}
		foundMap[origID] = &vfsv1.ItemFeatures{
			ItemId:          origID,
			Embedding:       p.Vector,
			BrandId:         p.Payload.BrandID,
			Category:        p.Payload.Category,
			PriceEur:        p.Payload.PriceEur,
			SellerCountry:   p.Payload.SellerCountry,
			UpdatedAtUnixMs: p.Payload.UpdatedAtUnixMs,
		}
	}

	items := make([]*vfsv1.ItemFeatures, 0, len(itemIDs))
	var missing []string
	for _, id := range itemIDs {
		if it, ok := foundMap[id]; ok {
			items = append(items, it)
		} else {
			missing = append(missing, id)
		}
	}

	return items, missing, nil
}

func (b *QdrantBackend) SearchSimilar(
	ctx context.Context,
	queryVec []float32,
	topK uint32,
	maxPrice float32,
	countries []string,
) ([]*vfsv1.ScoredItemFeatures, error) {
	if topK == 0 {
		topK = 10
	}

	searchBody := map[string]any{
		"vector":       queryVec,
		"limit":        topK,
		"with_payload": true,
		"with_vector":  false, // Omit embeddings in search results to avoid massive serialization overhead
	}

	var mustConditions []map[string]any
	if maxPrice > 0 {
		mustConditions = append(mustConditions, map[string]any{
			"key": "price_eur",
			"range": map[string]any{
				"lte": maxPrice,
			},
		})
	}
	if len(countries) > 0 {
		mustConditions = append(mustConditions, map[string]any{
			"key": "seller_country",
			"match": map[string]any{
				"any": countries,
			},
		})
	}

	if len(mustConditions) > 0 {
		searchBody["filter"] = map[string]any{
			"must": mustConditions,
		}
	}

	raw, err := json.Marshal(searchBody)
	if err != nil {
		return nil, fmt.Errorf("marshal search req: %w", err)
	}

	searchURL := fmt.Sprintf("%s/collections/%s/points/search", b.baseURL, b.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search points status %d: %s", resp.StatusCode, string(respBody))
	}

	var res qdrantSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	hits := make([]*vfsv1.ScoredItemFeatures, len(res.Result))
	for i, hit := range res.Result {
		hits[i] = &vfsv1.ScoredItemFeatures{
			Item: &vfsv1.ItemFeatures{
				ItemId:          hit.Payload.ItemID,
				BrandId:         hit.Payload.BrandID,
				Category:        hit.Payload.Category,
				PriceEur:        hit.Payload.PriceEur,
				SellerCountry:   hit.Payload.SellerCountry,
				UpdatedAtUnixMs: hit.Payload.UpdatedAtUnixMs,
			},
			CosineSimilarity: hit.Score,
		}
	}

	return hits, nil
}
