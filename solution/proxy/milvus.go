package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
)

// MilvusBackend implements EngineBackend for Milvus 2.4+ Standalone via v2 REST API.
type MilvusBackend struct {
	baseURL    string
	collection string
	client     *http.Client
}

func NewMilvusBackend(ctx context.Context, baseURL, collection string) (*MilvusBackend, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	if collection == "" {
		collection = "items"
	}

	tr := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
	}
	client := &http.Client{
		Transport: tr,
		Timeout:   10 * time.Second,
	}

	b := &MilvusBackend{
		baseURL:    baseURL,
		collection: collection,
		client:     client,
	}

	if err := b.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("ensure milvus collection: %w", err)
	}

	return b, nil
}

func (b *MilvusBackend) Name() string {
	return "milvus"
}

func (b *MilvusBackend) Ping(ctx context.Context) error {
	type hasResp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Data    struct {
			Has bool `json:"has"`
		} `json:"data"`
	}
	var res hasResp
	reqBody := map[string]string{"collectionName": b.collection}
	if err := b.doJSON(ctx, "/v2/vectordb/collections/has", reqBody, &res); err != nil {
		return err
	}
	if res.Code != 0 {
		return fmt.Errorf("milvus ping error %d: %s", res.Code, res.Message)
	}
	return nil
}

func (b *MilvusBackend) Close() {
	b.client.CloseIdleConnections()
}

func (b *MilvusBackend) doJSON(ctx context.Context, path string, reqBody any, respOut any) error {
	var bodyReader io.Reader
	if reqBody != nil {
		raw, err := json.Marshal(reqBody)
		if err != nil {
			return fmt.Errorf("marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(raw)
	}

	url := b.baseURL + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bodyReader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("milvus request %s status %d: %s", path, resp.StatusCode, string(respBytes))
	}

	if respOut != nil {
		if err := json.Unmarshal(respBytes, respOut); err != nil {
			return fmt.Errorf("unmarshal response from %s: %w", path, err)
		}
	}
	return nil
}

type milvusElementParams struct {
	MaxLength string `json:"max_length,omitempty"`
	Dim       string `json:"dim,omitempty"`
}

type milvusFieldSchema struct {
	FieldName         string               `json:"fieldName"`
	DataType          string               `json:"dataType"`
	IsPrimary         bool                 `json:"isPrimary,omitempty"`
	IsPartitionKey    bool                 `json:"isPartitionKey,omitempty"`
	ElementTypeParams *milvusElementParams `json:"elementTypeParams,omitempty"`
}

type milvusIndexParam struct {
	FieldName  string         `json:"fieldName"`
	MetricType string         `json:"metricType,omitempty"`
	IndexName  string         `json:"indexName"`
	IndexType  string         `json:"indexType"`
	Params     map[string]any `json:"params,omitempty"`
}

type milvusCreateCollectionRequest struct {
	CollectionName string             `json:"collectionName"`
	Schema         map[string]any     `json:"schema"`
	IndexParams    []milvusIndexParam `json:"indexParams"`
}

func (b *MilvusBackend) ensureCollection(ctx context.Context) error {
	type hasResp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Data    struct {
			Has bool `json:"has"`
		} `json:"data"`
	}

	var has hasResp
	reqHas := map[string]string{"collectionName": b.collection}

	// Poll until Milvus API is online
	ready := false
	for attempt := 1; attempt <= 30; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := b.doJSON(ctx, "/v2/vectordb/collections/has", reqHas, &has); err == nil && has.Code == 0 {
			ready = true
			break
		}
		time.Sleep(1 * time.Second)
	}

	if !ready {
		return fmt.Errorf("milvus did not become ready within 30s")
	}

	if has.Data.Has {
		slog.Info("milvus collection already exists", "collection", b.collection)
		return nil
	}

	// Create collection with schema, scalar indexes, and HNSW index
	createReq := milvusCreateCollectionRequest{
		CollectionName: b.collection,
		Schema: map[string]any{
			"fields": []milvusFieldSchema{
				{
					FieldName: "item_id",
					DataType:  "VarChar",
					IsPrimary: true,
					ElementTypeParams: &milvusElementParams{
						MaxLength: "128",
					},
				},
				{
					FieldName: "embedding",
					DataType:  "FloatVector",
					ElementTypeParams: &milvusElementParams{
						Dim: "768",
					},
				},
				{
					FieldName: "brand_id",
					DataType:  "VarChar",
					ElementTypeParams: &milvusElementParams{
						MaxLength: "64",
					},
				},
				{
					FieldName: "category",
					DataType:  "VarChar",
					ElementTypeParams: &milvusElementParams{
						MaxLength: "64",
					},
				},
				{
					FieldName: "price_eur",
					DataType:  "Float",
				},
				{
					FieldName:      "seller_country",
					DataType:       "VarChar",
					ElementTypeParams: &milvusElementParams{
						MaxLength: "8",
					},
				},
				{
					FieldName: "updated_at_unix_ms",
					DataType:  "Int64",
				},
			},
		},
		IndexParams: []milvusIndexParam{
			{
				FieldName:  "embedding",
				MetricType: "COSINE",
				IndexName:  "embedding_hnsw",
				IndexType:  "HNSW",
				Params: map[string]any{
					"M":              16,
					"efConstruction": 64,
				},
			},
			{
				FieldName: "seller_country",
				IndexName: "seller_country_inv",
				IndexType: "INVERTED",
			},
			{
				FieldName: "price_eur",
				IndexName: "price_eur_inv",
				IndexType: "INVERTED",
			},
		},
	}

	type genericResp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}

	var createOut genericResp
	if err := b.doJSON(ctx, "/v2/vectordb/collections/create", createReq, &createOut); err != nil {
		return fmt.Errorf("create collection: %w", err)
	}
	if createOut.Code != 0 {
		return fmt.Errorf("create collection error %d: %s", createOut.Code, createOut.Message)
	}

	var loadOut genericResp
	loadReq := map[string]string{"collectionName": b.collection}
	if err := b.doJSON(ctx, "/v2/vectordb/collections/load", loadReq, &loadOut); err != nil {
		return fmt.Errorf("load collection: %w", err)
	}
	if loadOut.Code != 0 {
		return fmt.Errorf("load collection error %d: %s", loadOut.Code, loadOut.Message)
	}

	slog.Info("milvus collection created and loaded successfully", "collection", b.collection)
	return nil
}

type milvusEntityDoc struct {
	ItemID          string    `json:"item_id"`
	Embedding       []float32 `json:"embedding,omitempty"`
	BrandID         string    `json:"brand_id"`
	Category        string    `json:"category"`
	PriceEur        float32   `json:"price_eur"`
	SellerCountry   string    `json:"seller_country"`
	UpdatedAtUnixMs int64     `json:"updated_at_unix_ms"`
	Distance        float32   `json:"distance,omitempty"`
}

type milvusUpsertRequest struct {
	CollectionName string            `json:"collectionName"`
	Data           []milvusEntityDoc `json:"data"`
}

func (b *MilvusBackend) UpsertItems(ctx context.Context, items []*vfsv1.ItemFeatures) (uint32, error) {
	if len(items) == 0 {
		return 0, nil
	}

	nowMs := time.Now().UTC().UnixMilli()
	docs := make([]milvusEntityDoc, len(items))
	for i, it := range items {
		ts := it.GetUpdatedAtUnixMs()
		if ts <= 0 {
			ts = nowMs
		}
		docs[i] = milvusEntityDoc{
			ItemID:          it.GetItemId(),
			Embedding:       it.GetEmbedding(),
			BrandID:         it.GetBrandId(),
			Category:        it.GetCategory(),
			PriceEur:        it.GetPriceEur(),
			SellerCountry:   it.GetSellerCountry(),
			UpdatedAtUnixMs: ts,
		}
	}

	req := milvusUpsertRequest{
		CollectionName: b.collection,
		Data:           docs,
	}

	type upsertResp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Data    struct {
			UpsertCount int `json:"upsertCount"`
		} `json:"data"`
	}

	var res upsertResp
	if err := b.doJSON(ctx, "/v2/vectordb/entities/upsert", req, &res); err != nil {
		return 0, fmt.Errorf("upsert entities: %w", err)
	}
	if res.Code != 0 {
		return 0, fmt.Errorf("upsert error %d: %s", res.Code, res.Message)
	}

	return uint32(len(items)), nil
}

type milvusGetRequest struct {
	CollectionName   string   `json:"collectionName"`
	ID               []string `json:"id"`
	OutputFields     []string `json:"outputFields"`
	ConsistencyLevel string   `json:"consistencyLevel,omitempty"`
}

type milvusGetResp struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Data    []milvusEntityDoc `json:"data"`
}

func (b *MilvusBackend) BatchGetItems(ctx context.Context, itemIDs []string) ([]*vfsv1.ItemFeatures, []string, error) {
	if len(itemIDs) == 0 {
		return nil, nil, nil
	}

	req := milvusGetRequest{
		CollectionName: b.collection,
		ID:             itemIDs,
		OutputFields: []string{
			"item_id", "embedding", "brand_id", "category", "price_eur", "seller_country", "updated_at_unix_ms",
		},
		ConsistencyLevel: "Eventually",
	}

	var res milvusGetResp
	if err := b.doJSON(ctx, "/v2/vectordb/entities/get", req, &res); err != nil {
		return nil, nil, fmt.Errorf("get entities: %w", err)
	}
	if res.Code != 0 {
		return nil, nil, fmt.Errorf("get error %d: %s", res.Code, res.Message)
	}

	foundMap := make(map[string]*vfsv1.ItemFeatures, len(res.Data))
	for _, doc := range res.Data {
		foundMap[doc.ItemID] = &vfsv1.ItemFeatures{
			ItemId:          doc.ItemID,
			Embedding:       doc.Embedding,
			BrandId:         doc.BrandID,
			Category:        doc.Category,
			PriceEur:        doc.PriceEur,
			SellerCountry:   doc.SellerCountry,
			UpdatedAtUnixMs: doc.UpdatedAtUnixMs,
		}
	}

	var (
		items   []*vfsv1.ItemFeatures
		missing []string
	)
	for _, id := range itemIDs {
		if it, ok := foundMap[id]; ok {
			items = append(items, it)
		} else {
			missing = append(missing, id)
		}
	}

	return items, missing, nil
}

func (b *MilvusBackend) GetItem(ctx context.Context, itemID string) (*vfsv1.ItemFeatures, bool, error) {
	items, missing, err := b.BatchGetItems(ctx, []string{itemID})
	if err != nil {
		return nil, false, err
	}
	if len(missing) > 0 || len(items) == 0 {
		return nil, false, nil
	}
	return items[0], true, nil
}

type milvusSearchParams struct {
	MetricType string         `json:"metricType"`
	Params     map[string]int `json:"params"`
}

type milvusSearchRequest struct {
	CollectionName   string             `json:"collectionName"`
	Data             [][]float32        `json:"data"`
	AnnsField        string             `json:"annsField"`
	Limit            uint32             `json:"limit"`
	Filter           string             `json:"filter,omitempty"`
	OutputFields     []string           `json:"outputFields"`
	ConsistencyLevel string             `json:"consistencyLevel,omitempty"`
	SearchParams     milvusSearchParams `json:"searchParams"`
}

type milvusSearchResp struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Data    []milvusEntityDoc `json:"data"`
}

func (b *MilvusBackend) SearchSimilar(
	ctx context.Context,
	queryVec []float32,
	topK uint32,
	maxPrice float32,
	countries []string,
) ([]*vfsv1.ScoredItemFeatures, error) {
	if topK == 0 {
		topK = 10
	}

	filterParts := make([]string, 0, 2)
	if maxPrice > 0 {
		filterParts = append(filterParts, fmt.Sprintf("price_eur <= %s", strconv.FormatFloat(float64(maxPrice), 'f', -1, 32)))
	}
	if len(countries) > 0 {
		countryList := make([]string, len(countries))
		for i, c := range countries {
			countryList[i] = fmt.Sprintf("%q", c)
		}
		filterParts = append(filterParts, fmt.Sprintf("seller_country in [%s]", strings.Join(countryList, ", ")))
	}

	filterStr := strings.Join(filterParts, " and ")

	req := milvusSearchRequest{
		CollectionName: b.collection,
		Data:           [][]float32{queryVec},
		AnnsField:      "embedding",
		Limit:          topK,
		Filter:         filterStr,
		OutputFields: []string{
			"item_id", "brand_id", "category", "price_eur", "seller_country", "updated_at_unix_ms",
		},
		ConsistencyLevel: "Eventually",
		SearchParams: milvusSearchParams{
			MetricType: "COSINE",
			Params: map[string]int{
				"ef": 24,
			},
		},
	}

	var res milvusSearchResp
	if err := b.doJSON(ctx, "/v2/vectordb/entities/search", req, &res); err != nil {
		return nil, fmt.Errorf("search entities: %w", err)
	}
	if res.Code != 0 {
		return nil, fmt.Errorf("search error %d: %s", res.Code, res.Message)
	}

	hits := make([]*vfsv1.ScoredItemFeatures, len(res.Data))
	for i, doc := range res.Data {
		hits[i] = &vfsv1.ScoredItemFeatures{
			Item: &vfsv1.ItemFeatures{
				ItemId:          doc.ItemID,
				BrandId:         doc.BrandID,
				Category:        doc.Category,
				PriceEur:        doc.PriceEur,
				SellerCountry:   doc.SellerCountry,
				UpdatedAtUnixMs: doc.UpdatedAtUnixMs,
			},
			CosineSimilarity: doc.Distance,
		}
	}

	return hits, nil
}
