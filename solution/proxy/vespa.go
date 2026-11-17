package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
)

const (
	defaultVespaServicesXML = `<?xml version="1.0" encoding="utf-8" ?>
<services version="1.0">
  <container id="default" version="1.0">
    <document-api />
    <search />
    <nodes>
      <jvm options="-Xms1536m -Xmx1536m -XX:+AlwaysPreTouch" />
      <node hostalias="node1" />
    </nodes>
  </container>
  <content id="items" version="1.0">
    <redundancy>1</redundancy>
    <documents>
      <document type="item" mode="index" />
    </documents>
    <nodes>
      <node hostalias="node1" distribution-key="0" />
    </nodes>
    <engine>
      <proton>
        <tuning>
          <searchnode>
            <requestthreads>
              <search>16</search>
              <persearch>2</persearch>
              <summary>8</summary>
            </requestthreads>
          </searchnode>
        </tuning>
      </proton>
    </engine>
    <tuning>
      <resource-limits>
        <disk>0.98</disk>
        <memory>0.95</memory>
      </resource-limits>
    </tuning>
  </content>
</services>`

	defaultVespaItemSD = `schema item {
  document item {
    field item_id type string {
      indexing: summary | attribute
      attribute: fast-search
    }
    field embedding type tensor<float>(x[768]) {
      indexing: attribute | index
      attribute {
        distance-metric: prenormalized-angular
      }
      index {
        hnsw {
          max-links-per-node: 16
          neighbors-to-explore-at-insert: 64
        }
      }
    }
    field brand_id type string {
      indexing: summary | attribute
    }
    field category type string {
      indexing: summary | attribute
    }
    field price_eur type float {
      indexing: summary | attribute
      attribute: fast-search
    }
    field seller_country type string {
      indexing: summary | attribute
      attribute: fast-search
    }
    field updated_at_unix_ms type long {
      indexing: summary | attribute
    }
  }

  document-summary metadata_only {
    summary item_id {}
    summary brand_id {}
    summary category {}
    summary price_eur {}
    summary seller_country {}
    summary updated_at_unix_ms {}
  }

  rank-profile similarity inherits default {
    inputs {
      query(q_emb) tensor<float>(x[768])
    }
    first-phase {
      expression: 1.0 - distance(field, embedding)
    }
  }
}`
)

// VespaBackend implements EngineBackend for Vespa 8.
type VespaBackend struct {
	baseURL   string
	configURL string
	namespace string
	docType   string
	client    *http.Client
}

func NewVespaBackend(ctx context.Context, baseURL, configURL, namespace, docType string) (*VespaBackend, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	configURL = strings.TrimRight(configURL, "/")
	if configURL == "" {
		configURL = "http://localhost:19071"
	}
	if namespace == "" {
		namespace = "vfs"
	}
	if docType == "" {
		docType = "item"
	}

	tr := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true, // Avoid gzip decompression overhead across local loopback
	}
	client := &http.Client{
		Transport: tr,
		Timeout:   10 * time.Second,
	}

	b := &VespaBackend{
		baseURL:   baseURL,
		configURL: configURL,
		namespace: namespace,
		docType:   docType,
		client:    client,
	}

	if err := b.ensureApplication(ctx); err != nil {
		return nil, fmt.Errorf("ensure vespa application: %w", err)
	}

	return b, nil
}

func (b *VespaBackend) Name() string {
	return "vespa"
}

func (b *VespaBackend) Ping(ctx context.Context) error {
	healthURL := fmt.Sprintf("%s/state/v1/health", b.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return err
	}
	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vespa health check returned %d", resp.StatusCode)
	}
	return nil
}

func (b *VespaBackend) Close() {
	b.client.CloseIdleConnections()
}

func (b *VespaBackend) ensureApplication(ctx context.Context) error {
	// Build in-memory application package ZIP
	var zipBuf bytes.Buffer
	zw := zip.NewWriter(&zipBuf)

	servicesFw, err := zw.Create("services.xml")
	if err != nil {
		return fmt.Errorf("create zip services.xml: %w", err)
	}
	if _, err := io.WriteString(servicesFw, defaultVespaServicesXML); err != nil {
		return fmt.Errorf("write zip services.xml: %w", err)
	}

	schemaFw, err := zw.Create("schemas/item.sd")
	if err != nil {
		return fmt.Errorf("create zip schemas/item.sd: %w", err)
	}
	if _, err := io.WriteString(schemaFw, defaultVespaItemSD); err != nil {
		return fmt.Errorf("write zip schemas/item.sd: %w", err)
	}

	if err := zw.Close(); err != nil {
		return fmt.Errorf("close zip writer: %w", err)
	}

	zipBytes := zipBuf.Bytes()
	deployURL := fmt.Sprintf("%s/application/v2/tenant/default/prepareandactivate", b.configURL)
	var lastDeployErr error
	activated := false

	// Retry deployment to Config Server
	for attempt := 1; attempt <= 15; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, deployURL, bytes.NewReader(zipBytes))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/zip")

		resp, err := b.client.Do(req)
		if err == nil {
			body, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				activated = true
				slog.Info("vespa application prepared and activated successfully", "attempt", attempt)
				break
			}
			lastDeployErr = fmt.Errorf("deploy status %d: %s", resp.StatusCode, string(body))
		} else {
			lastDeployErr = err
		}

		time.Sleep(1 * time.Second)
	}

	if !activated {
		return fmt.Errorf("failed to activate vespa application after 15 attempts: %w", lastDeployErr)
	}

	// Poll data container until ready
	for attempt := 1; attempt <= 90; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := b.Ping(ctx); err == nil {
			slog.Info("vespa data container is healthy", "base_url", b.baseURL, "attempt", attempt)
			return nil
		}

		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("vespa data container did not become healthy within 90s")
}

type vespaGetResponse struct {
	Fields struct {
		ItemID          string  `json:"item_id"`
		BrandID         string  `json:"brand_id"`
		Category        string  `json:"category"`
		PriceEur        float32 `json:"price_eur"`
		SellerCountry   string  `json:"seller_country"`
		UpdatedAtUnixMs int64   `json:"updated_at_unix_ms"`
		Embedding       struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	} `json:"fields"`
}

func (b *VespaBackend) GetItem(ctx context.Context, itemID string) (*vfsv1.ItemFeatures, bool, error) {
	docURL := fmt.Sprintf("%s/document/v1/%s/%s/docid/%s?timeout=200ms", b.baseURL, b.namespace, b.docType, url.PathEscape(itemID))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, docURL, nil)
	if err != nil {
		return nil, false, err
	}

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, false, nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, fmt.Errorf("read get item response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("get item status %d: %s", resp.StatusCode, string(body))
	}

	var res vespaGetResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return nil, false, fmt.Errorf("decode get item: %w", err)
	}

	actualID := res.Fields.ItemID
	if actualID == "" {
		actualID = itemID
	}

	return &vfsv1.ItemFeatures{
		ItemId:          actualID,
		Embedding:       res.Fields.Embedding.Values,
		BrandId:         res.Fields.BrandID,
		Category:        res.Fields.Category,
		PriceEur:        res.Fields.PriceEur,
		SellerCountry:   res.Fields.SellerCountry,
		UpdatedAtUnixMs: res.Fields.UpdatedAtUnixMs,
	}, true, nil
}

func (b *VespaBackend) BatchGetItems(ctx context.Context, itemIDs []string) ([]*vfsv1.ItemFeatures, []string, error) {
	var (
		items   []*vfsv1.ItemFeatures
		missing []string
	)
	for _, id := range itemIDs {
		item, found, err := b.GetItem(ctx, id)
		if err != nil {
			return nil, nil, err
		}
		if found {
			items = append(items, item)
		} else {
			missing = append(missing, id)
		}
	}
	return items, missing, nil
}

type vespaTensorValues struct {
	Values []float32 `json:"values"`
}

type vespaUpsertFields struct {
	ItemID          string            `json:"item_id"`
	Embedding       vespaTensorValues `json:"embedding"`
	BrandID         string            `json:"brand_id"`
	Category        string            `json:"category"`
	PriceEur        float32           `json:"price_eur"`
	SellerCountry   string            `json:"seller_country"`
	UpdatedAtUnixMs int64             `json:"updated_at_unix_ms"`
}

type vespaUpsertRequest struct {
	Fields vespaUpsertFields `json:"fields"`
}

func (b *VespaBackend) upsertSingleItem(ctx context.Context, it *vfsv1.ItemFeatures, nowMs int64) error {
	ts := it.GetUpdatedAtUnixMs()
	if ts <= 0 {
		ts = nowMs
	}

	payload := vespaUpsertRequest{
		Fields: vespaUpsertFields{
			ItemID: it.GetItemId(),
			Embedding: vespaTensorValues{
				Values: it.GetEmbedding(),
			},
			BrandID:         it.GetBrandId(),
			Category:        it.GetCategory(),
			PriceEur:        it.GetPriceEur(),
			SellerCountry:   it.GetSellerCountry(),
			UpdatedAtUnixMs: ts,
		},
	}

	raw, err := json.Marshal(&payload)
	if err != nil {
		return fmt.Errorf("marshal item: %w", err)
	}

	docURL := fmt.Sprintf("%s/document/v1/%s/%s/docid/%s?timeout=200ms", b.baseURL, b.namespace, b.docType, url.PathEscape(it.GetItemId()))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, docURL, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upsert item status %d: %s", resp.StatusCode, string(body))
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	return nil
}

func (b *VespaBackend) UpsertItems(ctx context.Context, items []*vfsv1.ItemFeatures) (uint32, error) {
	if len(items) == 0 {
		return 0, nil
	}

	nowMs := time.Now().UTC().UnixMilli()
	if len(items) == 1 {
		if err := b.upsertSingleItem(ctx, items[0], nowMs); err != nil {
			return 0, err
		}
		return 1, nil
	}

	// Concurrency pool for bulk batch seeding
	concurrency := 16
	sem := make(chan struct{}, concurrency)
	var (
		wg       sync.WaitGroup
		firstMu  sync.Mutex
		firstErr error
		count    uint32
	)

	for _, it := range items {
		wg.Add(1)
		sem <- struct{}{}
		go func(item *vfsv1.ItemFeatures) {
			defer wg.Done()
			defer func() { <-sem }()

			firstMu.Lock()
			hasErr := firstErr != nil
			firstMu.Unlock()
			if hasErr {
				return
			}

			if err := b.upsertSingleItem(ctx, item, nowMs); err != nil {
				firstMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				firstMu.Unlock()
				return
			}
			atomic.AddUint32(&count, 1)
		}(it)
	}

	wg.Wait()
	if firstErr != nil {
		return count, firstErr
	}
	return count, nil
}

type vespaRankingFeatures struct {
	QueryEmb []float32 `json:"query(q_emb)"`
}

type vespaRanking struct {
	Profile  string               `json:"profile"`
	Features vespaRankingFeatures `json:"features"`
}

type vespaPresentation struct {
	Summary string `json:"summary"`
}

type vespaSearchRequest struct {
	YQL          string            `json:"yql"`
	Hits         uint32            `json:"hits"`
	Timeout      string            `json:"timeout"`
	Presentation vespaPresentation `json:"presentation"`
	Ranking      vespaRanking      `json:"ranking"`
}

type vespaSearchResponse struct {
	Root struct {
		Children []struct {
			Relevance float32 `json:"relevance"`
			Fields    struct {
				ItemID          string  `json:"item_id"`
				BrandID         string  `json:"brand_id"`
				Category        string  `json:"category"`
				PriceEur        float32 `json:"price_eur"`
				SellerCountry   string  `json:"seller_country"`
				UpdatedAtUnixMs int64   `json:"updated_at_unix_ms"`
			} `json:"fields"`
		} `json:"children"`
	} `json:"root"`
}

func (b *VespaBackend) SearchSimilar(
	ctx context.Context,
	queryVec []float32,
	topK uint32,
	maxPrice float32,
	countries []string,
) ([]*vfsv1.ScoredItemFeatures, error) {
	if topK == 0 {
		topK = 10
	}

	// Construct YQL where clause
	whereParts := make([]string, 0, 3)
	if maxPrice > 0 {
		whereParts = append(whereParts, fmt.Sprintf("price_eur <= %s", strconv.FormatFloat(float64(maxPrice), 'f', -1, 32)))
	}
	if len(countries) > 0 {
		countryDisjunction := make([]string, len(countries))
		for i, c := range countries {
			countryDisjunction[i] = fmt.Sprintf("seller_country contains %q", c)
		}
		whereParts = append(whereParts, fmt.Sprintf("(%s)", strings.Join(countryDisjunction, " or ")))
	}
	whereParts = append(whereParts, fmt.Sprintf("({targetHits:%d}nearestNeighbor(embedding, q_emb))", topK))

	yql := fmt.Sprintf("select item_id, brand_id, category, price_eur, seller_country, updated_at_unix_ms from %s where %s",
		b.docType, strings.Join(whereParts, " and "))

	searchReq := vespaSearchRequest{
		YQL:     yql,
		Hits:    topK,
		Timeout: "200ms",
		Presentation: vespaPresentation{
			Summary: "metadata_only",
		},
		Ranking: vespaRanking{
			Profile: "similarity",
			Features: vespaRankingFeatures{
				QueryEmb: queryVec,
			},
		},
	}

	raw, err := json.Marshal(&searchReq)
	if err != nil {
		return nil, fmt.Errorf("marshal search req: %w", err)
	}

	searchURL := fmt.Sprintf("%s/search/", b.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := b.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vespa search request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read search response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vespa search status %d: %s", resp.StatusCode, string(body))
	}

	var res vespaSearchResponse
	if err := json.Unmarshal(body, &res); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	hits := make([]*vfsv1.ScoredItemFeatures, len(res.Root.Children))
	for i, child := range res.Root.Children {
		hits[i] = &vfsv1.ScoredItemFeatures{
			Item: &vfsv1.ItemFeatures{
				ItemId:          child.Fields.ItemID,
				BrandId:         child.Fields.BrandID,
				Category:        child.Fields.Category,
				PriceEur:        child.Fields.PriceEur,
				SellerCountry:   child.Fields.SellerCountry,
				UpdatedAtUnixMs: child.Fields.UpdatedAtUnixMs,
			},
			CosineSimilarity: child.Relevance,
		}
	}

	return hits, nil
}
