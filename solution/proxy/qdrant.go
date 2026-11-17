package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strings"
	"time"

	"connectrpc.com/connect"
	vfsv1 "github.com/fs-benchmark/fs-benchmark-sandbox/gen/vfs/v1"
	"golang.org/x/net/http2"
	"google.golang.org/protobuf/encoding/protowire"
)

type rawBytes []byte

type rawProtoCodec struct{}

func (rawProtoCodec) Name() string { return "proto" }
func (rawProtoCodec) Marshal(v any) ([]byte, error) {
	b, ok := v.(*rawBytes)
	if !ok {
		return nil, fmt.Errorf("expected *rawBytes")
	}
	return append([]byte(nil), (*b)...), nil
}
func (rawProtoCodec) Unmarshal(data []byte, v any) error {
	b, ok := v.(*rawBytes)
	if !ok {
		return fmt.Errorf("expected *rawBytes")
	}
	*b = append((*b)[:0], data...)
	return nil
}

// QdrantBackend implements EngineBackend using gRPC (:6334) for fast path operations
// and HTTP REST (:6333) for declarative startup setup.
type QdrantBackend struct {
	baseURL      string
	grpcURL      string
	collection   string
	restClient   *http.Client
	h2cClient    *http.Client
	healthClient *connect.Client[rawBytes, rawBytes]
	upsertClient *connect.Client[rawBytes, rawBytes]
	getClient    *connect.Client[rawBytes, rawBytes]
	searchClient *connect.Client[rawBytes, rawBytes]
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

// NewQdrantBackend initializes and verifies the Qdrant collection and connects via gRPC.
func NewQdrantBackend(ctx context.Context, baseURL, grpcURL, collection string) (*QdrantBackend, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	if grpcURL == "" {
		if strings.Contains(baseURL, ":6333") {
			grpcURL = strings.Replace(baseURL, ":6333", ":6334", 1)
		} else {
			grpcURL = baseURL
		}
	}
	grpcURL = strings.TrimRight(grpcURL, "/")
	if collection == "" {
		collection = "items"
	}

	h2cClient := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http2.Transport{
			AllowHTTP: true,
			DialTLSContext: func(ctx context.Context, network, addr string, _ *tls.Config) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, network, addr)
			},
		},
	}

	backend := &QdrantBackend{
		baseURL:    baseURL,
		grpcURL:    grpcURL,
		collection: collection,
		restClient: &http.Client{
			Timeout: 5 * time.Second,
			Transport: &http.Transport{
				DisableKeepAlives: true,
			},
		},
		h2cClient: h2cClient,
		healthClient: connect.NewClient[rawBytes, rawBytes](
			h2cClient,
			grpcURL+"/qdrant.Qdrant/HealthCheck",
			connect.WithGRPC(),
			connect.WithCodec(rawProtoCodec{}),
		),
		upsertClient: connect.NewClient[rawBytes, rawBytes](
			h2cClient,
			grpcURL+"/qdrant.Points/Upsert",
			connect.WithGRPC(),
			connect.WithCodec(rawProtoCodec{}),
		),
		getClient: connect.NewClient[rawBytes, rawBytes](
			h2cClient,
			grpcURL+"/qdrant.Points/Get",
			connect.WithGRPC(),
			connect.WithCodec(rawProtoCodec{}),
		),
		searchClient: connect.NewClient[rawBytes, rawBytes](
			h2cClient,
			grpcURL+"/qdrant.Points/Search",
			connect.WithGRPC(),
			connect.WithCodec(rawProtoCodec{}),
		),
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
	b.restClient.CloseIdleConnections()
	b.h2cClient.CloseIdleConnections()
}

func (b *QdrantBackend) Ping(ctx context.Context) error {
	empty := rawBytes{}
	_, err := b.healthClient.CallUnary(ctx, connect.NewRequest(&empty))
	if err != nil {
		// Fallback to readyz on restClient in case gRPC is not available
		req, rErr := http.NewRequestWithContext(ctx, http.MethodGet, b.baseURL+"/readyz", nil)
		if rErr != nil {
			return err
		}
		resp, rErr := b.restClient.Do(req)
		if rErr != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			return nil
		}
		return err
	}
	return nil
}

// EnsureCollection verifies collection existence or creates it via REST with HNSW cosine config.
func (b *QdrantBackend) EnsureCollection(ctx context.Context) error {
	collURL := fmt.Sprintf("%s/collections/%s", b.baseURL, b.collection)

	checkReq, err := http.NewRequestWithContext(ctx, http.MethodGet, collURL, nil)
	if err != nil {
		return err
	}
	resp, err := b.restClient.Do(checkReq)
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
				"indexing_threshold": 20000,
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
		createResp, err := b.restClient.Do(createReq)
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
		idxResp, err := b.restClient.Do(idxReq)
		if err != nil {
			return err
		}
		idxResp.Body.Close()
	}

	return nil
}

// Protowire encoding helpers for gRPC wire format
func appendStringField(b []byte, num protowire.Number, s string) []byte {
	b = protowire.AppendTag(b, num, protowire.BytesType)
	return protowire.AppendString(b, s)
}

func appendBytesField(b []byte, num protowire.Number, v []byte) []byte {
	b = protowire.AppendTag(b, num, protowire.BytesType)
	return protowire.AppendBytes(b, v)
}

func appendFloatSlice(b []byte, fieldNum protowire.Number, vec []float32) []byte {
	b = protowire.AppendTag(b, fieldNum, protowire.BytesType)
	b = protowire.AppendVarint(b, uint64(len(vec)*4))
	for _, f := range vec {
		b = protowire.AppendFixed32(b, math.Float32bits(f))
	}
	return b
}

func appendPayloadEntry(b []byte, fieldNum protowire.Number, key, val string) []byte {
	var v []byte
	v = appendStringField(v, 4, val) // Value: 4 = string_value
	var entry []byte
	entry = appendStringField(entry, 1, key)
	entry = appendBytesField(entry, 2, v)
	return appendBytesField(b, fieldNum, entry)
}

func appendPayloadDouble(b []byte, fieldNum protowire.Number, key string, val float64) []byte {
	var v []byte
	v = protowire.AppendTag(v, 2, protowire.Fixed64Type) // Value: 2 = double_value
	v = protowire.AppendFixed64(v, math.Float64bits(val))
	var entry []byte
	entry = appendStringField(entry, 1, key)
	entry = appendBytesField(entry, 2, v)
	return appendBytesField(b, fieldNum, entry)
}

func appendPayloadInt(b []byte, fieldNum protowire.Number, key string, val int64) []byte {
	var v []byte
	v = protowire.AppendTag(v, 3, protowire.VarintType) // Value: 3 = integer_value
	v = protowire.AppendVarint(v, uint64(val))
	var entry []byte
	entry = appendStringField(entry, 1, key)
	entry = appendBytesField(entry, 2, v)
	return appendBytesField(b, fieldNum, entry)
}

func appendMapStrVal(b []byte, key, val string) []byte {
	return appendPayloadEntry(b, 3, key, val)
}

func appendMapDoubleVal(b []byte, key string, val float64) []byte {
	return appendPayloadDouble(b, 3, key, val)
}

func appendMapIntVal(b []byte, key string, val int64) []byte {
	return appendPayloadInt(b, 3, key, val)
}

func (b *QdrantBackend) UpsertItems(ctx context.Context, items []*vfsv1.ItemFeatures) (uint32, error) {
	if len(items) == 0 {
		return 0, nil
	}

	var req []byte
	req = appendStringField(req, 1, b.collection)
	req = protowire.AppendTag(req, 2, protowire.VarintType)
	waitVal := uint64(0)
	if len(items) > 1 {
		waitVal = 1 // Synchronous wait for bulk catalog seeding
	}
	req = protowire.AppendVarint(req, waitVal)

	nowMs := time.Now().UTC().UnixMilli()
	for _, it := range items {
		ts := it.GetUpdatedAtUnixMs()
		if ts <= 0 {
			ts = nowMs
		}
		var pid []byte
		pid = appendStringField(pid, 2, itemIDToUUID(it.GetItemId()))

		var vecMsg []byte
		vecMsg = appendFloatSlice(vecMsg, 1, it.GetEmbedding())
		var vecsMsg []byte
		vecsMsg = appendBytesField(vecsMsg, 1, vecMsg)

		var pt []byte
		pt = appendBytesField(pt, 1, pid)
		pt = appendMapStrVal(pt, "item_id", it.GetItemId())
		pt = appendMapStrVal(pt, "brand_id", it.GetBrandId())
		pt = appendMapStrVal(pt, "category", it.GetCategory())
		pt = appendMapDoubleVal(pt, "price_eur", float64(it.GetPriceEur()))
		pt = appendMapStrVal(pt, "seller_country", it.GetSellerCountry())
		pt = appendMapIntVal(pt, "updated_at_unix_ms", ts)
		pt = appendBytesField(pt, 4, vecsMsg)

		req = appendBytesField(req, 3, pt) // UpsertPoints: 3 = repeated PointStruct points
	}

	rawReq := rawBytes(req)
	_, err := b.upsertClient.CallUnary(ctx, connect.NewRequest(&rawReq))
	if err != nil {
		return 0, fmt.Errorf("qdrant grpc upsert: %w", err)
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

func parseQdrantValue(b []byte) (strVal string, dblVal float64, intVal int64) {
	for len(b) > 0 {
		num, wtype, n := protowire.ConsumeTag(b)
		if n < 0 {
			return
		}
		b = b[n:]
		switch num {
		case 2:
			if wtype == protowire.Fixed64Type {
				v, m := protowire.ConsumeFixed64(b)
				if m > 0 {
					dblVal = math.Float64frombits(v)
					b = b[m:]
					continue
				}
			}
		case 3:
			if wtype == protowire.VarintType {
				v, m := protowire.ConsumeVarint(b)
				if m > 0 {
					intVal = int64(v)
					b = b[m:]
					continue
				}
			}
		case 4:
			if wtype == protowire.BytesType {
				v, m := protowire.ConsumeString(b)
				if m > 0 {
					strVal = v
					b = b[m:]
					continue
				}
			}
		}
		m := protowire.ConsumeFieldValue(num, wtype, b)
		if m < 0 {
			return
		}
		b = b[m:]
	}
	return
}

func parsePayloadEntry(b []byte, item *vfsv1.ItemFeatures) {
	var key string
	var valBytes []byte
	for len(b) > 0 {
		num, wtype, n := protowire.ConsumeTag(b)
		if n < 0 {
			return
		}
		b = b[n:]
		if num == 1 && wtype == protowire.BytesType {
			s, m := protowire.ConsumeString(b)
			if m > 0 {
				key = s
				b = b[m:]
				continue
			}
		} else if num == 2 && wtype == protowire.BytesType {
			v, m := protowire.ConsumeBytes(b)
			if m > 0 {
				valBytes = v
				b = b[m:]
				continue
			}
		}
		m := protowire.ConsumeFieldValue(num, wtype, b)
		if m < 0 {
			return
		}
		b = b[m:]
	}
	strVal, dblVal, intVal := parseQdrantValue(valBytes)
	switch key {
	case "item_id":
		item.ItemId = strVal
	case "brand_id":
		item.BrandId = strVal
	case "category":
		item.Category = strVal
	case "price_eur":
		item.PriceEur = float32(dblVal)
	case "seller_country":
		item.SellerCountry = strVal
	case "updated_at_unix_ms":
		item.UpdatedAtUnixMs = intVal
	}
}

func parseVectorsOutput(b []byte) []float32 {
	for len(b) > 0 {
		num, wtype, n := protowire.ConsumeTag(b)
		if n < 0 {
			return nil
		}
		b = b[n:]
		if num == 1 && wtype == protowire.BytesType {
			vo, m := protowire.ConsumeBytes(b)
			if m < 0 {
				return nil
			}
			b = b[m:]
			var out []float32
			for len(vo) > 0 {
				vnum, vwtype, vn := protowire.ConsumeTag(vo)
				if vn < 0 {
					break
				}
				vo = vo[vn:]
				if vnum == 1 && vwtype == protowire.BytesType {
					packed, vm := protowire.ConsumeBytes(vo)
					if vm < 0 {
						break
					}
					vo = vo[vm:]
					if out == nil {
						out = make([]float32, 0, len(packed)/4)
					}
					for len(packed) >= 4 {
						bits, fm := protowire.ConsumeFixed32(packed)
						if fm < 0 {
							break
						}
						out = append(out, math.Float32frombits(bits))
						packed = packed[fm:]
					}
					continue
				} else if vnum == 1 && vwtype == protowire.Fixed32Type {
					bits, vm := protowire.ConsumeFixed32(vo)
					if vm < 0 {
						break
					}
					out = append(out, math.Float32frombits(bits))
					vo = vo[vm:]
					continue
				}
				vm := protowire.ConsumeFieldValue(vnum, vwtype, vo)
				if vm < 0 {
					break
				}
				vo = vo[vm:]
			}
			return out
		}
		m := protowire.ConsumeFieldValue(num, wtype, b)
		if m < 0 {
			return nil
		}
		b = b[m:]
	}
	return nil
}

func parseRetrievedPoint(b []byte) *vfsv1.ItemFeatures {
	it := &vfsv1.ItemFeatures{}
	for len(b) > 0 {
		num, wtype, n := protowire.ConsumeTag(b)
		if n < 0 {
			break
		}
		b = b[n:]
		switch num {
		case 1: // PointId
			if wtype == protowire.BytesType {
				pidBytes, m := protowire.ConsumeBytes(b)
				if m > 0 {
					for len(pidBytes) > 0 {
						pnum, pwtype, pn := protowire.ConsumeTag(pidBytes)
						if pn < 0 {
							break
						}
						pidBytes = pidBytes[pn:]
						if pnum == 2 && pwtype == protowire.BytesType {
							uuidStr, um := protowire.ConsumeString(pidBytes)
							if um > 0 && it.ItemId == "" {
								it.ItemId = uuidStr
							}
						}
						pm := protowire.ConsumeFieldValue(pnum, pwtype, pidBytes)
						if pm < 0 {
							break
						}
						pidBytes = pidBytes[pm:]
					}
					b = b[m:]
					continue
				}
			}
		case 2: // map<string, Value> payload
			if wtype == protowire.BytesType {
				entry, m := protowire.ConsumeBytes(b)
				if m > 0 {
					parsePayloadEntry(entry, it)
					b = b[m:]
					continue
				}
			}
		case 4: // VectorsOutput vectors
			if wtype == protowire.BytesType {
				vo, m := protowire.ConsumeBytes(b)
				if m > 0 {
					it.Embedding = parseVectorsOutput(vo)
					b = b[m:]
					continue
				}
			}
		}
		m := protowire.ConsumeFieldValue(num, wtype, b)
		if m < 0 {
			break
		}
		b = b[m:]
	}
	return it
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

	var req []byte
	req = appendStringField(req, 1, b.collection)
	for _, u := range uuids {
		var pid []byte
		pid = appendStringField(pid, 2, u)
		req = appendBytesField(req, 2, pid)
	}
	req = appendBytesField(req, 4, []byte{0x08, 0x01}) // WithPayloadSelector
	req = appendBytesField(req, 5, []byte{0x08, 0x01}) // WithVectorsSelector

	rawReq := rawBytes(req)
	resp, err := b.getClient.CallUnary(ctx, connect.NewRequest(&rawReq))
	if err != nil {
		return nil, nil, fmt.Errorf("qdrant grpc get points: %w", err)
	}

	foundMap := make(map[string]*vfsv1.ItemFeatures)
	raw := *resp.Msg
	for len(raw) > 0 {
		num, wtype, n := protowire.ConsumeTag(raw)
		if n < 0 {
			break
		}
		raw = raw[n:]
		if num == 1 && wtype == protowire.BytesType {
			ptBytes, m := protowire.ConsumeBytes(raw)
			if m > 0 {
				it := parseRetrievedPoint(ptBytes)
				if it.ItemId != "" {
					foundMap[it.ItemId] = it
				}
				raw = raw[m:]
				continue
			}
		}
		m := protowire.ConsumeFieldValue(num, wtype, raw)
		if m < 0 {
			break
		}
		raw = raw[m:]
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

func parseScoredPoint(b []byte) *vfsv1.ScoredItemFeatures {
	hit := &vfsv1.ScoredItemFeatures{Item: &vfsv1.ItemFeatures{}}
	for len(b) > 0 {
		num, wtype, n := protowire.ConsumeTag(b)
		if n < 0 {
			break
		}
		b = b[n:]
		switch num {
		case 2: // map<string, Value> payload
			if wtype == protowire.BytesType {
				entry, m := protowire.ConsumeBytes(b)
				if m > 0 {
					parsePayloadEntry(entry, hit.Item)
					b = b[m:]
					continue
				}
			}
		case 3: // float score
			if wtype == protowire.Fixed32Type {
				bits, m := protowire.ConsumeFixed32(b)
				if m > 0 {
					hit.CosineSimilarity = math.Float32frombits(bits)
					b = b[m:]
					continue
				}
			}
		}
		m := protowire.ConsumeFieldValue(num, wtype, b)
		if m < 0 {
			break
		}
		b = b[m:]
	}
	return hit
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

	var req []byte
	req = appendStringField(req, 1, b.collection)
	req = appendFloatSlice(req, 2, queryVec)

	var mustConds [][]byte
	if maxPrice > 0 {
		var rng []byte
		rng = protowire.AppendTag(rng, 4, protowire.Fixed64Type) // lte
		rng = protowire.AppendFixed64(rng, math.Float64bits(float64(maxPrice)))
		var fc []byte
		fc = appendStringField(fc, 1, "price_eur")
		fc = appendBytesField(fc, 3, rng)
		var cond []byte
		cond = appendBytesField(cond, 1, fc)
		mustConds = append(mustConds, cond)
	}
	if len(countries) > 0 {
		var rs []byte
		for _, c := range countries {
			rs = appendStringField(rs, 1, c)
		}
		var m []byte
		m = appendBytesField(m, 5, rs) // RepeatedStrings keywords
		var fc []byte
		fc = appendStringField(fc, 1, "seller_country")
		fc = appendBytesField(fc, 2, m)
		var cond []byte
		cond = appendBytesField(cond, 1, fc)
		mustConds = append(mustConds, cond)
	}
	if len(mustConds) > 0 {
		var filter []byte
		for _, c := range mustConds {
			filter = appendBytesField(filter, 2, c) // Filter: 2 = repeated Condition must
		}
		req = appendBytesField(req, 3, filter)
	}

	req = protowire.AppendTag(req, 4, protowire.VarintType)
	req = protowire.AppendVarint(req, uint64(topK))
	req = appendBytesField(req, 6, []byte{0x08, 0x01}) // WithPayloadSelector: true (with_vector is omitted)

	rawReq := rawBytes(req)
	resp, err := b.searchClient.CallUnary(ctx, connect.NewRequest(&rawReq))
	if err != nil {
		return nil, fmt.Errorf("qdrant grpc search: %w", err)
	}

	var hits []*vfsv1.ScoredItemFeatures
	raw := *resp.Msg
	for len(raw) > 0 {
		num, wtype, n := protowire.ConsumeTag(raw)
		if n < 0 {
			break
		}
		raw = raw[n:]
		if num == 1 && wtype == protowire.BytesType {
			ptBytes, m := protowire.ConsumeBytes(raw)
			if m > 0 {
				hits = append(hits, parseScoredPoint(ptBytes))
				raw = raw[m:]
				continue
			}
		}
		m := protowire.ConsumeFieldValue(num, wtype, raw)
		if m < 0 {
			break
		}
		raw = raw[m:]
	}

	return hits, nil
}
