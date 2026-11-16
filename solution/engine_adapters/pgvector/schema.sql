CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS items (
    item_id        TEXT PRIMARY KEY,
    embedding      vector(768) NOT NULL,
    brand_id       TEXT NOT NULL,
    category       TEXT NOT NULL,
    price_eur      REAL NOT NULL CHECK (price_eur >= 0.0),
    seller_country VARCHAR(8) NOT NULL,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS items_filter_country_price_idx
    ON items (seller_country, price_eur);

CREATE INDEX IF NOT EXISTS items_embedding_hnsw_cosine_idx
    ON items USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
