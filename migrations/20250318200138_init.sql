-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Media table to store basic media information
CREATE TABLE media (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Embeddings table to store vector representations
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    media_id UUID REFERENCES media(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    embedding vector(512), -- TODO: Adjust dimension based on model
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Tags table for user-defined tags
CREATE TABLE tags (
    id UUID PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many relationship between media and tags
CREATE TABLE media_tags (
    media_id UUID REFERENCES media(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (media_id, tag_id)
);

-- Vector search optimization using HNSW index for better performance
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m=16, ef_construction=64);
