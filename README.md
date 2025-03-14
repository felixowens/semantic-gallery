# Semantic Gallery

A fast, lightweight semantic media search engine built in Rust with personal entity recognition capabilities. This system allows users to search their image collection using natural language queries and personal tags. This is a personal project for learning purposes.

## Project Overview

This project implements a personal media semantic search system with the following capabilities:

- Semantic image search: Find images by describing their content
- Personal entity recognition: Tag personal objects/people for improved search
- Fast vector search: Optimized for performance with ~10k image collections
- Self-hosted: Complete privacy with local processing
- CLI + API: Flexible interfaces for both scripting and applications

## Tech Stack

- Language: Rust (strongly typed, performant)
- Web Framework: Actix-web
- Database: PostgreSQL + pgvector extension
- Embedding Generation: candle or EmbedAnything
- CLI: clap for argument parsing
- Image Processing: image-rs crate
- Database Access: sqlx for type-safe queries

## Implementation Plan

### Phase 1: Project Setup & Core Infrastructure

- Rust project initialization with Actix-web
- Database setup with migrations
- Basic configuration and environment management
- Media upload and storage functionality
- CLI scaffolding

### Phase 2: Embedding Pipeline

- Integration of candle for ML capabilities
- CLIP model implementation for image embedding
- Embedding storage in PostgreSQL/pgvector
- Batch processing for multiple images

### Phase 3: Search Implementation

- Text-to-embedding conversion for queries
- Vector similarity search with pgvector
- Result ranking and relevance scoring
- Search API endpoint implementation

### Phase 4: Personal Entity Features

- Tag management system
- Media-tag associations
- Tag-aware search implementation

### Phase 5: Performance Optimizations

- HNSW vector indexing for faster queries
- Tiered search strategy (tags → partial tags → vector similarity)
- Query caching
- Parallel processing for batch operations
- Connection pooling
- Hybrid filtering + vector search approach

### Phase 6: CLI Usage

- Ingest images
- Search for images
- Tag an image
- List all tags

### Phase 7: API Implementation

- API endpoints for media ingestion
- API endpoints for search
- API endpoints for tag management

### Phase 8: Performance Optimizations

- HNSW vector indexing for faster queries
- Tiered search strategy (tags → partial tags → vector similarity)
- Query caching
- Parallel processing for batch operations
- Connection pooling
- Hybrid filtering + vector search approach

### Phase 9: Documentation & Testing

- Detailed documentation
- Integration testing
- Performance benchmarks

## CLI Usage

### Ingest images

```shell
media-search ingest /path/to/photos --recursive
```

### Search for images

```shell
media-search search "dog playing in the snow"
```

### Tag an image

```shell
media-search tag 5fd3a8c1-3d3f-4b0e-8d7f-28a48ad8a58b --add "Rover,pet,dog"
```

### List all tags

```shell
media-search list-tags
```

## API Endpoints

### Upload media files

```txt
POST /api/media              # Upload media files
GET /api/media/:id           # Get media details
DELETE /api/media/:id        # Remove media

POST /api/tags               # Create new tag
GET /api/tags                # List all tags
DELETE /api/tags/:id         # Remove tag

POST /api/media/:id/tags     # Associate tags with media
DELETE /api/media/:id/tags/:tag_id  # Remove tag from media

GET /api/search?q=query      # Search media by semantic query
```

## Setup

Detailed setup instructions to be added as development progresses.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## LLM Context Prompt

This is a helpful prompt to help LLMs understand the project during development.

```txt
I'm working on a semantic media search project called "Semantic Media Search" built in Rust. 
The core functionality:
- Self-hosted service for searching personal media collections (10K images)
- Uses CLIP model via candle/EmbedAnything for generating embeddings
- PostgreSQL + pgvector for vector storage and search
- Personal entity tagging (e.g., "Felix" for a specific cat)
- High performance search with HNSW indexing
- CLI + API interfaces with Actix web framework

Key components:
1. Media ingestion pipeline
2. Embedding generation
3. Vector search
4. Tag management
5. Performance optimizations

The project is in [CURRENT PHASE] and I'm working on [SPECIFIC COMPONENT].
```
