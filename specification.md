# Fashion Product Search & Display System — Specification

## Overview

A system that allows users to describe what they're looking for in natural language, retrieves matching fashion products from a database, and generates a visually appealing web page to display the results. The presentation adapts based on the type of query and the items retrieved.

## Dataset

- **Source**: `ashraq/fashion-product-images-small` on Hugging Face
- **Size**: ~44,000 fashion products
- **Per product**: image + metadata (id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName)

## Architecture

```
User query (natural language)
         │
         ▼
┌─────────────────────────┐
│  1. Query Understanding │  LLM extracts structured filters
│     (LLM)               │  from the user's natural language
└────────┬────────────────┘
         │
         ▼  (44K → small subset)
┌─────────────────────────┐
│  2. Metadata Filtering  │  Pandas filters on extracted
│     (Pandas/Parquet)    │  field values (color, category…)
└────────┬────────────────┘
         │
         ▼  (CLIP on reduced set)
┌─────────────────────────┐
│  3. CLIP Ranking        │  Cosine similarity between query
│     (NumPy)             │  text embedding and product image
│                         │  embeddings for fuzzy/visual match
└────────┬────────────────┘
         │
         ▼  Top K results
┌─────────────────────────┐
│  4. Page Generation     │  LLM generates an HTML/CSS page
│     (LLM)               │  adapted to the query type and
│                         │  retrieved items
└─────────────────────────┘
```

## Components

### 1. Query Understanding

The LLM receives the user's natural language query along with a reference list of all unique metadata values per field (e.g. all known colors, all article types). It outputs:

- **Structured filters**: mapped to exact metadata field values (e.g. "denim-coloured" → `baseColour: Blue`)
- **Residual query**: anything that can't be mapped to a field, passed to CLIP for visual/semantic matching (e.g. "floral", "beach party vibes")

The metadata vocabulary is small enough (10-20 colors, ~50 article types, a handful of seasons/usages) to include in full in the LLM prompt. This solves the synonym problem at the filter stage — the LLM bridges the gap between user language and catalog vocabulary.

### 2. Metadata Filtering

Simple pandas DataFrame filtering on the structured fields extracted in step 1. This narrows the search space from ~44K to a manageable subset before the more expensive CLIP similarity step.

### 3. CLIP Retrieval & Ranking

- **Offline**: Pre-compute CLIP image embeddings for all 44K products. Store as a NumPy array (44K × 512) in an `.npz` file, with a parallel array of product IDs.
- **Online**: Encode the user's text query (or the residual query from step 1) with CLIP's text encoder. Compute cosine similarity against the filtered subset's embeddings. Return top K results.

**Why CLIP**: Handles the fuzzy matching that metadata filters can't — visual concepts ("floral", "streetwear"), subjective descriptions ("beach party vibes"), and cases where metadata is incomplete or inaccurate. Operates in a shared image-text embedding space so text queries match against what items actually look like.

**Complexity**: O(N × D) for brute-force cosine similarity where N = filtered subset size and D = 512. At 44K items this is sub-millisecond. For production scale, would move to approximate nearest neighbor (HNSW via pgvector).

### 4. Generative Page Construction

The LLM receives the top K products (metadata + image paths) and the original user query, then generates an HTML/CSS page. The layout adapts to context:

- **Catalog query** ("blue summer dresses") → grid layout
- **Outfit/styling query** ("outfit for a beach wedding") → lookbook / mood board
- **Try-on query** ("show me this jacket on someone") → model/mannequin display

## Storage

### Current (development)

| Data | Format | Notes |
|---|---|---|
| Product metadata | `.parquet` | Columnar, typed, compressed. Loaded into pandas. |
| CLIP embeddings | `.npz` | NumPy array (44K × 512) + product ID index. ~100MB in memory. |
| Product images | `data/images/{id}.jpg` | Raw image files on disk. |

All stored under `data/` (gitignored).

### Production

PostgreSQL with pgvector extension. Single table with metadata columns + a `vector(512)` column for embeddings. Enables combined metadata filtering and approximate nearest neighbor search in one SQL query:

```sql
SELECT *, embedding <=> $query_vector AS distance
FROM products
WHERE gender = 'Women' AND season = 'Summer'
ORDER BY distance
LIMIT 20;
```

## Dependencies

- **datasets**, **Pillow** — downloading and processing the HF dataset
- **transformers** / **open_clip** — CLIP model for image and text encoding
- **numpy**, **pandas**, **pyarrow** — data storage and manipulation
- **anthropic** / LLM API — query understanding and page generation
- **fastapi** / **uvicorn** — API server (TBD)

## Scripts

| Script | Purpose |
|---|---|
| `download_dataset.py` | Downloads dataset from HF, saves images + JSONL metadata |
| `build_index.py` | Pre-computes CLIP embeddings, builds parquet metadata file, extracts unique field values |
| `search.py` | Core retrieval pipeline: query understanding → filter → CLIP rank |
| `serve.py` | Web server exposing the search + page generation pipeline |
