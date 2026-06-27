# Day 6 : Hybrid RAG Retriever

FastAPI service implementing hybrid document retrieval combining dense vector search (FAISS) and sparse keyword search (BM25).

## How it works

1. Documents are embedded using `all-MiniLM-L6-v2` (384-dimensional vectors)
2. Embeddings are stored in a FAISS flat index for cosine similarity search
3. Documents are also indexed with BM25 for keyword matching
4. At query time, both scores are normalised to [0,1] and averaged
5. Top-k results returned with individual scores for transparency

## Endpoints

- POST /retrieve : takes a query string, returns top-k chunks with scores and latency
- GET /health : health check

## Running locally

```bash
uv venv && source .venv/Scripts/activate
uv sync
uvicorn app.main:app --reload
```

## Example

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"how do neural networks learn\", \"top_k\": 5}"
```