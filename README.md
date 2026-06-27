# Day 6 & 7 : Hybrid RAG Pipeline with LLM Streaming

FastAPI service implementing a Retrieval-Augmented Generation (RAG) pipeline using hybrid retrieval (FAISS + BM25) and streaming responses from a Groq-hosted Llama 3.1 model.

## How it works

1. Documents are embedded using `all-MiniLM-L6-v2` (384-dimensional vectors)
2. Embeddings are stored in a FAISS `IndexFlatIP` for dense semantic retrieval
3. Documents are also indexed using BM25 for sparse keyword retrieval
4. Dense and BM25 scores are min-max normalised to `[0,1]` and averaged with equal weight
5. Top-5 retrieved chunks are used as context for the LLM
6. The prompt is sent to Groq (`llama-3.1-8b-instant`)
7. The generated response is streamed token-by-token using Server-Sent Events (SSE)

## Endpoints

- **POST /retrieve** : returns the top-k retrieved chunks with dense, BM25, combined scores, and retrieval latency
- **POST /ask** : retrieves the top-5 chunks and streams an LLM-generated answer grounded in the retrieved context
- **GET /health** : health check

## Running locally

```bash
uv venv && source .venv/Scripts/activate
uv sync
uvicorn app.main:app --reload --port 8001
