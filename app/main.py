import time
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retriever import hybrid_search
from app.llm import stream_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievalResult(BaseModel):
    document: str
    combined_score: float
    dense_score: float
    bm25_score: float


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    latency_ms: float


@app.post("/retrieve", response_model=QueryResponse)
def retrieve(request: QueryRequest):
    start = time.time()
    results = hybrid_search(request.query, request.top_k)
    latency_ms = (time.time() - start) * 1000
    return QueryResponse(
        query=request.query,
        results=results,
        latency_ms=latency_ms,
    )


@app.post("/ask")
def ask(request: QueryRequest):
    """
    Retrieve relevant chunks then stream an LLM answer.
    Returns a text/event-stream (Server-Sent Events).
    """
    results = hybrid_search(request.query, request.top_k)
    context_chunks = [r["document"] for r in results]

    def event_stream():
        # First, send the sources so the frontend can display them
        sources_payload = json.dumps({"sources": context_chunks})
        yield f"data: {sources_payload}\n\n"

        # Then stream the answer tokens
        for token in stream_answer(request.query, context_chunks):
            token_payload = json.dumps({"token": token})
            yield f"data: {token_payload}\n\n"

        # Signal completion
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}