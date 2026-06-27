import time
from fastapi import FastAPI
from pydantic import BaseModel
from app.retriever import hybrid_search

app = FastAPI()


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


@app.get("/health")
def health():
    return {"status": "ok"}