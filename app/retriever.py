from dotenv import load_dotenv
load_dotenv()
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from app.documents import DOCUMENTS

# Load the embedding model once at startup — this takes a few seconds
# all-MiniLM-L6-v2 produces 384-dimensional embeddings
# It is small, fast, and good enough for this use case
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Build the dense index (FAISS) ---

# Embed all documents — returns a numpy array of shape (50, 384)
print("Embedding documents...")
doc_embeddings = model.encode(DOCUMENTS, convert_to_numpy=True)

# Normalise embeddings so that dot product == cosine similarity
# This lets us use the faster IndexFlatIP (inner product) instead of L2
faiss.normalize_L2(doc_embeddings)

# Build the index
dimension = doc_embeddings.shape[1]  # 384
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(doc_embeddings)

print(f"FAISS index built with {faiss_index.ntotal} vectors")

# --- Build the sparse index (BM25) ---

# BM25 works on tokenised text — split each document into words
tokenised_docs = [doc.lower().split() for doc in DOCUMENTS]
bm25_index = BM25Okapi(tokenised_docs)

print("BM25 index built")


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve top_k documents using hybrid search.
    Combines normalised BM25 scores with normalised FAISS scores.
    """

    # --- Dense retrieval ---
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search FAISS — returns distances and indices for top_k results
    # With normalised vectors and IndexFlatIP, distances are cosine similarities
    dense_scores, dense_indices = faiss_index.search(query_embedding, len(DOCUMENTS))
    dense_scores = dense_scores[0]   # shape: (50,)
    dense_indices = dense_indices[0] # shape: (50,)

    # Build a score array indexed by document position
    dense_score_map = np.zeros(len(DOCUMENTS))
    for idx, score in zip(dense_indices, dense_scores):
        dense_score_map[idx] = score

    # --- Sparse retrieval (BM25) ---
    tokenised_query = query.lower().split()
    bm25_scores = np.array(bm25_index.get_scores(tokenised_query))

    # --- Normalise both score arrays to [0, 1] ---
    def normalise(arr):
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    dense_norm = normalise(dense_score_map)
    bm25_norm = normalise(bm25_scores)

    # --- Combine: equal weight to both ---
    combined = 0.5 * dense_norm + 0.5 * bm25_norm

    # --- Get top_k indices sorted by combined score ---
    top_indices = np.argsort(combined)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "document": DOCUMENTS[idx],
            "combined_score": float(combined[idx]),
            "dense_score": float(dense_norm[idx]),
            "bm25_score": float(bm25_norm[idx]),
        })

    return results