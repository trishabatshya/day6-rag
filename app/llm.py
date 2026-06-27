import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def stream_answer(query: str, context_chunks: list[str]):
    """
    Given a query and retrieved context chunks, stream an answer from the LLM.
    Yields text deltas as they arrive.
    """
    context = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""You are a helpful assistant. Answer the question using only the provided context.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=512,
        temperature=0.3,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            yield delta