"""
embedder.py

Stage 2 of the RAG pipeline: converts text chunks into embedding vectors
using a local Sentence Transformers model (all-MiniLM-L6-v2).

This model runs entirely on your own machine (CPU is fine) - no API
calls, no cost, no rate limits. It outputs a 384-dimensional vector
for any input text.
"""

from sentence_transformers import SentenceTransformer

# Loading the model is relatively slow (a few seconds), so we load it
# once at module level rather than inside a function that might be
# called repeatedly.
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazily load and cache the embedding model, so it's only loaded once
    per program run even if this function is called multiple times.
    """
    global _model
    if _model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)... this may take a moment on first run.")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings into a list of vectors.

    Args:
        texts: list of strings (e.g. chunk texts) to embed.

    Returns:
        A list of vectors (one per input text), each a list of 384 floats.
    """
    model = get_model()
    # convert_to_numpy=False keeps output as plain Python lists, which are
    # what Chroma expects when we store them in the next step.
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def embed_single_text(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string (e.g. a user query)."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Quick manual test: embed a couple of real chunks from your documents
    # and actually inspect the resulting vector, so it stops being abstract.
    from document_loader import load_documents
    from chunker import chunk_documents

    docs = load_documents("data")
    all_chunks = chunk_documents(docs, chunk_size=800, overlap_sentences=1)

    if not all_chunks:
        print("No chunks found - make sure data/ has documents and chunker.py runs cleanly first.")
        exit()

    # Embed just the first 3 chunks for inspection (embedding everything
    # happens for real in the next step, when we store into Chroma).
    sample_chunks = all_chunks[:3]
    sample_texts = [c["text"] for c in sample_chunks]

    vectors = embed_texts(sample_texts)

    print(f"\nEmbedded {len(vectors)} chunk(s).\n")

    for chunk, vector in zip(sample_chunks, vectors):
        print(f"--- {chunk['source']} | chunk {chunk['chunk_index']} ---")
        print(f"Text preview: {chunk['text'][:100]}...")
        print(f"Vector length: {len(vector)} dimensions")
        print(f"First 8 values: {vector[:8]}")
        print()

    # Bonus sanity check: embed two texts we KNOW should be similar in
    # meaning despite sharing no words, and two that should be unrelated,
    # then print their similarity scores so cosine similarity stops being
    # theoretical and becomes something you've actually seen work.
    import numpy as np

    def cosine_similarity(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    similar_pair = embed_texts([
        "The cat sat on the mat.",
        "A feline rested on the rug."
    ])
    unrelated_pair = embed_texts([
        "The cat sat on the mat.",
        "Stock markets fell sharply today."
    ])

    print("--- Sanity check: does semantic similarity actually work? ---")
    print(f"Similarity (related meaning, different words): "
          f"{cosine_similarity(similar_pair[0], similar_pair[1]):.4f}")
    print(f"Similarity (unrelated meaning): "
          f"{cosine_similarity(unrelated_pair[0], unrelated_pair[1]):.4f}")
