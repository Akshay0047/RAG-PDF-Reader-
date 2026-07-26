"""
vector_store.py

Stage 3 of the RAG pipeline: stores embedded chunks in a persistent
Chroma vector database, and provides the function used later for
Stage 4 (retrieval).

Run this file directly to build (or rebuild) the index from everything
in data/. This is the "ingestion" phase - done once per document set,
not on every question asked.
"""

import chromadb
from embedder import embed_texts

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"


def get_collection():
    """
    Connect to (or create) the persistent Chroma collection.

    Using a PersistentClient means the collection is saved to disk in
    PERSIST_DIR, so it survives between separate runs of your program -
    you don't need to re-embed everything every time you ask a question.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # explicitly use cosine similarity for search
    )
    return collection


def build_index(chunks: list[dict]):
    """
    Embed a list of chunks and store them in the Chroma collection.

    Args:
        chunks: list of dicts as produced by chunker.chunk_documents(),
            each with "text", "source", and "chunk_index" keys.

    This clears any existing collection first, so re-running this
    function rebuilds the index cleanly rather than appending duplicates.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Delete any existing collection so re-running this script gives a
    # clean rebuild instead of duplicate/stale entries.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet - nothing to delete

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{chunk['source']}_chunk_{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
        for chunk in chunks
    ]

    print(f"Embedding {len(texts)} chunk(s)...")
    embeddings = embed_texts(texts)

    print("Storing in Chroma...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Index built: {collection.count()} chunk(s) stored in '{PERSIST_DIR}'.")


def query_index(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Stage 4 (retrieval): embed a query string and find the top_k most
    similar chunks stored in the collection.

    Returns a list of dicts, each:
        {
            "text": "...",
            "source": "os_process_scheduling.pdf",
            "chunk_index": 2,
            "similarity": 0.83   # higher = more similar (cosine similarity)
        }
    """
    collection = get_collection()
    query_vector = embed_texts([query_text])[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    retrieved = []
    # Chroma returns parallel lists (one entry per query - we only sent one).
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine DISTANCE, not similarity

    for doc_text, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc_text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            # Chroma returns cosine DISTANCE (0 = identical, 2 = opposite).
            # We convert to similarity (1 = identical) to match the
            # cosine similarity concept we've been discussing.
            "similarity": 1 - distance,
        })

    return retrieved


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Test mode: python vector_store.py query "your question here"
        if len(sys.argv) < 3:
            print('Usage: python vector_store.py query "your question"')
            exit()
        question = sys.argv[2]
        results = query_index(question, top_k=3)

        print(f"\nTop {len(results)} result(s) for: \"{question}\"\n")
        for r in results:
            print(f"--- {r['source']} | chunk {r['chunk_index']} | similarity: {r['similarity']:.4f} ---")
            print(r["text"])
            print()
    else:
        # Default mode: build the index from everything in data/
        from document_loader import load_documents
        from chunker import chunk_documents

        docs = load_documents("data")
        chunks = chunk_documents(docs, chunk_size=800, overlap_sentences=1)
        build_index(chunks)
