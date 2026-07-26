"""
chunker.py

Stage 1b of the RAG pipeline: splits raw document text into small,
coherent chunks suitable for embedding.

Strategy: sentence-aware chunking with overlap.
  - Text is split into sentences first, so we never cut a sentence in half.
  - Sentences are greedily grouped into chunks up to a target character size.
  - Each new chunk starts by re-including the last sentence(s) from the
    previous chunk, so ideas that span a chunk boundary aren't lost.
"""

import re


def split_into_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: splits on '.', '!', or '?' followed by
    whitespace. Not linguistically perfect (e.g. "Dr. Smith" would
    incorrectly split), but good enough for typical prose and avoids
    adding an NLP library dependency for this project.
    """
    # Collapse newlines/extra whitespace first so splitting is clean
    text = re.sub(r"\s+", " ", text).strip()

    # Split, keeping the punctuation attached to the sentence before it
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Drop any empty strings that can result from edge cases
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Split a single document's text into overlapping, sentence-aware chunks.

    Args:
        text: the raw document text (already extracted from PDF/txt).
        source: filename this text came from, tagged onto every chunk.
        chunk_size: target maximum characters per chunk.
        overlap_sentences: how many trailing sentences from the previous
            chunk to repeat at the start of the next chunk.

    Returns:
        A list of dicts, one per chunk:
            {
                "text": "the chunk's actual text",
                "source": "lecture3.pdf",
                "chunk_index": 0   # position of this chunk within the document
            }
    """
    sentences = split_into_sentences(text)
    chunks = []

    current_sentences: list[str] = []
    current_length = 0

    def flush_chunk():
        """Turn the currently accumulated sentences into a chunk dict."""
        chunk_text_value = " ".join(current_sentences).strip()
        if chunk_text_value:
            chunks.append({
                "text": chunk_text_value,
                "source": source,
                "chunk_index": len(chunks),
            })

    for sentence in sentences:
        sentence_length = len(sentence) + 1  # +1 for the joining space

        # If adding this sentence would exceed our target size, and we
        # already have at least one sentence in the current chunk, close
        # off the current chunk before adding the new sentence.
        if current_sentences and current_length + sentence_length > chunk_size:
            flush_chunk()

            # Start the next chunk with overlap: carry over the last
            # `overlap_sentences` sentences from the chunk we just closed.
            overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_sentences = list(overlap)
            current_length = sum(len(s) + 1 for s in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_length

    # Flush whatever's left after the loop ends
    flush_chunk()

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    overlap_sentences: int = 1,
) -> list[dict]:
    """
    Chunk a list of documents (as returned by document_loader.load_documents).

    Returns a flat list of chunk dicts across all documents, each tagged
    with its source filename and its chunk_index within that document.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(
            doc["text"],
            source=doc["source"],
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks


if __name__ == "__main__":
    # Quick manual test: load real documents and inspect the chunks
    # produced, so you can see exactly what gets fed into embeddings next.
    from document_loader import load_documents

    docs = load_documents("data")
    all_chunks = chunk_documents(docs, chunk_size=800, overlap_sentences=1)

    print(f"\nProduced {len(all_chunks)} chunk(s) from {len(docs)} document(s).\n")

    for chunk in all_chunks:
        print(f"--- {chunk['source']} | chunk {chunk['chunk_index']} "
              f"({len(chunk['text'])} chars) ---")
        print(chunk["text"])
        print()
