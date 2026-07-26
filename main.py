"""
main.py

Entry point for the RAG app. Wires together every stage built so far:

  build command (Stages 1-3, run occasionally when documents change):
      load documents -> chunk -> embed -> store in Chroma

  chat command (Stages 4-5, run repeatedly for each question):
      retrieve relevant chunks -> build prompt -> generate answer

Usage:
    python main.py build
    python main.py chat
"""

import sys

from document_loader import load_documents
from chunker import chunk_documents
from vector_store import build_index, query_index, get_collection
from prompt_builder import build_messages
from generator import generate_answer


def run_build():
    """Stages 1-3: (re)build the vector index from everything in data/."""
    print("Loading documents from 'data/'...")
    docs = load_documents("data")

    if not docs:
        print("No documents found. Add PDFs or .txt files to the data/ folder first.")
        return

    print("\nChunking documents...")
    chunks = chunk_documents(docs, chunk_size=800, overlap_sentences=1)
    print(f"Produced {len(chunks)} chunk(s).")

    print("\nEmbedding and storing in Chroma...")
    build_index(chunks)

    print("\nBuild complete. You can now run: python main.py chat")


def run_chat():
    """Stages 4-5: interactive question-answering loop."""
    # Fail fast with a clear message if the index hasn't been built yet,
    # rather than letting an empty-collection query silently return nothing.
    collection = get_collection()
    if collection.count() == 0:
        print("No documents indexed yet. Run 'python main.py build' first.")
        return

    print(f"Ready - {collection.count()} chunk(s) indexed. Ask a question, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Stage 4: retrieval
        retrieved = query_index(question, top_k=3)

        # Show what was retrieved - useful for you to see retrieval
        # quality separately from generation quality, as planned back
        # in Step 4 of the build plan.
        print("\n  [Retrieved from:", end=" ")
        print(", ".join(f"{c['source']} (chunk {c['chunk_index']}, sim {c['similarity']:.2f})"
                         for c in retrieved), end="]\n\n")

        # Stage 5: prompt construction + generation
        messages = build_messages(question, retrieved)
        answer = generate_answer(messages)

        print(f"Assistant: {answer}\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "chat"):
        print("Usage:")
        print("  python main.py build   # (re)build the index from data/")
        print("  python main.py chat    # ask questions interactively")
        return

    command = sys.argv[1]

    if command == "build":
        run_build()
    elif command == "chat":
        run_chat()


if __name__ == "__main__":
    main()
