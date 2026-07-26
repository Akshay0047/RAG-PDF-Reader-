"""
main_langchain.py

The SAME RAG pipeline as main.py + document_loader.py + chunker.py +
embedder.py + vector_store.py + prompt_builder.py + generator.py,
rebuilt using LangChain's abstractions.

Compare this file against your hand-built version to see exactly what
each LangChain piece replaces:

    Your code                          LangChain equivalent
    ------------------------------      ------------------------------
    document_loader.py                  PyPDFLoader / TextLoader
    chunker.py (sentence-aware)         RecursiveCharacterTextSplitter
    embedder.py (MiniLM wrapper)        HuggingFaceEmbeddings
    vector_store.py (Chroma client)     our own chromadb client (langchain-chroma
                                         doesn't yet support chromadb 1.x - see note below)
    prompt_builder.py                   ChatPromptTemplate
    generator.py (Groq client)          ChatGroq
    main.py (manual glue code)          a chain built with LCEL ( | )

Usage:
    python main_langchain.py build
    python main_langchain.py chat
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# NOTE: langchain-chroma does not yet support chromadb 1.x (Chroma's Rust
# rewrite) - every langchain-chroma release so far caps chromadb below
# 0.7.0, which conflicts with the chromadb==1.0.21 we need for a clean
# Windows install (see requirements.txt history). Rather than fight that
# version conflict, we reuse our own chromadb client directly here (same
# approach as vector_store.py) for storage/retrieval, and use LangChain
# for everything else (loader, splitter, prompt template, LLM call).
# This is a realistic situation: an integration package can lag behind
# a fast-moving core dependency, and routing around it is a normal fix.
import chromadb

load_dotenv()

PERSIST_DIR = "chroma_db_langchain"  # separate folder so it doesn't clash with your hand-built index
COLLECTION_NAME = "documents_langchain"
DATA_DIR = "data"

# Same model choices as your hand-built version, for a fair comparison.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below, which comes from the user's own documents. "
    "If the answer is not present in the context, say clearly that you "
    "don't have enough information in the provided documents to answer - "
    "do not make up an answer from outside knowledge. "
    "When useful, mention which source document your answer is based on."
)


def load_documents_langchain(data_dir: str = DATA_DIR):
    """
    Equivalent of document_loader.py's load_documents().

    LangChain loaders return a list of Document objects, each with
    .page_content (the text) and .metadata (a dict, auto-populated with
    at least the source file path).
    """
    all_docs = []

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        docs = loader.load()
        # PyPDFLoader returns one Document PER PAGE - metadata["source"] is
        # already the file path, which is enough for our citation purposes.
        all_docs.extend(docs)
        print(f"Loaded '{filename}' ({len(docs)} page/section(s))")

    return all_docs


def run_build():
    """
    Equivalent of chunker.py + embedder.py + vector_store.py's build_index(),
    combined - this is exactly where LangChain's composability shows up:
    what took 3 separate files by hand is a few chained calls here.
    """
    print("Loading documents...")
    raw_docs = load_documents_langchain()

    if not raw_docs:
        print("No documents found in data/.")
        return

    print("\nChunking documents...")
    # RecursiveCharacterTextSplitter tries to split on paragraph breaks
    # first, then sentences, then words - a more automated version of
    # the sentence-aware strategy you built by hand in chunker.py.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,  # character-based overlap, unlike your sentence-based overlap
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Produced {len(chunks)} chunk(s).")

    print("\nEmbedding and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # LangChain's embed_documents() does the same job as your embedder.py's
    # embed_texts() - it just calls MiniLM under the hood, wrapped in a
    # standard interface.
    texts = [doc.page_content for doc in chunks]
    vectors = embeddings.embed_documents(texts)

    # Same raw chromadb client + PersistentClient pattern as vector_store.py -
    # this is the piece langchain-chroma would normally wrap for us.
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    print(f"\nBuild complete. {collection.count()} chunk(s) saved to '{PERSIST_DIR}'. "
          f"Run: python main_langchain.py chat")


def format_docs(retrieved_chunks: list[dict]) -> str:
    """
    Equivalent of prompt_builder.py's format_context() - labels each
    retrieved chunk with its source so the LLM (and we, when debugging)
    can tell where each piece of context came from.
    """
    sections = []
    for chunk in retrieved_chunks:
        sections.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(sections)


def retrieve(question: str, embeddings, collection, top_k: int = 3) -> list[dict]:
    """
    Equivalent of vector_store.py's query_index() - embeds the question
    and asks chromadb for the top_k most similar stored chunks.
    (This is normally what a langchain-chroma "retriever" would wrap for
    us - see the note at the top of this file on why we're calling
    chromadb directly instead.)
    """
    query_vector = embeddings.embed_query(question)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)

    retrieved = []
    for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({"text": text, "source": metadata.get("source", "unknown")})
    return retrieved


def run_chat():
    """
    Equivalent of vector_store.py's query_index() + prompt_builder.py's
    build_messages() + generator.py's generate_answer() + main.py's chat
    loop - composed here using LangChain's prompt template and LLM call,
    with retrieval handled by our own chromadb client (see note above).
    """
    if not os.path.isdir(PERSIST_DIR):
        print(f"No index found at '{PERSIST_DIR}'. Run 'python main_langchain.py build' first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        print("Index is empty. Run 'python main_langchain.py build' first.")
        return

    # ChatPromptTemplate is LangChain's version of prompt_builder.py's
    # build_messages() - same system+user role split, template syntax
    # instead of an f-string.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Context from documents:\n\n{context}\n\nQuestion: {question}"),
    ])

    llm = ChatGroq(model=LLM_MODEL, temperature=0.2, max_tokens=500)

    # THE CHAIN - this is LangChain's actual value-add over your hand-built
    # main.py: prompt-building and generation are declared as a pipeline
    # using the | operator. (Retrieval is done just before invoking the
    # chain below, since we're using our own chromadb call for that part.)
    chain = prompt | llm | StrOutputParser()

    print(f"Ready - {collection.count()} chunk(s) indexed. Ask a question, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        retrieved = retrieve(question, embeddings, collection, top_k=3)
        print("\n  [Retrieved from:", end=" ")
        print(", ".join(c["source"] for c in retrieved), end="]\n\n")

        context = format_docs(retrieved)
        answer = chain.invoke({"context": context, "question": question})
        print(f"Assistant: {answer}\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "chat"):
        print("Usage:")
        print("  python main_langchain.py build")
        print("  python main_langchain.py chat")
        return

    if sys.argv[1] == "build":
        run_build()
    else:
        run_chat()


if __name__ == "__main__":
    main()