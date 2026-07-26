# Chat With My Documents — A Hand-Built RAG Pipeline

A small Retrieval-Augmented Generation (RAG) app that lets you ask questions about your own documents (PDFs/text files) and get answers grounded in their content — built from scratch, component by component, to understand exactly how RAG works under the hood rather than just using a framework.

This project includes **two implementations of the same pipeline**:
1. **A hand-built version** — every stage (chunking, embeddings, vector storage, retrieval, prompt construction, generation) written from raw Python, with no orchestration framework.
2. **A LangChain version** — the same pipeline rebuilt using LangChain's abstractions, for direct comparison against the hand-built version.

Building both was intentional: the hand-built version forces you to understand every moving part, and having done that, the LangChain abstractions become transparent instead of magic.

## Why This Exists

Most RAG tutorials jump straight to `from langchain import ...` and a working demo in 10 lines — which teaches you the vocabulary, but not the mechanics. This project goes the other way: build every stage by hand first, understand it, *then* see what a framework abstracts away.

## How RAG Works (Quick Overview)

1. **Chunking** — split documents into small, coherent, overlapping text chunks.
2. **Embedding** — convert each chunk into a vector (list of numbers) that captures its meaning, using a local embedding model (`all-MiniLM-L6-v2`).
3. **Vector storage** — store chunks + vectors in Chroma, a local vector database.
4. **Retrieval** — embed the user's question, then find the most semantically similar stored chunks via cosine similarity.
5. **Generation** — inject the retrieved chunks into a prompt and send it to an LLM (via Groq's API), which answers grounded in that context instead of guessing from its training data alone.

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem fit for AI/ML tooling |
| PDF/text loading | `pypdf` | Simple, no heavy dependencies |
| Chunking | Hand-written sentence-aware chunker (+ LangChain's `RecursiveCharacterTextSplitter` in the LangChain version) | Sentence-boundary aware, with overlap to avoid severing ideas |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local, free | Fast on CPU, no API key or cost |
| Vector storage | `chromadb`, persisted locally | Simple embedded vector DB, no server to run |
| LLM (generation) | Groq API (`openai/gpt-oss-20b`) | Free tier, very fast inference |
| Orchestration | Plain Python (hand-built version) / LangChain (comparison version) | See both approaches side by side |

## Project Structure

```
rag-chat/
├── data/                      # your source documents (PDFs/.txt) go here
├── chroma_db/                  # persisted vector index (hand-built version) - auto-created
├── chroma_db_langchain/         # persisted vector index (LangChain version) - auto-created
│
├── document_loader.py           # Stage 1: load PDFs/text into raw strings
├── chunker.py                    # Stage 1b: sentence-aware chunking with overlap
├── embedder.py                   # Stage 2: text -> embedding vectors (MiniLM)
├── vector_store.py               # Stage 3-4: Chroma storage + retrieval
├── prompt_builder.py              # Stage 5a: build system/user messages from retrieved chunks
├── generator.py                   # Stage 5b: call Groq API for the final answer
├── main.py                        # wires everything together (build + chat CLI)
│
├── main_langchain.py              # same pipeline rebuilt using LangChain
│
├── requirements.txt
├── .env.example                    # template for your Groq API key
└── .gitignore
```

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd rag-chat

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Git Bash):
source venv/Scripts/activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Get a free API key from [console.groq.com](https://console.groq.com), then:

```bash
cp .env.example .env
# edit .env and paste your key: GROQ_API_KEY=your_key_here
```

Add some PDFs or `.txt` files to the `data/` folder.

## Usage

### Hand-built version

```bash
python main.py build   # ingest documents: chunk -> embed -> store
python main.py chat    # ask questions interactively
```

### LangChain version

```bash
python main_langchain.py build
python main_langchain.py chat
```

Both maintain separate persisted indexes (`chroma_db/` vs `chroma_db_langchain/`), so you can build and query either independently.

### Testing individual stages

Each pipeline file can also be run directly to inspect its output in isolation:

```bash
python document_loader.py    # see extracted raw text per document
python chunker.py             # see chunks produced, with overlap
python embedder.py            # inspect actual embedding vectors + a similarity sanity check
python vector_store.py query "your question here"   # test retrieval only
python prompt_builder.py      # see the exact prompt sent to the LLM
python generator.py           # run one full question through the entire pipeline
```

## Design Notes / Key Decisions

- **Chunking:** sentence-aware, ~800 characters per chunk, 1-sentence overlap — avoids severing ideas mid-sentence while keeping chunks focused enough for good retrieval precision.
- **Embedding model:** `all-MiniLM-L6-v2` (384 dimensions) — small, fast, free, and good enough quality for typical prose/technical documents. Bigger models exist and would improve nuance on ambiguous text, at the cost of speed and size.
- **Similarity metric:** cosine similarity — measures semantic *direction* rather than raw distance, which is what makes "meaning-based" search work even when the user's wording differs completely from the document's wording.
- **Prompt design:** the system prompt explicitly instructs the model to answer only from provided context and to say so if the answer isn't present — this is what keeps answers grounded and reduces hallucination.
- **Why two implementations:** building the hand-written version first, then comparing it to LangChain's abstractions, made it possible to actually understand what each LangChain component was doing internally, rather than treating it as a black box.

## Known Issues Solved Along the Way

A few real dependency issues came up building this on Windows — worth documenting since they're common and not obvious from error messages alone:

- **`chroma-hnswlib` build failure on Windows** (`Microsoft Visual C++ 14.0 required`): fixed by using `chromadb>=1.0`, which is Rust-based and ships prebuilt wheels for Windows + Python 3.12, avoiding the need for a C++ compiler entirely.
- **`Client.__init__() got an unexpected keyword argument 'proxies'`**: caused by `httpx>=0.28` removing a parameter the `groq` SDK still passed internally. Fixed by pinning `httpx==0.27.2`.
- **`langchain-chroma` dependency conflict with `chromadb==1.0.x`**: `langchain-chroma` hasn't yet been updated for Chroma's Rust-based 1.x release, so it caps `chromadb<0.7.0`. Rather than downgrade Chroma (and reintroduce the Windows build issue), the LangChain version calls the `chromadb` client directly for storage/retrieval, using LangChain only for the loader, splitter, prompt template, and LLM call.

## Possible Extensions

- Conversation memory, so follow-up questions ("what about UDP?") work naturally
- Streaming responses instead of waiting for the full answer
- A simple web UI (React/Django REST Framework)
- Swap in a larger embedding model and compare retrieval quality
- Support more file types (`.docx`, `.md`, web pages)

## License

MIT (or your preferred license — update as needed)
