"""
document_loader.py

Loads all supported documents (.pdf, .txt) from a folder and returns
their raw text content, tagged with the source filename.

This is Stage 1a of the RAG pipeline: turning files on disk into
plain text strings we can later chunk and embed.
"""

import os
from pypdf import PdfReader


def load_pdf(filepath: str) -> str:
    """
    Extract raw text from a single PDF file.

    Note: PDF text extraction reconstructs reading order from positioned
    characters. Simple single-column PDFs extract cleanly; multi-column
    layouts, tables, and scanned (image-based) PDFs may extract poorly
    or produce empty text. That's a known limitation, not a bug here.
    """
    reader = PdfReader(filepath)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # some pages (e.g. scanned/image pages) may return None or ""
            pages_text.append(text)
    return "\n".join(pages_text)


def load_txt(filepath: str) -> str:
    """Read a plain text file as-is."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(data_dir: str = "data") -> list[dict]:
    """
    Load every supported file in data_dir.

    Returns a list of dicts, one per document:
        {
            "source": "lecture3.pdf",   # filename, used later for citations/debugging
            "text": "full extracted raw text of the document..."
        }

    Unsupported file types are skipped with a printed warning, rather
    than crashing the whole load.
    """
    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            f"Create it and add some .pdf or .txt files first."
        )

    filenames = sorted(os.listdir(data_dir))

    if not filenames:
        print(f"Warning: '{data_dir}' is empty. Add some .pdf or .txt files to it.")
        return documents

    for filename in filenames:
        filepath = os.path.join(data_dir, filename)

        if not os.path.isfile(filepath):
            continue  # skip subfolders, if any

        lower_name = filename.lower()

        if lower_name.endswith(".pdf"):
            text = load_pdf(filepath)
        elif lower_name.endswith(".txt"):
            text = load_txt(filepath)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        if not text.strip():
            print(f"Warning: no text extracted from '{filename}' (empty or scanned PDF?)")
            continue

        documents.append({"source": filename, "text": text})
        print(f"Loaded '{filename}' ({len(text)} characters)")

    return documents


if __name__ == "__main__":
    # Quick manual test: run this file directly to see what gets loaded
    # from your data/ folder, and confirm extraction looks sane.
    docs = load_documents("data")

    print(f"\nLoaded {len(docs)} document(s) total.\n")

    for doc in docs:
        preview = doc["text"][:300].replace("\n", " ")
        print(f"--- {doc['source']} ---")
        print(f"{preview}...\n")
