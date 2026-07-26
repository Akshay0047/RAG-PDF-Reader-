"""
prompt_builder.py

Stage 5 (part 1) of the RAG pipeline: builds the actual messages sent
to the LLM, combining the system instructions with the retrieved
context chunks and the user's question.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below, which comes from the user's own documents. "
    "If the answer is not present in the context, say clearly that you "
    "don't have enough information in the provided documents to answer - "
    "do not make up an answer from outside knowledge. "
    "When useful, mention which source document your answer is based on."
)


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string, each labeled
    with its source document, so the LLM can reference where information
    came from.

    Args:
        retrieved_chunks: list of dicts as returned by
            vector_store.query_index(), each with "text" and "source".
    """
    formatted_sections = []
    for chunk in retrieved_chunks:
        formatted_sections.append(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
        )
    return "\n\n".join(formatted_sections)


def build_messages(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    """
    Build the full messages list to send to the LLM API.

    Returns a list in the format Groq's (and most chat LLM APIs') expect:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
    """
    context = format_context(retrieved_chunks)

    user_message = (
        f"Context from documents:\n\n{context}\n\n"
        f"Question: {question}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


if __name__ == "__main__":
    # Quick manual test: retrieve real chunks for a sample question and
    # inspect the exact prompt that would be sent to the LLM.
    from vector_store import query_index

    question = "what is round robin scheduling"
    retrieved = query_index(question, top_k=3)

    messages = build_messages(question, retrieved)

    print("--- SYSTEM MESSAGE ---\n")
    print(messages[0]["content"])
    print("\n--- USER MESSAGE ---\n")
    print(messages[1]["content"])
