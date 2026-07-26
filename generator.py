"""
generator.py

Stage 5 (part 2) of the RAG pipeline: sends the constructed messages
(system + user, built by prompt_builder.py) to the Groq API and returns
the generated answer.

Uses non-streaming responses for simplicity - the full answer is
returned in one go rather than printed token-by-token.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # loads GROQ_API_KEY from your .env file into the environment

_client = None


def get_client() -> Groq:
    """Lazily create and cache the Groq client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Make sure you have a .env file "
                "with GROQ_API_KEY=your_key_here in your project folder."
            )
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(
    messages: list[dict],
    model: str = "openai/gpt-oss-20b",
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    Send messages to the Groq API and return the generated answer text.

    Args:
        messages: list of {"role": ..., "content": ...} dicts, as built
            by prompt_builder.build_messages().
        model: which Groq-hosted model to use.
        temperature: lower = more deterministic/factual, higher = more varied.
        max_tokens: maximum length of the generated response.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick manual test: run the full pipeline end to end for one
    # question - retrieval, prompt construction, and generation together.
    from vector_store import query_index
    from prompt_builder import build_messages

    question = "how is the weather"

    retrieved = query_index(question, top_k=3)
    messages = build_messages(question, retrieved)

    print(f"Question: {question}\n")
    print("Retrieved chunks from:")
    for chunk in retrieved:
        print(f"  - {chunk['source']} (chunk {chunk['chunk_index']}, similarity {chunk['similarity']:.3f})")

    print("\nGenerating answer...\n")
    answer = generate_answer(messages)

    print("--- ANSWER ---")
    print(answer)
