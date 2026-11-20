from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# Safe word-window limits to stay well under BGE-base-en-v1.5's 512-token limit (~350-380 tokens max per chunk)
SHORT_CONTENT_WORD_LIMIT = 160
CHUNK_WINDOW_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Load and cache the FastEmbed BAAI/bge-base-en-v1.5 model singleton."""
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


def embed_passages(texts: Sequence[str]) -> list[list[float]]:
    """Batch-embed document chunks into 768-dimensional float vectors."""
    if not texts:
        return []
    model = get_embedding_model()
    return [vec.tolist() for vec in model.embed(list(texts))]


def embed_query(query: str) -> list[float]:
    """Embed a search query using BGE query instruction prefix."""
    model = get_embedding_model()
    return next(iter(model.query_embed(query))).tolist()


def build_document_chunks(
    *,
    title: str,
    content: str,
    summary: str | None = None,
    client_first_name: str | None = None,
    client_last_name: str | None = None,
    client_email: str | None = None,
    category_slug: str | None = None,
    category_name: str | None = None,
    category_summary: str | None = None,
    category_examples: Sequence[str] | None = None,
) -> list[str]:
    """
    Build contextualized document chunks bounded under the 512-token limit of BAAI/bge-base-en-v1.5.

    - Chunk 0 (Overview & Taxonomy Chunk): Combines document title, client identity, category taxonomy
      (name, slug, summary, and example document types), and document summary. If content is short
      (<= 160 words), includes the full content directly in Chunk 0.
    - Chunks 1..N (Sliding Content Windows): For longer documents, splits content into overlapping
      200-word windows (40-word overlap) prefixed with a compact contextual header.
    """
    client_parts: list[str] = []
    full_name = " ".join(
        part for part in (client_first_name, client_last_name) if part
    ).strip()
    if full_name:
        client_parts.append(full_name)
    if client_email:
        client_parts.append(f"({client_email})")
    client_line = " ".join(client_parts)

    category_line = ""
    if category_name and category_slug:
        slug_words = category_slug.replace("_", " ")
        category_line = f"{category_name} ({slug_words})"
        if category_summary:
            category_line = f"{category_line} - {category_summary}"
    elif category_name:
        category_line = category_name
    elif category_slug:
        category_line = category_slug.replace("_", " ")

    overview_lines: list[str] = [f"Document: {title}"]
    if client_line:
        overview_lines.append(f"Client: {client_line}")
    if category_line:
        overview_lines.append(f"Category: {category_line}")
    if category_examples:
        overview_lines.append(
            f"Related Document Types: {', '.join(category_examples)}"
        )
    if summary:
        overview_lines.append(f"Summary: {summary}")

    words = content.split()
    if len(words) <= SHORT_CONTENT_WORD_LIMIT:
        overview_lines.append(f"Content: {content}")
        return ["\n".join(overview_lines)]

    chunks: list[str] = ["\n".join(overview_lines)]

    compact_header_parts = [f"Document: {title}"]
    if client_line:
        compact_header_parts.append(f"Client: {client_line}")
    if category_name:
        compact_header_parts.append(f"Category: {category_name}")
    compact_header = " | ".join(compact_header_parts)

    step = max(1, CHUNK_WINDOW_WORDS - CHUNK_OVERLAP_WORDS)
    for start in range(0, len(words), step):
        window_words = words[start : start + CHUNK_WINDOW_WORDS]
        if not window_words:
            break
        window_text = " ".join(window_words)
        chunks.append(f"{compact_header}\nContent: {window_text}")
        if start + CHUNK_WINDOW_WORDS >= len(words):
            break

    return chunks
