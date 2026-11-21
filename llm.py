from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model

from config import settings
from schemas import DocumentCategoryCreate

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = settings.OPENAI_MODEL


class DocumentAnalysisOutput(BaseModel):
    """Structured output returned by OpenAI for document summarization and category classification."""

    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "A concise 1-2 sentence summary capturing the key client/entity names, "
            "dates, financial or legal details, and purpose of the document."
        ),
    )
    category: str = Field(
        ...,
        min_length=1,
        description="The single best-matching category slug from the defined document categories taxonomy.",
    )


def build_analysis_response_model(
    categories: Sequence[DocumentCategoryCreate],
) -> type[DocumentAnalysisOutput]:
    """Dynamically construct a DocumentAnalysisOutput subclass whose `category` field is restricted to the DB category slugs."""
    slugs = tuple(cat.category for cat in categories)
    if not slugs:
        return DocumentAnalysisOutput

    category_literal_type = Literal[slugs]  # type: ignore[valid-type]
    dynamic_model = create_model(
        "DocumentAnalysisOutput",
        __base__=DocumentAnalysisOutput,
        category=(
            category_literal_type,
            Field(
                ...,
                description="The single best-matching category slug from the defined document categories taxonomy.",
            ),
        ),
    )
    return cast(type[DocumentAnalysisOutput], dynamic_model)


def build_category_system_prompt(
    categories: Sequence[DocumentCategoryCreate],
) -> str:
    """Build the system prompt instructing the model to summarize and classify a document using defined categories."""
    taxonomy_lines: list[str] = []
    for cat in categories:
        examples_str = ", ".join(cat.examples) if cat.examples else "N/A"
        taxonomy_lines.append(
            f"- slug: `{cat.category}` | Name: {cat.name}\n"
            f"  Summary: {cat.summary}\n"
            f"  Examples: {examples_str}"
        )
    taxonomy_block = "\n".join(taxonomy_lines)

    return (
        "You are an expert wealth management document classifier and summarizer.\n"
        "Given a client document's title and content, generate:\n"
        "1. `summary`: A concise 1-2 sentence summary of the document capturing the core subject, "
        "relevant parties, dates, addresses or financial/legal details.\n"
        "2. `category`: The single most appropriate category slug from the following defined taxonomy:\n\n"
        f"{taxonomy_block}"
    )


def get_openai_client(api_key: str | None = None) -> AsyncOpenAI | None:
    """Create an AsyncOpenAI client if OPENAI_API_KEY is configured, otherwise return None."""
    resolved_key = (
        api_key if api_key is not None else settings.OPENAI_API_KEY
    ).strip()
    if not resolved_key:
        return None
    return AsyncOpenAI(api_key=resolved_key)


async def analyze_document(
    *,
    title: str,
    content: str,
    categories: Sequence[DocumentCategoryCreate],
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> DocumentAnalysisOutput | None:
    """
    Generate a formatted summary and category slug for a document using OpenAI Structured Outputs.
    Gracefully falls back to None (leaving summary and category empty) if OPENAI_API_KEY is not
    configured, if no categories are provided, or if the OpenAI request fails.
    """
    if not categories:
        logger.info(
            "No document categories available; skipping LLM document enrichment for '%s'.",
            title,
        )
        return None

    openai_client = client if client is not None else get_openai_client()
    if openai_client is None:
        logger.info(
            "OPENAI_API_KEY is not configured; skipping LLM document enrichment for '%s'.",
            title,
        )
        return None

    model_name = (
        model or settings.OPENAI_MODEL or DEFAULT_OPENAI_MODEL
    ).strip() or DEFAULT_OPENAI_MODEL
    system_prompt = build_category_system_prompt(categories)
    user_prompt = f"Document Title: {title}\n\nDocument Content:\n{content}"
    response_model = build_analysis_response_model(categories)

    try:
        completion = await openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
        )
        message = completion.choices[0].message
        return message.parsed
    except Exception as exc:
        logger.warning(
            "OpenAI document analysis failed for '%s' using model '%s': %s. "
            "Falling back to empty summary and category.",
            title,
            model_name,
            exc,
        )
        return None
