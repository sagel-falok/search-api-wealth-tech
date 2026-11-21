from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

import repo
from db import AsyncSessionLocal, DocumentChunkModel
from embeddings import build_document_chunks
from llm import (
    DEFAULT_OPENAI_MODEL,
    DocumentAnalysisOutput,
    analyze_document,
    build_category_system_prompt,
)
from main import app
from schemas import DocumentCategoryCreate
from seed import seed_all


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def seeded_db() -> AsyncGenerator[None, None]:
    from db import engine

    await engine.dispose()
    await seed_all()
    yield
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio(loop_scope="module")
async def test_document_chunks_populated_on_seed() -> None:
    """Verify db-sync populates contextual chunks and 768-d embeddings in document_chunks."""
    async with AsyncSessionLocal() as session:
        count = await session.scalar(
            select(func.count()).select_from(DocumentChunkModel)
        )
        assert count is not None and count >= 15

        sample_chunk = await session.scalar(
            select(DocumentChunkModel).limit(1)
        )
        assert sample_chunk is not None
        assert sample_chunk.chunk_index == 0
        assert "Document:" in sample_chunk.chunk_text
        assert len(sample_chunk.embedding) == 768


@pytest.mark.asyncio(loop_scope="module")
async def test_search_address_proof_returns_utility_bills(
    client: AsyncClient,
) -> None:
    """Core PDF requirement: searching 'address proof' returns documents containing 'utility bill'."""
    response = await client.get("/search", params={"q": "address proof"})
    assert response.status_code == 200
    data = response.json()

    docs = [item for item in data if item["type"] == "document"]
    assert len(docs) >= 3

    titles = [doc["title"] for doc in docs]
    assert (
        "Thames Water & EDF Electricity Utility Bill - October 2026" in titles
    )
    assert "Eversource Electric & Boston Water Utility Statement" in titles
    assert "City of Westminster Annual Council Tax Bill 2026/2027" in titles

    # Top document results should all be address_proof category with summaries populated
    for doc in docs[:3]:
        assert doc["category"] == "address_proof"
        assert doc["summary"] is not None and len(doc["summary"]) > 10


@pytest.mark.asyncio(loop_scope="module")
async def test_search_semantic_synonyms_without_exact_keywords(
    client: AsyncClient,
) -> None:
    """Verify semantic pgvector search matches conceptual synonyms not present verbatim in document text."""
    # 'electricity invoice' -> utility bills (word 'invoice' does not appear in any document)
    resp_elec = await client.get("/search", params={"q": "electricity invoice"})
    assert resp_elec.status_code == 200
    elec_titles = [
        item["title"]
        for item in resp_elec.json()
        if item["type"] == "document"
    ]
    assert (
        "Thames Water & EDF Electricity Utility Bill - October 2026"
        in elec_titles
    )
    assert (
        "Eversource Electric & Boston Water Utility Statement" in elec_titles
    )

    # 'living will and estate succession' -> Doe Family Revocable Living Trust Deed
    resp_trust = await client.get(
        "/search", params={"q": "estate succession and living will"}
    )
    assert resp_trust.status_code == 200
    trust_docs = [
        item for item in resp_trust.json() if item["type"] == "document"
    ]
    assert len(trust_docs) >= 1
    assert (
        trust_docs[0]["title"] == "The Doe Family Revocable Living Trust Deed"
    )

    # 'pension valuation' -> Meridian SIPP Valuation
    resp_pension = await client.get(
        "/search", params={"q": "pension valuation"}
    )
    assert resp_pension.status_code == 200
    pension_docs = [
        item for item in resp_pension.json() if item["type"] == "document"
    ]
    assert len(pension_docs) >= 1
    assert (
        pension_docs[0]["title"]
        == "Meridian Executive Self-Invested Personal Pension (SIPP) Valuation"
    )

    # 'fine art insurance' -> Chubb Masterpiece Fine Art Schedule
    resp_art = await client.get("/search", params={"q": "fine art insurance"})
    assert resp_art.status_code == 200
    art_docs = [item for item in resp_art.json() if item["type"] == "document"]
    assert len(art_docs) >= 1
    assert (
        art_docs[0]["title"]
        == "Chubb Masterpiece Fine Art Schedule & Sotheby's Provenance Appraisal"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_search_multi_entity_client_and_document_synergy(
    client: AsyncClient,
) -> None:
    """Verify multi-entity queries ('Client Name + Document Type') rank that specific client's document #1."""
    # John Doe utility bill -> John Doe's Thames Water bill ranks above Arjun Patel's Eversource bill
    resp_john = await client.get(
        "/search", params={"q": "John Doe utility bill"}
    )
    assert resp_john.status_code == 200
    john_docs = [
        item for item in resp_john.json() if item["type"] == "document"
    ]
    assert len(john_docs) >= 2
    john_titles = [d["title"] for d in john_docs]
    assert (
        john_titles[0]
        == "Thames Water & EDF Electricity Utility Bill - October 2026"
    )
    assert john_titles.index(
        "Thames Water & EDF Electricity Utility Bill - October 2026"
    ) < john_titles.index(
        "Eversource Electric & Boston Water Utility Statement"
    )

    # Arjun Patel utility bill -> Arjun Patel's Eversource bill ranks #1 above John Doe's Thames Water bill
    resp_arjun = await client.get(
        "/search", params={"q": "Arjun Patel utility bill"}
    )
    assert resp_arjun.status_code == 200
    arjun_docs = [
        item for item in resp_arjun.json() if item["type"] == "document"
    ]
    assert len(arjun_docs) >= 2
    arjun_titles = [d["title"] for d in arjun_docs]
    assert (
        arjun_titles[0]
        == "Eversource Electric & Boston Water Utility Statement"
    )
    assert arjun_titles.index(
        "Eversource Electric & Boston Water Utility Statement"
    ) < arjun_titles.index(
        "Thames Water & EDF Electricity Utility Bill - October 2026"
    )

    # Alistair Sterling address proof -> Alistair Sterling's Westminster Council Tax Bill ranks #1
    resp_alistair = await client.get(
        "/search", params={"q": "Alistair Sterling address proof"}
    )
    assert resp_alistair.status_code == 200
    alistair_docs = [
        item for item in resp_alistair.json() if item["type"] == "document"
    ]
    assert len(alistair_docs) >= 1
    assert (
        alistair_docs[0]["title"]
        == "City of Westminster Annual Council Tax Bill 2026/2027"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_search_typo_insensitive_documents(client: AsyncClient) -> None:
    """Verify pg_trgm + pgvector hybrid search handles severe typos like 'adres proof', 'utlity bill', and 'pasport'."""
    # 'utlity bill'
    resp_util = await client.get("/search", params={"q": "utlity bill"})
    assert resp_util.status_code == 200
    util_titles = [
        item["title"]
        for item in resp_util.json()
        if item["type"] == "document"
    ]
    assert (
        "Thames Water & EDF Electricity Utility Bill - October 2026"
        in util_titles
    )
    assert (
        "Eversource Electric & Boston Water Utility Statement" in util_titles
    )

    # 'adres proof' (where vector similarity alone drops to ~0.53, rescued by category_search_text trigram)
    resp_adres = await client.get("/search", params={"q": "adres proof"})
    assert resp_adres.status_code == 200
    adres_docs = [
        item for item in resp_adres.json() if item["type"] == "document"
    ]
    assert len(adres_docs) >= 3
    adres_categories = {d["category"] for d in adres_docs[:3]}
    assert adres_categories == {"address_proof"}

    # 'pasport verification'
    resp_pass = await client.get(
        "/search", params={"q": "pasport verification"}
    )
    assert resp_pass.status_code == 200
    pass_docs = [
        item for item in resp_pass.json() if item["type"] == "document"
    ]
    assert len(pass_docs) >= 1
    assert (
        pass_docs[0]["title"]
        == "United States Passport - Biometric Identity Verification"
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_search_unrelated_queries_return_empty(
    client: AsyncClient,
) -> None:
    """Verify the 0.60 semantic threshold and lexical/trigram gates reject unrelated queries."""
    for unrelated_query in (
        "zzzznonexistentclient99999",
        "chocolate cake baking recipe",
        "mars rover telemetry",
    ):
        response = await client.get("/search", params={"q": unrelated_query})
        assert response.status_code == 200
        assert response.json() == []


def test_contextual_chunking_handles_long_documents() -> None:
    """Verify documents exceeding 512 tokens are split into Chunk 0 (overview) + overlapping content chunks."""
    long_content = " ".join(
        [f"Clause {i}: trust distribution and fiduciary covenant terms." for i in range(1, 80)]
    )
    chunks = build_document_chunks(
        title="Master Family Dynasty Trust Deed",
        content=long_content,
        summary="Multi-generational dynasty trust deed.",
        client_first_name="John",
        client_last_name="Doe",
        client_email="john.doe@neviswealth.com",
        category_slug="estate_planning_and_trusts",
        category_name="Estate Planning, Wills & Trusts",
        category_summary="Fiduciary and legal instruments governing wealth transfer.",
        category_examples=["Revocable Living Trust Agreement", "Last Will and Testament"],
    )

    assert len(chunks) >= 4
    # Chunk 0 is the overview + category taxonomy chunk
    assert "Related Document Types:" in chunks[0]
    assert "Summary: Multi-generational dynasty trust deed." in chunks[0]

    # Chunks 1..N carry the compact header and bounded content windows (< 512 tokens)
    for chunk in chunks[1:]:
        assert "Document: Master Family Dynasty Trust Deed | Client: John Doe" in chunk
        assert "Content: Clause" in chunk
        assert len(chunk.split()) <= 250


@pytest.mark.asyncio(loop_scope="module")
async def test_analyze_document_uses_gpt_6_luna_and_defined_categories() -> None:
    """Verify analyze_document accepts categories and calls OpenAI structured output with gpt-6-luna."""
    async with AsyncSessionLocal() as session:
        db_categories = await repo.list_document_categories(session)
    categories = [
        DocumentCategoryCreate(
            category=cat.category,
            name=cat.name,
            summary=cat.summary,
            examples=cat.examples or [],
        )
        for cat in db_categories
    ]
    assert len(categories) == 16
    prompt = build_category_system_prompt(categories)
    assert "address_proof" in prompt
    assert "estate_planning_and_trusts" in prompt

    expected_parsed = DocumentAnalysisOutput(
        summary="November 2026 British Gas heating bill confirming residential address in London.",
        category="address_proof",
    )
    mock_parse = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=expected_parsed))]
        )
    )
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = mock_parse

    result = await analyze_document(
        title="British Gas Statement - Nov 2026",
        content="British Gas residential heating statement for 42 Kensington Palace Gardens.",
        categories=categories,
        client=mock_client,
    )
    assert result is not None
    assert result.category == "address_proof"
    assert result.summary == expected_parsed.summary

    mock_parse.assert_awaited_once()
    call_kwargs = mock_parse.await_args.kwargs
    assert call_kwargs["model"] == DEFAULT_OPENAI_MODEL == "gpt-6-luna"
    response_format = call_kwargs["response_format"]
    assert issubclass(response_format, DocumentAnalysisOutput)
    schema = response_format.model_json_schema()
    assert set(schema["properties"]["category"]["enum"]) == {
        cat.category for cat in categories
    }


@pytest.mark.asyncio(loop_scope="module")
async def test_create_document_pipeline_with_openai_and_chunks(
    client: AsyncClient,
) -> None:
    """Verify POST /clients/{id}/documents enriches summary & category via OpenAI, stores chunks + vectors, and is searchable."""
    async with AsyncSessionLocal() as session:
        john = await repo.get_client_by_email(session, "john.doe@neviswealth.com")
        assert john is not None
        john_id = john.id

    mock_analysis = DocumentAnalysisOutput(
        summary="October 2026 Swiss cantonal wealth and property tax assessment for John Doe's Zurich residence.",
        category="tax_filings_and_reporting",
    )

    with patch(
        "repo.analyze_document",
        new=AsyncMock(return_value=mock_analysis),
    ):
        resp = await client.post(
            f"/clients/{john_id}/documents",
            json={
                "title": "Zurich Cantonal Wealth Tax Assessment 2026",
                "content": (
                    "Official Swiss cantonal tax assessment notice issued to John Doe "
                    "for fiscal year 2026 covering net wealth and municipal tax obligations in Zurich."
                ),
            },
        )

    assert resp.status_code == 201
    created_doc = resp.json()
    doc_id = UUID(created_doc["id"])

    try:
        assert created_doc["summary"] == mock_analysis.summary
        assert created_doc["category"] == "tax_filings_and_reporting"
        assert created_doc["category_id"] is not None

        # Verify chunks and 768-d vectors were created in document_chunks
        async with AsyncSessionLocal() as session:
            chunks = (
                await session.execute(
                    select(DocumentChunkModel)
                    .where(DocumentChunkModel.document_id == doc_id)
                    .order_by(DocumentChunkModel.chunk_index)
                )
            ).scalars().all()
            assert len(chunks) >= 1
            assert "Zurich Cantonal Wealth Tax Assessment 2026" in chunks[0].chunk_text
            assert "Tax Filings & Regulatory Tax Reporting" in chunks[0].chunk_text
            assert mock_analysis.summary in chunks[0].chunk_text
            assert len(chunks[0].embedding) == 768

        # Verify the newly created document is searchable semantically & lexically
        search_resp = await client.get(
            "/search", params={"q": "Zurich cantonal wealth tax"}
        )
        assert search_resp.status_code == 200
        doc_ids = [
            item["id"]
            for item in search_resp.json()
            if item["type"] == "document"
        ]
        assert str(doc_id) in doc_ids
    finally:
        async with AsyncSessionLocal() as session:
            await repo.delete_document(session, doc_id)


@pytest.mark.asyncio(loop_scope="module")
async def test_create_document_pipeline_graceful_fallback_without_openai_key(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /clients/{id}/documents gracefully leaves summary & category empty when OPENAI_API_KEY is unset, while still indexing chunks and vectors."""
    from config import settings

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")

    async with AsyncSessionLocal() as session:
        john = await repo.get_client_by_email(session, "john.doe@neviswealth.com")
        assert john is not None
        john_id = john.id

    resp = await client.post(
        f"/clients/{john_id}/documents",
        json={
            "title": "Bespoke Family Aviation Charter Memo",
            "content": (
                "Private aviation charter memorandum for John Doe detailing NetJets fractional "
                "flight hours between London Farnborough and Geneva."
            ),
        },
    )
    assert resp.status_code == 201
    created_doc = resp.json()
    doc_id = UUID(created_doc["id"])

    try:
        assert created_doc["summary"] is None
        assert created_doc["category_id"] is None
        assert created_doc["category"] is None

        # Verify chunks and 768-d vectors are still generated even when LLM enrichment is skipped
        async with AsyncSessionLocal() as session:
            chunks = (
                await session.execute(
                    select(DocumentChunkModel)
                    .where(DocumentChunkModel.document_id == doc_id)
                    .order_by(DocumentChunkModel.chunk_index)
                )
            ).scalars().all()
            assert len(chunks) == 1
            assert "Bespoke Family Aviation Charter Memo" in chunks[0].chunk_text
            assert len(chunks[0].embedding) == 768

        # Verify document is still discoverable via search
        search_resp = await client.get(
            "/search", params={"q": "NetJets fractional aviation charter"}
        )
        assert search_resp.status_code == 200
        doc_ids = [
            item["id"]
            for item in search_resp.json()
            if item["type"] == "document"
        ]
        assert str(doc_id) in doc_ids
    finally:
        async with AsyncSessionLocal() as session:
            await repo.delete_document(session, doc_id)

