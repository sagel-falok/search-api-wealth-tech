from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from db import AsyncSessionLocal, DocumentChunkModel
from embeddings import build_document_chunks
from main import app
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
