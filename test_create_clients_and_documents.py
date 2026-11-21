from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

import repo
from db import AsyncSessionLocal, DocumentChunkModel
from llm import DocumentAnalysisOutput
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
async def test_create_client_success_and_normalization(
    client: AsyncClient,
) -> None:
    """Verify POST /clients creates a client, normalizes name/email fields, and indexes it for search."""
    payload = {
        "first_name": "  Victoria  ",
        "last_name": "  Kensington ",
        "email": "  Victoria.Kensington@MayfairFamilyOffice.co.uk  ",
        "description": "Principal of Mayfair Family Office focusing on philanthropic endowments and offshore trusts.",
        "social_links": [
            "https://linkedin.com/in/victoria-kensington",
            "https://x.com/vkensington_fo",
        ],
    }

    resp = await client.post("/clients", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    client_id = UUID(data["id"])

    try:
        assert data["first_name"] == "victoria"
        assert data["last_name"] == "kensington"
        assert data["email"] == "victoria.kensington@mayfairfamilyoffice.co.uk"
        assert data["description"] == payload["description"]
        assert data["social_links"] == payload["social_links"]

        # Verify persisted in DB with is_synthetic=False
        async with AsyncSessionLocal() as session:
            db_client = await repo.get_client(session, client_id)
            assert db_client is not None
            assert db_client.is_synthetic is False
            assert db_client.email == "victoria.kensington@mayfairfamilyoffice.co.uk"

        # Verify immediately searchable via GET /search
        search_resp = await client.get(
            "/search", params={"q": "MayfairFamilyOffice"}
        )
        assert search_resp.status_code == 200
        matched_ids = [
            item["id"]
            for item in search_resp.json()
            if item["type"] == "client"
        ]
        assert str(client_id) in matched_ids
    finally:
        async with AsyncSessionLocal() as session:
            await repo.delete_client(session, client_id)


@pytest.mark.asyncio(loop_scope="module")
async def test_create_client_minimal_required_fields(
    client: AsyncClient,
) -> None:
    """Verify POST /clients succeeds when optional fields (description, social_links) are omitted."""
    resp = await client.post(
        "/clients",
        json={
            "first_name": "Henrik",
            "last_name": "Lindqvist",
            "email": "henrik.lindqvist@nordiccapital.se",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    client_id = UUID(data["id"])

    try:
        assert data["first_name"] == "henrik"
        assert data["last_name"] == "lindqvist"
        assert data["email"] == "henrik.lindqvist@nordiccapital.se"
        assert data["description"] is None
        assert data["social_links"] == []
    finally:
        async with AsyncSessionLocal() as session:
            await repo.delete_client(session, client_id)


@pytest.mark.asyncio(loop_scope="module")
async def test_create_client_duplicate_email_conflict(
    client: AsyncClient,
) -> None:
    """Verify POST /clients returns 409 Conflict for duplicate emails (including case/whitespace variations)."""
    resp = await client.post(
        "/clients",
        json={
            "first_name": "John",
            "last_name": "Doe",
            "email": "  JOHN.DOE@NEVISWEALTH.COM ",
        },
    )
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.asyncio(loop_scope="module")
async def test_create_client_validation_errors(client: AsyncClient) -> None:
    """Verify POST /clients returns 422 Unprocessable Entity for invalid email or empty required fields."""
    # Invalid email address
    resp_invalid_email = await client.post(
        "/clients",
        json={
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "not-a-valid-email",
        },
    )
    assert resp_invalid_email.status_code == 422

    # Whitespace-only first_name (strips to empty string -> violates min_length=1)
    resp_empty_name = await client.post(
        "/clients",
        json={
            "first_name": "   ",
            "last_name": "Smith",
            "email": "alice.smith@example.com",
        },
    )
    assert resp_empty_name.status_code == 422

    # Missing required fields
    resp_missing = await client.post("/clients", json={"first_name": "Alice"})
    assert resp_missing.status_code == 422


@pytest.mark.asyncio(loop_scope="module")
async def test_create_client_and_attach_document_end_to_end(
    client: AsyncClient,
) -> None:
    """Verify creating a new client followed by POST /clients/{id}/documents enriches, indexes, and searches both."""
    client_resp = await client.post(
        "/clients",
        json={
            "first_name": "Sebastian",
            "last_name": "Von Berg",
            "email": "sebastian.vonberg@alpine-private.ch",
            "description": "Swiss UHNW industrialist holding structured private credit and Geneva real estate.",
        },
    )
    assert client_resp.status_code == 201
    created_client = client_resp.json()
    client_id = UUID(created_client["id"])

    mock_analysis = DocumentAnalysisOutput(
        summary="2026 Geneva residential property deed and mortgage schedule for Sebastian Von Berg.",
        category="real_estate_and_property",
    )

    try:
        with patch(
            "repo.analyze_document",
            new=AsyncMock(return_value=mock_analysis),
        ):
            doc_resp = await client.post(
                f"/clients/{client_id}/documents",
                json={
                    "title": "Geneva Lakefront Villa Purchase & Land Registry Deed",
                    "content": (
                        "Official Canton of Geneva land registry title deed for Sebastian Von Berg "
                        "confirming ownership of Rue du Rhône 88, 1204 Genève, valued at CHF 18,500,000."
                    ),
                },
            )

        assert doc_resp.status_code == 201
        doc_data = doc_resp.json()
        doc_id = UUID(doc_data["id"])

        assert doc_data["client_id"] == str(client_id)
        assert doc_data["title"] == "Geneva Lakefront Villa Purchase & Land Registry Deed"
        assert doc_data["summary"] == mock_analysis.summary
        assert "category" not in doc_data
        assert "category_id" not in doc_data
        assert doc_data["created_at"] is not None

        # Verify internal category enrichment, contextual chunks, and 768-d embeddings in DB
        async with AsyncSessionLocal() as session:
            db_doc = await repo.get_document(session, doc_id)
            assert db_doc is not None
            assert db_doc.category == "real_estate_and_property"
            assert db_doc.category_id is not None

            chunks = (
                await session.execute(
                    select(DocumentChunkModel)
                    .where(DocumentChunkModel.document_id == doc_id)
                    .order_by(DocumentChunkModel.chunk_index)
                )
            ).scalars().all()
            assert len(chunks) >= 1
            assert "sebastian von berg" in chunks[0].chunk_text.lower()
            assert "Real Estate & Property Holdings" in chunks[0].chunk_text
            assert len(chunks[0].embedding) == 768

        # Verify cross-entity search ("Sebastian Von Berg Geneva villa") finds the new document
        search_resp = await client.get(
            "/search", params={"q": "Sebastian Von Berg Geneva villa"}
        )
        assert search_resp.status_code == 200
        search_docs = [
            item for item in search_resp.json() if item["type"] == "document"
        ]
        assert len(search_docs) >= 1
        assert search_docs[0]["id"] == str(doc_id)
        assert "category" not in search_docs[0]
        assert "category_id" not in search_docs[0]
    finally:
        # Deleting the client cascades to its documents and document_chunks
        async with AsyncSessionLocal() as session:
            await repo.delete_client(session, client_id)


@pytest.mark.asyncio(loop_scope="module")
async def test_create_document_not_found_and_validation_errors(
    client: AsyncClient,
) -> None:
    """Verify POST /clients/{id}/documents handles non-existent client (404), bad UUID (422), and empty fields (422)."""
    # Non-existent client UUID -> 404 Not Found
    missing_client_id = uuid4()
    resp_404 = await client.post(
        f"/clients/{missing_client_id}/documents",
        json={
            "title": "Orphaned Document",
            "content": "Document attached to a client ID that does not exist.",
        },
    )
    assert resp_404.status_code == 404
    assert str(missing_client_id) in resp_404.json()["detail"]

    # Invalid UUID in path -> 422 Unprocessable Entity
    resp_bad_uuid = await client.post(
        "/clients/not-a-valid-uuid/documents",
        json={
            "title": "Invalid Path UUID",
            "content": "Valid content.",
        },
    )
    assert resp_bad_uuid.status_code == 422

    # Empty title or content -> 422 Unprocessable Entity
    async with AsyncSessionLocal() as session:
        john = await repo.get_client_by_email(session, "john.doe@neviswealth.com")
        assert john is not None
        john_id = john.id

    resp_empty_title = await client.post(
        f"/clients/{john_id}/documents",
        json={"title": "", "content": "Non-empty content"},
    )
    assert resp_empty_title.status_code == 422

    resp_empty_content = await client.post(
        f"/clients/{john_id}/documents",
        json={"title": "Valid Title", "content": ""},
    )
    assert resp_empty_content.status_code == 422
