from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

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
async def test_search_neviswealth_camelcase(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "NevisWealth"})
    assert response.status_code == 200
    data = response.json()
    emails = [item["email"] for item in data if item["type"] == "client"]
    assert "john.doe@neviswealth.com" in emails
    # Direct @neviswealth.com clients should rank above description-only mentions (r.whitaker1958@gmail.com)
    assert emails.index("john.doe@neviswealth.com") < emails.index("r.whitaker1958@gmail.com")


@pytest.mark.asyncio(loop_scope="module")
async def test_search_nevis_wealth_spaced(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "Nevis wealth"})
    assert response.status_code == 200
    data = response.json()
    emails = [item["email"] for item in data if item["type"] == "client"]
    assert "john.doe@neviswealth.com" in emails


@pytest.mark.asyncio(loop_scope="module")
async def test_search_nefis_wealth_typo_and_spaced(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "nefis wealth"})
    assert response.status_code == 200
    data = response.json()
    emails = [item["email"] for item in data if item["type"] == "client"]
    assert "john.doe@neviswealth.com" in emails
    assert emails[0] == "john.doe@neviswealth.com"


@pytest.mark.asyncio(loop_scope="module")
async def test_search_john_ranks_exact_first_name_first(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "john"})
    assert response.status_code == 200
    data = response.json()
    client_results = [item for item in data if item["type"] == "client"]
    assert len(client_results) >= 1
    assert client_results[0]["email"] == "john.doe@neviswealth.com"
    assert client_results[0]["first_name"] == "john"
    assert client_results[0]["last_name"] == "doe"


@pytest.mark.asyncio(loop_scope="module")
async def test_search_by_description_terms(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "estate planning"})
    assert response.status_code == 200
    data = response.json()
    emails = [item["email"] for item in data if item["type"] == "client"]
    assert "john.doe@neviswealth.com" in emails


@pytest.mark.asyncio(loop_scope="module")
async def test_search_accent_insensitive_and_fuzzy(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "Chloé"})
    assert response.status_code == 200
    emails = [item["email"] for item in response.json() if item["type"] == "client"]
    assert "chloe.dubois@vanguardia-bio.com" in emails

    response_unaccented = await client.get("/search", params={"q": "chloe"})
    assert response_unaccented.status_code == 200
    emails_unaccented = [
        item["email"] for item in response_unaccented.json() if item["type"] == "client"
    ]
    assert "chloe.dubois@vanguardia-bio.com" in emails_unaccented


@pytest.mark.asyncio(loop_scope="module")
async def test_search_empty_or_unmatched_query(client: AsyncClient) -> None:
    response = await client.get("/search", params={"q": "zzzznonexistentclient99999"})
    assert response.status_code == 200
    assert response.json() == []

    response_missing = await client.get("/search", params={"q": ""})
    assert response_missing.status_code == 422
