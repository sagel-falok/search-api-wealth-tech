from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Path, Query, status
from fastapi_mcp import FastApiMCP
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import repo
from db import get_db, init_db
from schemas import (
    Client,
    ClientCreate,
    ClientSearchResult,
    Document,
    DocumentCreate,
    DocumentSearchResult,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await init_db()
    except Exception as exc:
        logger.warning("Database initialization skipped at startup: %s", exc)
    yield


app = FastAPI(
    title="API",
    version="1.0.0",
    description="WealthTech Search API across clients and documents.",
    lifespan=lifespan,
)


@app.post(
    "/clients",
    operation_id="create_client",
    response_model=Client,
    status_code=status.HTTP_201_CREATED,
    tags=["Clients"],
    summary="Create a new client",
    description="Create a new client record with contact details, optional description, and social links.",
    responses={
        status.HTTP_201_CREATED: {
            "description": "Client successfully created",
            "model": Client,
        },
        status.HTTP_409_CONFLICT: {
            "description": "A client with this email address already exists",
        },
    },
)
async def create_client_endpoint(
    payload: ClientCreate,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Client:
    existing = await repo.get_client_by_email(session, str(payload.email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client with email '{payload.email}' already exists.",
        )

    try:
        client = await repo.create_client(session, payload)
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client with email '{payload.email}' already exists.",
        ) from exc

    return Client.model_validate(client)


@app.post(
    "/clients/{id}/documents",
    operation_id="create_client_document",
    response_model=Document,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Create a document for a client",
    description="Attach a new text document to an existing client by client ID.",
    responses={
        status.HTTP_201_CREATED: {
            "description": "Document successfully created",
            "model": Document,
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Client not found",
        },
    },
)
async def create_client_document_endpoint(
    id: Annotated[
        UUID,
        Path(
            description="Unique identifier of the client",
            examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        ),
    ],
    payload: DocumentCreate,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> Document:
    client = await repo.get_client(session, id)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client with id '{id}' not found.",
        )

    document = await repo.create_document(session, id, payload)
    return Document.model_validate(document)


@app.get(
    "/search",
    operation_id="search",
    response_model=list[ClientSearchResult | DocumentSearchResult],
    status_code=status.HTTP_200_OK,
    tags=["Search"],
    summary="Search across clients and documents",
    description=(
        "Search across clients (matching email, name, or description) and documents "
        "(matching similar terms in document content)."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "List of matching clients and documents",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "first_name": "john",
                            "last_name": "doe",
                            "email": "john.doe@neviswealth.com",
                            "description": "High-net-worth tech executive looking for estate planning",
                            "social_links": ["https://linkedin.com/in/johndoe"],
                            "type": "client",
                        },
                        {
                            "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                            "client_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "title": "Utility Bill - October 2026",
                            "content": "Electricity utility bill for 123 Nevis Way serving as proof of address.",
                            "summary": "October 2026 electricity bill confirming residential address at 123 Nevis Way.",
                            "created_at": "2026-11-19T12:00:00Z",
                            "type": "document",
                        },
                    ]
                }
            },
        }
    },
)
async def search_endpoint(
    q: Annotated[
        str,
        Query(
            ...,
            min_length=1,
            description="Search query string across clients and documents",
            examples=["NevisWealth", "address proof"],
        ),
    ],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> list[dict[str, Any]]:
    client_rows = await repo.search_clients_with_scores(session, q)
    document_rows = await repo.search_documents_with_scores(session, q)

    combined: list[tuple[float, dict[str, Any]]] = [
        (
            score,
            ClientSearchResult.model_validate(client).model_dump(),
        )
        for client, score in client_rows
    ] + [
        (
            score,
            DocumentSearchResult.model_validate(doc).model_dump(
                exclude={"category", "category_id"}
            ),
        )
        for doc, score in document_rows
    ]

    combined.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in combined]


mcp = FastApiMCP(
    app,
    name="Nevis WealthTech Search MCP",
    description="Search and manage WealthTech clients and client documents.",
)
mcp.mount_http()

