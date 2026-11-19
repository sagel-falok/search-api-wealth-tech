from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import TypeAdapter

import repo
from db import AsyncSessionLocal, engine, init_db
from schemas import (
    ClientCreate,
    DocumentCategoryCreate,
    SyntheticDocumentSeed,
)

DATA_DIR = Path(__file__).parent / "data"
DOCUMENT_CATEGORIES_PATH = DATA_DIR / "document_categories.json"
SYNTHETIC_CLIENTS_PATH = DATA_DIR / "synthetic_clients.json"
SYNTHETIC_DOCUMENTS_PATH = DATA_DIR / "synthetic_documents.json"


async def seed_all(
    categories_path: Path = DOCUMENT_CATEGORIES_PATH,
    clients_path: Path = SYNTHETIC_CLIENTS_PATH,
    documents_path: Path = SYNTHETIC_DOCUMENTS_PATH,
) -> tuple[int, int, int]:
    await init_db()

    raw_categories = json.loads(categories_path.read_text(encoding="utf-8"))
    categories_to_seed = TypeAdapter(list[DocumentCategoryCreate]).validate_python(
        raw_categories
    )

    raw_clients = json.loads(clients_path.read_text(encoding="utf-8"))
    clients_to_seed = TypeAdapter(list[ClientCreate]).validate_python(
        raw_clients
    )

    raw_documents = json.loads(documents_path.read_text(encoding="utf-8"))
    documents_to_seed = TypeAdapter(list[SyntheticDocumentSeed]).validate_python(
        raw_documents
    )

    async with AsyncSessionLocal() as session:
        synced_categories = await repo.sync_document_categories(
            session, categories_to_seed
        )
        synced_clients = await repo.sync_synthetic_clients(
            session, clients_to_seed
        )
        synced_documents = await repo.sync_synthetic_documents(
            session, documents_to_seed
        )
        return (
            len(synced_categories),
            len(synced_clients),
            len(synced_documents),
        )


async def main() -> None:
    try:
        cat_count, client_count, doc_count = await seed_all()
        print(
            f"Successfully synced:\n"
            f"  - {cat_count} document categories\n"
            f"  - {client_count} synthetic clients\n"
            f"  - {doc_count} synthetic documents (linked via client email & category slug)"
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
