from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from db import ClientModel, DocumentCategoryModel, DocumentModel
from schemas import (
    ClientCreate,
    ClientUpdate,
    DocumentCategoryCreate,
    DocumentCreate,
    DocumentUpdate,
    SyntheticDocumentSeed,
)


async def sync_document_categories(
    session: AsyncSession,
    categories_data: Sequence[DocumentCategoryCreate],
) -> list[DocumentCategoryModel]:
    """Synchronize document categories from seed list (upsert by category slug and remove stale slugs)."""
    existing_result = await session.execute(select(DocumentCategoryModel))
    existing_by_slug = {
        cat.category: cat for cat in existing_result.scalars().all()
    }
    incoming_slugs = {item.category for item in categories_data}

    for slug, cat_model in list(existing_by_slug.items()):
        if slug not in incoming_slugs:
            await session.delete(cat_model)

    synced: list[DocumentCategoryModel] = []
    for item in categories_data:
        cat_model = existing_by_slug.get(item.category)
        if cat_model is None:
            cat_model = DocumentCategoryModel(
                category=item.category,
                name=item.name,
                summary=item.summary,
                examples=item.examples,
            )
            session.add(cat_model)
        else:
            cat_model.name = item.name
            cat_model.summary = item.summary
            cat_model.examples = item.examples
        synced.append(cat_model)

    await session.commit()
    for cat_model in synced:
        await session.refresh(cat_model)
    return synced


async def get_document_category_by_slug(
    session: AsyncSession,
    category_slug: str,
) -> DocumentCategoryModel | None:
    result = await session.execute(
        select(DocumentCategoryModel).where(
            DocumentCategoryModel.category == category_slug
        )
    )
    return result.scalar_one_or_none()


async def list_document_categories(
    session: AsyncSession,
) -> Sequence[DocumentCategoryModel]:
    result = await session.execute(
        select(DocumentCategoryModel).order_by(DocumentCategoryModel.category)
    )
    return result.scalars().all()


async def create_client(
    session: AsyncSession,
    data: ClientCreate,
    *,
    is_synthetic: bool = False,
) -> ClientModel:
    client = ClientModel(
        first_name=data.first_name.strip().lower(),
        last_name=data.last_name.strip().lower(),
        email=str(data.email).strip().lower(),
        description=data.description,
        social_links=data.social_links,
        is_synthetic=is_synthetic,
    )
    session.add(client)
    await session.commit()
    await session.refresh(client)
    return client


async def sync_synthetic_clients(
    session: AsyncSession,
    clients_data: Sequence[ClientCreate],
) -> list[ClientModel]:
    """Delete all existing synthetic clients (and their cascaded documents) and insert the provided synthetic clients."""
    await session.execute(
        delete(ClientModel).where(ClientModel.is_synthetic.is_(True))
    )
    created: list[ClientModel] = []
    for item in clients_data:
        client = ClientModel(
            first_name=item.first_name.strip().lower(),
            last_name=item.last_name.strip().lower(),
            email=str(item.email).strip().lower(),
            description=item.description,
            social_links=item.social_links,
            is_synthetic=True,
        )
        session.add(client)
        created.append(client)

    await session.commit()
    for client in created:
        await session.refresh(client)
    return created


async def get_client(
    session: AsyncSession,
    client_id: UUID,
) -> ClientModel | None:
    result = await session.execute(
        select(ClientModel).where(ClientModel.id == client_id)
    )
    return result.scalar_one_or_none()


async def get_client_by_email(
    session: AsyncSession,
    email: str,
) -> ClientModel | None:
    normalized_email = email.strip().lower()
    result = await session.execute(
        select(ClientModel).where(ClientModel.email == normalized_email)
    )
    return result.scalar_one_or_none()


async def list_clients(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 100,
) -> Sequence[ClientModel]:
    result = await session.execute(
        select(ClientModel).offset(skip).limit(limit)
    )
    return result.scalars().all()


async def update_client(
    session: AsyncSession,
    client_id: UUID,
    data: ClientUpdate,
) -> ClientModel | None:
    client = await get_client(session, client_id)
    if client is None:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for key in ("first_name", "last_name", "email"):
        if key in update_data and update_data[key] is not None:
            update_data[key] = str(update_data[key]).strip().lower()

    for field, value in update_data.items():
        setattr(client, field, value)

    await session.commit()
    await session.refresh(client)
    return client


async def delete_client(
    session: AsyncSession,
    client_id: UUID,
) -> bool:
    client = await get_client(session, client_id)
    if client is None:
        return False

    await session.delete(client)
    await session.commit()
    return True


async def create_document(
    session: AsyncSession,
    client_id: UUID,
    data: DocumentCreate,
    *,
    is_synthetic: bool = False,
) -> DocumentModel:
    document = DocumentModel(
        client_id=client_id,
        title=data.title,
        content=data.content,
        is_synthetic=is_synthetic,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def sync_synthetic_documents(
    session: AsyncSession,
    documents_data: Sequence[SyntheticDocumentSeed],
) -> list[DocumentModel]:
    """Delete all existing synthetic documents and insert synthetic documents referencing clients by email and categories by slug."""
    await session.execute(
        delete(DocumentModel).where(DocumentModel.is_synthetic.is_(True))
    )

    clients_result = await session.execute(select(ClientModel))
    clients_by_email = {
        c.email.lower(): c for c in clients_result.scalars().all()
    }

    categories_result = await session.execute(select(DocumentCategoryModel))
    categories_by_slug = {
        cat.category: cat for cat in categories_result.scalars().all()
    }

    created: list[DocumentModel] = []
    for item in documents_data:
        email_key = str(item.client_email).lower()
        client = clients_by_email.get(email_key)
        if client is None:
            raise ValueError(
                f"Cannot seed document '{item.title}': client with email '{item.client_email}' not found."
            )

        category_id: UUID | None = None
        if item.category is not None:
            category_model = categories_by_slug.get(item.category)
            if category_model is None:
                raise ValueError(
                    f"Cannot seed document '{item.title}': category '{item.category}' not found."
                )
            category_id = category_model.id

        doc = DocumentModel(
            client_id=client.id,
            category_id=category_id,
            title=item.title,
            content=item.content,
            summary=item.summary,
            is_synthetic=True,
        )
        session.add(doc)
        created.append(doc)

    await session.commit()
    for doc in created:
        await session.refresh(doc)
    return created


async def get_document(
    session: AsyncSession,
    document_id: UUID,
) -> DocumentModel | None:
    result = await session.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    return result.scalar_one_or_none()


async def list_client_documents(
    session: AsyncSession,
    client_id: UUID,
    skip: int = 0,
    limit: int = 100,
) -> Sequence[DocumentModel]:
    result = await session.execute(
        select(DocumentModel)
        .where(DocumentModel.client_id == client_id)
        .order_by(DocumentModel.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def update_document(
    session: AsyncSession,
    document_id: UUID,
    data: DocumentUpdate,
) -> DocumentModel | None:
    document = await get_document(session, document_id)
    if document is None:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(document, field, value)

    await session.commit()
    await session.refresh(document)
    return document


async def delete_document(
    session: AsyncSession,
    document_id: UUID,
) -> bool:
    document = await get_document(session, document_id)
    if document is None:
        return False

    await session.delete(document)
    await session.commit()
    return True
