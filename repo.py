from __future__ import annotations

import re
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Float, Text, and_, case, cast, delete, func, literal, or_, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import (
    ClientModel,
    DocumentCategoryModel,
    DocumentChunkModel,
    DocumentModel,
)
from embeddings import build_document_chunks, embed_passages, embed_query
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


async def search_clients_with_scores(
    session: AsyncSession,
    query: str,
    limit: int = 20,
) -> list[tuple[ClientModel, float]]:
    """Search clients across name, email, and description and return (ClientModel, relevance_score) tuples."""
    q_lower = query.strip().lower()
    if not q_lower:
        return []

    q_compact = re.sub(r"[\s._@-]+", "", q_lower)
    has_compact = len(q_compact) >= 3 and q_compact != q_lower

    # Normalize accented characters on input query
    q_norm = func.unaccent(q_lower)
    q_compact_norm = func.unaccent(q_compact)

    desc_lower = func.lower(func.coalesce(ClientModel.description, ""))
    ts_query = func.websearch_to_tsquery("english", q_norm)

    # Index-backed filter predicates (uses ix_clients_search_vector, ix_clients_identity_trgm, ix_clients_description_trgm)
    fts_match = ClientModel.search_vector.op("@@")(ts_query)
    identity_substr_match = ClientModel.identity_search_text.ilike(f"%{q_lower}%")
    compact_substr_match = (
        ClientModel.identity_search_text.ilike(f"%{q_compact}%")
        if len(q_compact) >= 3
        else literal(False)
    )
    desc_substr_match = desc_lower.ilike(f"%{q_lower}%")

    identity_trgm_match = q_norm.op("<%")(ClientModel.identity_search_text)
    compact_trgm_match = (
        q_compact_norm.op("<%")(ClientModel.identity_search_text)
        if has_compact
        else literal(False)
    )
    desc_trgm_match = q_norm.op("<%")(desc_lower)

    # Exact identity match condition
    full_name_lower = func.lower(ClientModel.first_name) + literal(" ") + func.lower(ClientModel.last_name)
    exact_identity_match = or_(
        func.unaccent(func.lower(ClientModel.first_name)) == q_norm,
        func.unaccent(func.lower(ClientModel.last_name)) == q_norm,
        func.unaccent(full_name_lower) == q_norm,
        func.lower(ClientModel.email) == q_lower,
    )

    # Trigram similarity scores for ranking
    identity_sim = func.word_similarity(q_norm, ClientModel.identity_search_text)
    if len(q_compact) >= 3:
        identity_sim = func.greatest(
            identity_sim,
            func.word_similarity(q_compact_norm, ClientModel.identity_search_text),
        )
    desc_sim = func.word_similarity(q_norm, desc_lower)

    # Column-weighted relevance score (Tier A: identity > Tier B: description)
    score = (
        case((exact_identity_match, 4.0), else_=0.0)
        + case((or_(identity_substr_match, compact_substr_match), 2.0), else_=0.0)
        + cast(identity_sim, Float) * 1.5
        + case((fts_match, cast(func.ts_rank_cd(ClientModel.search_vector, ts_query, 33), Float) * 1.0), else_=0.0)
        + case((desc_substr_match, 0.6), else_=0.0)
        + cast(desc_sim, Float) * 0.5
    ).label("search_score")

    stmt = (
        select(ClientModel, score)
        .where(
            or_(
                fts_match,
                identity_substr_match,
                compact_substr_match,
                desc_substr_match,
                identity_trgm_match,
                compact_trgm_match,
                desc_trgm_match,
            )
        )
        .order_by(
            score.desc(),
            ClientModel.last_name.asc(),
            ClientModel.first_name.asc(),
        )
        .limit(limit)
    )

    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result.all()]


async def search_clients(
    session: AsyncSession,
    query: str,
    limit: int = 20,
) -> Sequence[ClientModel]:
    """Search clients across name, email, and description using GIN-indexed FTS, substring, and trigram similarity."""
    rows = await search_clients_with_scores(session, query, limit=limit)
    return [client for client, _ in rows]



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
    """Delete all existing synthetic documents and insert synthetic documents with contextual chunks and embeddings."""
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
    doc_chunk_specs: list[tuple[DocumentModel, int, str]] = []

    for item in documents_data:
        email_key = str(item.client_email).lower()
        client = clients_by_email.get(email_key)
        if client is None:
            raise ValueError(
                f"Cannot seed document '{item.title}': client with email '{item.client_email}' not found."
            )

        category_id: UUID | None = None
        category_model: DocumentCategoryModel | None = None
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

        chunk_texts = build_document_chunks(
            title=item.title,
            content=item.content,
            summary=item.summary,
            client_first_name=client.first_name,
            client_last_name=client.last_name,
            client_email=client.email,
            category_slug=category_model.category if category_model else None,
            category_name=category_model.name if category_model else None,
            category_summary=category_model.summary if category_model else None,
            category_examples=category_model.examples if category_model else None,
        )
        for idx, chunk_text in enumerate(chunk_texts):
            doc_chunk_specs.append((doc, idx, chunk_text))

    await session.flush()

    if doc_chunk_specs:
        embeddings = embed_passages([spec[2] for spec in doc_chunk_specs])
        for (doc, idx, chunk_text), embedding in zip(
            doc_chunk_specs, embeddings, strict=True
        ):
            session.add(
                DocumentChunkModel(
                    document_id=doc.id,
                    chunk_index=idx,
                    chunk_text=chunk_text,
                    embedding=embedding,
                )
            )

    await session.commit()
    for doc in created:
        await session.refresh(doc)
    return created


async def search_documents_with_scores(
    session: AsyncSession,
    query: str,
    limit: int = 20,
    semantic_threshold: float = 0.60,
) -> list[tuple[DocumentModel, float]]:
    """
    Hybrid document search combining:
    1. Semantic vector similarity via pgvector (FastEmbed BAAI/bge-base-en-v1.5) over document_chunks
    2. Lexical full-text search (ts_rank_cd with BM25-like length & saturation normalization)
    3. Typo-tolerant trigram similarity (pg_trgm) across document title/summary and category taxonomy
    4. Cross-entity Client + Document/Category synergy boosting
    """
    q_lower = query.strip().lower()
    if not q_lower:
        return []

    q_norm = func.unaccent(q_lower)
    q_vec = embed_query(q_lower)

    best_chunk = (
        select(
            DocumentChunkModel.document_id.label("document_id"),
            func.max(
                1.0 - DocumentChunkModel.embedding.cosine_distance(q_vec)
            ).label("semantic_sim"),
        )
        .group_by(DocumentChunkModel.document_id)
        .subquery("best_chunk")
    )

    semantic_sim = cast(func.coalesce(best_chunk.c.semantic_sim, 0.0), Float)

    ts_query = func.websearch_to_tsquery("english", q_norm)
    ts_or_query = cast(
        func.replace(
            cast(func.plainto_tsquery("english", q_norm), Text),
            " & ",
            " | ",
        ),
        TSQUERY,
    )

    empty_tsvector = func.to_tsvector("english", literal(""))
    cat_search_vector = func.coalesce(
        DocumentCategoryModel.search_vector, empty_tsvector
    )
    doc_cat_search_vector = DocumentModel.search_vector.op("||")(cat_search_vector)
    combined_search_vector = doc_cat_search_vector.op("||")(
        ClientModel.search_vector
    )

    title_lower = func.lower(func.coalesce(DocumentModel.title, ""))
    summary_lower = func.lower(func.coalesce(DocumentModel.summary, ""))
    content_lower = func.lower(func.coalesce(DocumentModel.content, ""))
    title_summary_lower = func.lower(
        func.coalesce(DocumentModel.title, "")
        + literal(" ")
        + func.coalesce(DocumentModel.summary, "")
    )
    cat_search_lower = func.coalesce(
        DocumentCategoryModel.category_search_text, literal("")
    )

    # Candidate Gate Predicates
    semantic_match = semantic_sim >= semantic_threshold
    doc_cat_fts_match = doc_cat_search_vector.op("@@", is_comparison=True)(
        ts_query
    )
    doc_cat_partial_fts_match = doc_cat_search_vector.op(
        "@@", is_comparison=True
    )(ts_or_query)
    combined_fts_match = and_(
        combined_search_vector.op("@@", is_comparison=True)(ts_query),
        doc_cat_partial_fts_match,
    )

    title_substr_match = title_lower.ilike(f"%{q_lower}%")
    summary_substr_match = summary_lower.ilike(f"%{q_lower}%")
    content_substr_match = content_lower.ilike(f"%{q_lower}%")

    title_summary_trgm_match = q_norm.op("<%", is_comparison=True)(
        title_summary_lower
    )
    category_trgm_match = q_norm.op("<%", is_comparison=True)(cat_search_lower)

    # Component Similarity Scores
    title_sim = cast(func.word_similarity(q_norm, title_lower), Float)
    summary_sim = cast(func.word_similarity(q_norm, summary_lower), Float)
    cat_sim = cast(func.word_similarity(q_norm, cat_search_lower), Float)

    # Normalization 33 = 1 (divides by 1 + log(doc length)) | 32 (rank / (rank + 1))
    doc_cat_fts_rank = cast(
        func.ts_rank_cd(doc_cat_search_vector, ts_query, 33), Float
    )
    doc_partial_fts_rank = cast(
        func.ts_rank_cd(DocumentModel.search_vector, ts_or_query, 33), Float
    )
    combined_fts_rank = cast(
        func.ts_rank_cd(combined_search_vector, ts_query, 33), Float
    )
    client_fts_rank = cast(
        func.ts_rank_cd(ClientModel.search_vector, ts_or_query, 33), Float
    )

    client_has_match = or_(
        ClientModel.search_vector.op("@@", is_comparison=True)(ts_or_query),
        q_norm.op("<%", is_comparison=True)(ClientModel.identity_search_text),
    )
    doc_has_match = or_(
        doc_cat_partial_fts_match,
        semantic_match,
        title_summary_trgm_match,
        category_trgm_match,
        title_substr_match,
        summary_substr_match,
        content_substr_match,
    )
    cross_entity_synergy = and_(client_has_match, doc_has_match)

    # Weighted Hybrid Relevance Score
    score = (
        # 1. Semantic vector similarity score (rescaled above 0.50 baseline)
        func.greatest(0.0, semantic_sim - 0.50) * 4.0
        + case((semantic_match, 0.8), else_=0.0)
        # 2. Lexical FTS / BM25-like score across Document + Category
        + case((doc_cat_fts_match, 1.2 + doc_cat_fts_rank * 2.5), else_=0.0)
        + case((doc_cat_partial_fts_match, doc_partial_fts_rank * 1.2), else_=0.0)
        # 3. Direct substring & trigram fuzzy matches (Title > Category > Summary/Content)
        + case((title_substr_match, 1.5), else_=0.0)
        + case((or_(summary_substr_match, content_substr_match), 0.7), else_=0.0)
        + title_sim * 1.2
        + summary_sim * 0.5
        + case((category_trgm_match, 0.8 + cat_sim * 1.5), else_=cat_sim * 0.6)
        # 4. Cross-Entity (Client + Document/Category) Synergy Boost
        + case(
            (
                combined_fts_match,
                1.5 + combined_fts_rank * 2.0 + client_fts_rank * 1.5,
            ),
            (cross_entity_synergy, 0.8 + client_fts_rank * 1.5),
            else_=0.0,
        )
    ).label("search_score")

    stmt = (
        select(DocumentModel, score)
        .outerjoin(best_chunk, best_chunk.c.document_id == DocumentModel.id)
        .join(ClientModel, ClientModel.id == DocumentModel.client_id)
        .outerjoin(
            DocumentCategoryModel,
            DocumentCategoryModel.id == DocumentModel.category_id,
        )
        .options(selectinload(DocumentModel.category_rel))
        .where(
            or_(
                semantic_match,
                doc_cat_fts_match,
                combined_fts_match,
                title_substr_match,
                summary_substr_match,
                content_substr_match,
                title_summary_trgm_match,
                category_trgm_match,
            )
        )
        .order_by(
            score.desc(),
            DocumentModel.created_at.desc(),
            DocumentModel.title.asc(),
        )
        .limit(limit)
    )

    result = await session.execute(stmt)
    return [(row[0], float(row[1])) for row in result.all()]


async def search_documents(
    session: AsyncSession,
    query: str,
    limit: int = 20,
) -> Sequence[DocumentModel]:
    """Search documents using hybrid semantic vector similarity, FTS, trigram similarity, and client/category context."""
    rows = await search_documents_with_scores(session, query, limit=limit)
    return [doc for doc, _ in rows]


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
