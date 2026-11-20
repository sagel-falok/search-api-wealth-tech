from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR, UUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5433/search_api",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


class DocumentCategoryModel(Base):
    __tablename__ = "document_categories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    category: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    examples: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
        default=list,
    )
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(replace(category, '_', ' '), '') || ' ' || coalesce(name, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(summary, '')), 'B')",
            persisted=True,
        ),
    )
    category_search_text: Mapped[str] = mapped_column(
        Text,
        Computed(
            "lower(coalesce(replace(category, '_', ' '), '') || ' ' || coalesce(name, '') || ' ' || coalesce(summary, ''))",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index(
            "ix_document_categories_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        Index(
            "ix_document_categories_trgm",
            "category_search_text",
            postgresql_using="gin",
            postgresql_ops={"category_search_text": "gin_trgm_ops"},
        ),
    )

    documents: Mapped[list[DocumentModel]] = relationship(
        "DocumentModel",
        back_populates="category_rel",
    )


class ClientModel(Base):
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    first_name: Mapped[str] = mapped_column(String(255), nullable=False)
    last_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        index=True,
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    social_links: Mapped[list[str] | None] = mapped_column(
        ARRAY(String),
        nullable=True,
        default=list,
    )
    is_synthetic: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
        index=True,
    )
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(first_name, '') || ' ' || coalesce(last_name, '') || ' ' || coalesce(email, '') || ' ' || regexp_replace(coalesce(email, ''), '[.@_+-]+', ' ', 'g')), 'A') || "
            "setweight(to_tsvector('english', coalesce(description, '')), 'B')",
            persisted=True,
        ),
    )
    identity_search_text: Mapped[str] = mapped_column(
        Text,
        Computed(
            "lower(coalesce(first_name, '') || ' ' || coalesce(last_name, '') || ' ' || coalesce(email, '') || ' ' || regexp_replace(coalesce(email, ''), '[\\s._@-]+', '', 'g') || ' ' || coalesce(first_name, '') || coalesce(last_name, ''))",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index(
            "ix_clients_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        Index(
            "ix_clients_identity_trgm",
            "identity_search_text",
            postgresql_using="gin",
            postgresql_ops={"identity_search_text": "gin_trgm_ops"},
        ),
        Index(
            "ix_clients_description_trgm",
            text("lower(coalesce(description, '')) gin_trgm_ops"),
            postgresql_using="gin",
        ),
    )

    documents: Mapped[list[DocumentModel]] = relationship(
        "DocumentModel",
        back_populates="client",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clients.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    category_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_categories.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=false(),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
            "setweight(to_tsvector('english', coalesce(summary, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(content, '')), 'C')",
            persisted=True,
        ),
    )

    __table_args__ = (
        Index(
            "ix_documents_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        Index(
            "ix_documents_title_summary_trgm",
            text("lower(coalesce(title, '') || ' ' || coalesce(summary, '')) gin_trgm_ops"),
            postgresql_using="gin",
        ),
    )

    client: Mapped[ClientModel] = relationship(
        "ClientModel",
        back_populates="documents",
    )
    category_rel: Mapped[DocumentCategoryModel | None] = relationship(
        "DocumentCategoryModel",
        back_populates="documents",
    )
    chunks: Mapped[list[DocumentChunkModel]] = relationship(
        "DocumentChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @property
    def category(self) -> str | None:
        if "category_rel" in self.__dict__ and self.category_rel is not None:
            return self.category_rel.category
        return None


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=False)

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    document: Mapped[DocumentModel] = relationship(
        "DocumentModel",
        back_populates="chunks",
    )


async def init_db() -> None:
    """Enable search extensions and create database tables if they do not exist."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        yield session
