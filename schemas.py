from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, EmailStr, Field

StrUUID = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, UUID) else v)]
NormalizedStr = Annotated[
    str,
    BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
]
NormalizedEmail = Annotated[
    EmailStr,
    BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
    AfterValidator(lambda v: str(v).strip().lower()),
]


class DocumentCategoryCreate(BaseModel):
    category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique machine-friendly category slug",
        examples=["address_proof"],
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable category name",
        examples=["Proof of Address & Residency"],
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="Concise summary of what this document category covers",
        examples=[
            "Documents confirming a client's current residential address and domicile."
        ],
    )
    examples: list[str] | None = Field(
        default=None,
        description="Representative document types belonging to this category",
        examples=[["Utility Bill", "Residential Lease Agreement"]],
    )


class DocumentCategory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrUUID = Field(
        ...,
        description="Unique category identifier",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    category: str = Field(..., examples=["address_proof"])
    name: str = Field(..., examples=["Proof of Address & Residency"])
    summary: str = Field(
        ...,
        examples=[
            "Documents confirming a client's current residential address and domicile."
        ],
    )
    examples: list[str] | None = Field(
        default=None,
        examples=[["Utility Bill", "Residential Lease Agreement"]],
    )


class ClientCreate(BaseModel):
    first_name: NormalizedStr = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Client's first name",
        examples=["john"],
    )
    last_name: NormalizedStr = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Client's last name",
        examples=["doe"],
    )
    email: NormalizedEmail = Field(
        ...,
        description="Client's unique email address",
        examples=["john.doe@neviswealth.com"],
    )
    description: str | None = Field(
        default=None,
        description="Optional background or notes about the client",
        examples=["High-net-worth tech executive looking for estate planning"],
    )
    social_links: list[str] | None = Field(
        default=None,
        description="List of client social profile URLs",
        examples=[["https://linkedin.com/in/johndoe"]],
    )


class ClientUpdate(BaseModel):
    first_name: NormalizedStr | None = Field(default=None, min_length=1, max_length=255)
    last_name: NormalizedStr | None = Field(default=None, min_length=1, max_length=255)
    email: NormalizedEmail | None = Field(default=None)
    description: str | None = Field(default=None)
    social_links: list[str] | None = Field(default=None)


class Client(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrUUID = Field(
        ...,
        description="Unique client identifier",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    first_name: str = Field(..., examples=["John"])
    last_name: str = Field(..., examples=["Doe"])
    email: EmailStr = Field(..., examples=["john.doe@neviswealth.com"])
    description: str | None = Field(
        default=None,
        examples=["High-net-worth tech executive looking for estate planning"],
    )
    social_links: list[str] | None = Field(
        default=None,
        examples=[["https://linkedin.com/in/johndoe"]],
    )


class ClientSearchResult(Client):
    type: Literal["client"] = Field(
        default="client",
        description="Search result entity type",
        examples=["client"],
    )


class DocumentCreate(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Document title",
        examples=["Utility Bill - October 2026"],
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Text content of the document",
        examples=["Electricity utility bill for 123 Nevis Way serving as proof of address."],
    )


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    content: str | None = Field(default=None, min_length=1)
    category_id: UUID | None = Field(default=None)
    summary: str | None = Field(default=None)


class Document(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: StrUUID = Field(
        ...,
        description="Unique document identifier",
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
    )
    client_id: StrUUID = Field(
        ...,
        description="ID of the client who owns this document",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    category_id: StrUUID | None = Field(
        default=None,
        description="Optional ID of the assigned document category",
        examples=["9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"],
    )
    title: str = Field(..., examples=["Utility Bill - October 2026"])
    content: str = Field(
        ...,
        examples=["Electricity utility bill for 123 Nevis Way serving as proof of address."],
    )
    summary: str | None = Field(
        default=None,
        description="Optional concise summary of the document content",
        examples=["October 2026 electricity bill confirming residential address at 123 Nevis Way."],
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the document was created",
        examples=["2026-11-19T12:00:00Z"],
    )


class DocumentSearchResult(Document):
    type: Literal["document"] = Field(
        default="document",
        description="Search result entity type",
        examples=["document"],
    )
    category: str | None = Field(
        default=None,
        description="Machine-friendly slug of the assigned document category",
        examples=["address_proof"],
    )


class SyntheticDocumentSeed(BaseModel):
    client_email: NormalizedEmail = Field(
        ...,
        description="Email of the client who owns this synthetic document",
    )
    category: str | None = Field(
        default=None,
        description="Category slug matching document_categories.category",
    )
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    summary: str | None = Field(default=None)


SearchResultItem = dict[str, Any]
