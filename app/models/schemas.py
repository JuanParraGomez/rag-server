from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class UploadTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str | None = None
    title: str | None = None

    @field_validator("metadata")
    @classmethod
    def require_tenant_id(cls, value: dict[str, Any]) -> dict[str, Any]:
        tenant_id = value.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("metadata.tenant_id is required and must be a non-empty string")
        return value


class UploadResponse(BaseModel):
    document_id: str
    tenant_id: str
    chunks_indexed: int
    metadata: dict[str, Any]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=30)


class SourceChunk(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class DocumentSummary(BaseModel):
    document_id: str
    tenant_id: str
    title: str | None = None
    source_type: str | None = None
    chunk_count: int
    metadata: dict[str, Any]
    first_seen_at: datetime | None = None


class DocumentListResponse(BaseModel):
    tenant_id: str
    total: int
    items: list[DocumentSummary]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    tenant_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    app: str
    collection: str
