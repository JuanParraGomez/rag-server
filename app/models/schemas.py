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
    semantic_score: float | None = None
    rerank_score: float | None = None
    text: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class QueryWithFiltersRequest(BaseModel):
    question: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=30)
    include_context: bool = Field(default=True)
    memory_limit: int = Field(default=10, ge=1, le=100)


class DocumentContextItem(BaseModel):
    document_id: str
    title: str | None = None
    project_id: str | None = None
    status: str | None = None
    current_version: int | None = None
    versions: list[dict[str, Any]] = Field(default_factory=list)
    memories: list[dict[str, Any]] = Field(default_factory=list)


class DocumentContextResponse(QueryResponse):
    documents: list[DocumentContextItem] = Field(default_factory=list)


class CanonicalMemoryPayload(BaseModel):
    memory_type: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    created_by: str = Field(default="agent")


class CanonicalIngestRequest(BaseModel):
    content_type: str = Field(default="text", pattern="^(text|file)$")
    title: str | None = None
    text: str | None = None
    file_name: str | None = None
    file_base64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    memory: CanonicalMemoryPayload | None = None
    canonical_document_id: str | None = None

    @field_validator("metadata")
    @classmethod
    def require_tenant_id(cls, value: dict[str, Any]) -> dict[str, Any]:
        tenant_id = value.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("metadata.tenant_id is required and must be a non-empty string")
        return value


class CanonicalIngestResponse(BaseModel):
    status: str
    request_id: str
    document_id: str
    document_family_id: str
    version: int
    is_latest: bool
    canonical_tags: list[str] = Field(default_factory=list)
    rag: UploadResponse
    memory: dict[str, Any] | None = None


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
