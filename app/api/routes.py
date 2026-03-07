from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.dependencies import (
    get_canonical_ingestion_service,
    get_document_service,
    get_query_service,
    get_vector_store_adapter,
)
from app.models.schemas import (
    CanonicalIngestRequest,
    CanonicalIngestResponse,
    DeleteDocumentResponse,
    DocumentContextResponse,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    QueryWithFiltersRequest,
    UploadResponse,
    UploadTextRequest,
)
from app.services.canonical_ingestion_service import CanonicalIngestionService
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.vector_store.base import VectorStoreAdapter

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(vector_store: VectorStoreAdapter = Depends(get_vector_store_adapter)) -> HealthResponse:
    return HealthResponse(status="ok", app="rag-server", collection=vector_store.collection_name)  # type: ignore[attr-defined]


@router.post("/upload-file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    document_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    return await document_service.ingest_file(file=file, metadata_json=metadata)


@router.post("/upload-text", response_model=UploadResponse)
def upload_text(
    payload: UploadTextRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> UploadResponse:
    return document_service.ingest_text(
        text=payload.text,
        metadata=payload.metadata,
        title=payload.title,
        document_id=payload.document_id,
    )


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return query_service.query(
        question=payload.question,
        tenant_id=payload.tenant_id,
        filters=payload.filters,
        top_k=payload.top_k,
    )


@router.post("/query/with-filters", response_model=DocumentContextResponse)
def query_with_filters(
    payload: QueryWithFiltersRequest,
    canonical_service: CanonicalIngestionService = Depends(get_canonical_ingestion_service),
    query_service: QueryService = Depends(get_query_service),
) -> DocumentContextResponse:
    if payload.include_context:
        return canonical_service.get_document_context(
            tenant_id=payload.tenant_id,
            question=payload.question,
            filters=payload.filters,
            top_k=payload.top_k,
            memory_limit=payload.memory_limit,
        )
    query_result = query_service.query(
        question=payload.question,
        tenant_id=payload.tenant_id,
        filters=payload.filters,
        top_k=payload.top_k,
    )
    return DocumentContextResponse(answer=query_result.answer, sources=query_result.sources, documents=[])


@router.post("/ingest/canonical", response_model=CanonicalIngestResponse)
def ingest_canonical(
    payload: CanonicalIngestRequest,
    canonical_service: CanonicalIngestionService = Depends(get_canonical_ingestion_service),
) -> CanonicalIngestResponse:
    return canonical_service.ingest(payload)


@router.post("/document-context", response_model=DocumentContextResponse)
def document_context(
    payload: QueryWithFiltersRequest,
    canonical_service: CanonicalIngestionService = Depends(get_canonical_ingestion_service),
) -> DocumentContextResponse:
    return canonical_service.get_document_context(
        tenant_id=payload.tenant_id,
        question=payload.question,
        filters=payload.filters,
        top_k=payload.top_k,
        memory_limit=payload.memory_limit,
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    tenant_id: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    return document_service.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    tenant_id: str = Query(..., min_length=1),
    document_service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponse:
    return document_service.delete_document(tenant_id=tenant_id, document_id=document_id)
