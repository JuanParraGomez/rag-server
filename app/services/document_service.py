from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.ingestion.extractors import TextExtractor, UnsupportedFileTypeError
from app.ingestion.pipeline import LlamaIngestionService
from app.models.schemas import DeleteDocumentResponse, DocumentListResponse, DocumentSummary, UploadResponse
from app.vector_store.base import VectorStoreAdapter

LOGGER = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, extractor: TextExtractor, ingestion: LlamaIngestionService, vector_store: VectorStoreAdapter) -> None:
        self._extractor = extractor
        self._ingestion = ingestion
        self._vector_store = vector_store

    async def ingest_file(self, file: UploadFile, metadata_json: str) -> UploadResponse:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        raw_content = await file.read()
        try:
            text = self._extractor.extract(file.filename or "", raw_content)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return self._ingest_text(
            text=text,
            metadata=metadata,
            source_type="file",
            title=file.filename,
            file_name=file.filename,
        )

    def ingest_text(self, text: str, metadata: dict[str, Any], title: str | None, document_id: str | None) -> UploadResponse:
        if not text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text cannot be empty")

        return self._ingest_text(
            text=text,
            metadata=metadata,
            source_type="text",
            title=title,
            document_id=document_id,
        )

    def list_documents(self, tenant_id: str, limit: int, offset: int) -> DocumentListResponse:
        total, items = self._vector_store.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)
        return DocumentListResponse(
            tenant_id=tenant_id,
            total=total,
            items=[DocumentSummary(**item) for item in items],
        )

    def delete_document(self, tenant_id: str, document_id: str) -> DeleteDocumentResponse:
        deleted = self._vector_store.delete_document(tenant_id=tenant_id, document_id=document_id)
        return DeleteDocumentResponse(document_id=document_id, tenant_id=tenant_id, deleted=deleted)

    def _ingest_text(
        self,
        text: str,
        metadata: dict[str, Any],
        source_type: str,
        title: str | None,
        file_name: str | None = None,
        document_id: str | None = None,
    ) -> UploadResponse:
        clean_metadata = self._normalize_metadata(metadata)
        tenant_id = clean_metadata.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="metadata.tenant_id is required",
            )

        doc_id = document_id or str(uuid4())
        payload = {
            **clean_metadata,
            "document_id": doc_id,
            "tenant_id": tenant_id,
            "source_type": source_type,
            "title": title,
            "file_name": file_name,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        nodes = self._ingestion.run(text=text, metadata=payload, document_id=doc_id)
        if not nodes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document produced no indexable chunks")

        self._vector_store.upsert_nodes(nodes)
        LOGGER.info(
            "Indexed document_id=%s tenant_id=%s chunks=%s source_type=%s",
            doc_id,
            tenant_id,
            len(nodes),
            source_type,
        )

        return UploadResponse(
            document_id=doc_id,
            tenant_id=tenant_id,
            chunks_indexed=len(nodes),
            metadata=clean_metadata,
        )

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = json.dumps(value, ensure_ascii=True)
        return normalized
