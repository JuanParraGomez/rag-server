from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
from httpx import HTTPError

from app.ingestion.extractors import TextExtractor, UnsupportedFileTypeError
from app.models.schemas import CanonicalIngestRequest, CanonicalIngestResponse, DocumentContextResponse, UploadResponse
from app.services.canondock_client import CanonDockClient
from app.services.document_service import DocumentService
from app.services.query_service import QueryService


class CanonicalIngestionService:
    def __init__(
        self,
        canondock: CanonDockClient,
        document_service: DocumentService,
        query_service: QueryService,
        extractor: TextExtractor,
    ) -> None:
        self._canondock = canondock
        self._document_service = document_service
        self._query_service = query_service
        self._extractor = extractor

    def ingest(self, payload: CanonicalIngestRequest) -> CanonicalIngestResponse:
        request_id = str(uuid4())
        metadata = self._normalize_metadata(payload.metadata)
        tenant_id = metadata.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="metadata.tenant_id is required")

        text = self._resolve_text(payload)
        resolved_tags = self._resolve_tags(payload.tags, metadata, request_id)
        canondock_doc = self._persist_document_in_canondock(payload=payload, text=text, resolved_tags=resolved_tags, metadata=metadata, request_id=request_id)
        rag_response = self._index_in_rag(payload=payload, text=text, metadata=metadata, canondock_doc=canondock_doc)
        memory_response = self._persist_memory_if_present(payload, canondock_doc, request_id)

        return CanonicalIngestResponse(
            status="indexed",
            request_id=request_id,
            document_id=canondock_doc["document_id"],
            document_family_id=canondock_doc["document_family_id"],
            version=canondock_doc["current_version"],
            is_latest=True,
            canonical_tags=resolved_tags,
            rag=rag_response,
            memory=memory_response,
        )

    def get_document_context(self, tenant_id: str, question: str, filters: dict[str, Any], top_k: int, memory_limit: int) -> DocumentContextResponse:
        query_result = self._query_service.query(question=question, tenant_id=tenant_id, filters=filters, top_k=top_k)
        seen: set[str] = set()
        documents: list[dict[str, Any]] = []

        for source in query_result.sources:
            document_id = source.metadata.get("document_id")
            if not isinstance(document_id, str) or not document_id or document_id in seen:
                continue
            seen.add(document_id)
            try:
                canondock_doc = self._canondock.get_document(document_id)
                versions = self._canondock.list_document_versions(document_id)
                memories = self._canondock.list_memories(document_id=document_id, limit=memory_limit)
            except HTTPError:
                canondock_doc = {}
                versions = []
                memories = []
            documents.append(
                {
                    "document_id": document_id,
                    "title": canondock_doc.get("title") or source.metadata.get("title"),
                    "project_id": canondock_doc.get("project_id") or source.metadata.get("proyecto"),
                    "status": canondock_doc.get("status"),
                    "current_version": canondock_doc.get("current_version"),
                    "versions": versions[:5],
                    "memories": memories,
                }
            )

        return DocumentContextResponse(
            answer=query_result.answer,
            sources=query_result.sources,
            documents=documents,
        )

    @staticmethod
    def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(metadata)
        if "proyecto" in normalized and "project_id" not in normalized:
            normalized["project_id"] = normalized["proyecto"]
        if "tipo_documento" in normalized and "doc_type" not in normalized:
            normalized["doc_type"] = normalized["tipo_documento"]
        return normalized

    def _resolve_text(self, payload: CanonicalIngestRequest) -> str:
        if payload.content_type == "text":
            if not payload.text or not payload.text.strip():
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="text is required when content_type=text")
            return payload.text

        if not payload.file_base64 or not payload.file_name:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="file_base64 and file_name are required when content_type=file")
        try:
            content = base64.b64decode(payload.file_base64)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_base64 is not valid base64") from exc
        try:
            return self._extractor.extract(payload.file_name, content)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    def _resolve_tags(self, tags: list[str], metadata: dict[str, Any], request_id: str) -> list[str]:
        category = str(metadata.get("categoria", "tema"))
        resolved: list[str] = []
        for tag in tags:
            try:
                tag_result = self._canondock.resolve_tag(name=tag, category=category, request_id=request_id)
            except HTTPError as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Canonical tag resolution failed: {exc}") from exc
            canonical_slug = tag_result.get("canonical_slug")
            if isinstance(canonical_slug, str) and canonical_slug:
                resolved.append(canonical_slug)
        return resolved

    def _persist_document_in_canondock(
        self,
        payload: CanonicalIngestRequest,
        text: str,
        resolved_tags: list[str],
        metadata: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        doc_type = str(metadata.get("doc_type", "nota"))
        project_id = self._safe_str(metadata.get("project_id"))
        author = self._safe_str(metadata.get("usuario_id"))
        source_fingerprint = self._safe_str(metadata.get("source_fingerprint"))
        if not source_fingerprint:
            source_fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

        base_payload: dict[str, Any] = {
            "title": payload.title or f"RAG canonical ingest {now_iso}",
            "doc_type": doc_type,
            "project_id": project_id,
            "author": author,
            "source_fingerprint": source_fingerprint,
            "tags": resolved_tags,
            "metadata": self._stringify_metadata(metadata),
        }

        try:
            if payload.canonical_document_id:
                version_payload = {
                    "title": base_payload["title"],
                    "source_fingerprint": source_fingerprint,
                    "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "metadata": base_payload["metadata"],
                    "status": "active",
                    "extracted_text": text,
                }
                return self._canondock.create_document_version(payload.canonical_document_id, version_payload, request_id=request_id)

            register_payload = {
                **base_payload,
                "extracted_text": text,
            }
            return self._canondock.register_text_document(register_payload, request_id=request_id)
        except HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Canonical persistence failed: {exc}") from exc

    def _index_in_rag(self, payload: CanonicalIngestRequest, text: str, metadata: dict[str, Any], canondock_doc: dict[str, Any]) -> UploadResponse:
        rag_metadata: dict[str, Any] = {
            **metadata,
            "tenant_id": metadata.get("tenant_id"),
            "document_id": canondock_doc["document_id"],
            "document_family_id": canondock_doc["document_family_id"],
            "version": canondock_doc["current_version"],
            "version_id": f"{canondock_doc['document_id']}:v{canondock_doc['current_version']}",
            "project_id": canondock_doc.get("project_id"),
            "status": canondock_doc.get("status"),
            "is_latest": True,
            "canonical_tags": canondock_doc.get("tags", []),
            "tipo_documento": canondock_doc.get("doc_type"),
        }
        return self._document_service.ingest_text(
            text=text,
            metadata=rag_metadata,
            title=payload.title,
            document_id=canondock_doc["document_id"],
        )

    def _persist_memory_if_present(self, payload: CanonicalIngestRequest, canondock_doc: dict[str, Any], request_id: str) -> dict[str, Any] | None:
        if not payload.memory:
            return None
        memory_payload = {
            "memory_type": payload.memory.memory_type,
            "title": payload.memory.title,
            "content": payload.memory.content,
            "project_id": canondock_doc.get("project_id"),
            "document_id": canondock_doc["document_id"],
            "document_family_id": canondock_doc["document_family_id"],
            "topic_tag_ids": [],
            "source_refs": [canondock_doc["document_id"]],
            "relevance_score": payload.memory.relevance_score,
            "created_by": payload.memory.created_by,
        }
        try:
            return self._canondock.create_memory(memory_payload, request_id=request_id)
        except HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Memory persistence failed after indexing: {exc}") from exc

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _stringify_metadata(metadata: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            result[str(key)] = str(value)
        return result
