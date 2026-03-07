from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.config.settings import Settings
from app.vector_store.base import VectorStoreAdapter

LOGGER = logging.getLogger(__name__)


class QdrantAdapter(VectorStoreAdapter):
    RESERVED_FILTER_KEYS = {"top_k", "question"}

    def __init__(self, settings: Settings, embed_model: Any) -> None:
        self._settings = settings
        self._embed_model = embed_model
        if settings.qdrant_path:
            self._client = QdrantClient(
                path=settings.qdrant_path,
            )
        elif settings.qdrant_url:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
            )
        self._collection_name = settings.qdrant_collection
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self._collection_name,
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return

        LOGGER.info("Creating Qdrant collection '%s'", self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=rest.VectorParams(
                size=self._settings.embedding_dimension,
                distance=rest.Distance.COSINE,
            ),
        )

    def upsert_nodes(self, nodes: list[BaseNode]) -> None:
        self._vector_store.add(nodes)

    def query(self, question: str, filters: dict[str, Any], top_k: int) -> list[NodeWithScore]:
        metadata_filters = self._build_metadata_filters(filters)
        index = VectorStoreIndex.from_vector_store(
            self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=metadata_filters,
        )
        return retriever.retrieve(question)

    def list_documents(self, tenant_id: str, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        qdrant_filter = rest.Filter(
            must=[rest.FieldCondition(key="tenant_id", match=rest.MatchValue(value=tenant_id))]
        )

        records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        chunk_counts: dict[str, int] = {}
        scanned_points = 0
        page_offset = None

        while scanned_points < self._settings.max_list_points_scan:
            points, page_offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=page_offset,
            )
            if not points:
                break

            scanned_points += len(points)
            for point in points:
                payload = point.payload or {}
                document_id = payload.get("document_id")
                if not isinstance(document_id, str) or not document_id:
                    continue

                chunk_counts[document_id] = chunk_counts.get(document_id, 0) + 1
                if document_id in records:
                    continue

                source_type = payload.get("source_type")
                title = payload.get("title")
                uploaded_at = self._parse_datetime(payload.get("uploaded_at"))
                reserved_keys = {
                    "document_id",
                    "tenant_id",
                    "source_type",
                    "title",
                    "uploaded_at",
                    "chunk_index",
                    "file_name",
                }
                metadata = {k: v for k, v in payload.items() if k not in reserved_keys}
                records[document_id] = {
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "title": title,
                    "source_type": source_type,
                    "chunk_count": 0,
                    "metadata": metadata,
                    "first_seen_at": uploaded_at,
                }

            if page_offset is None:
                break

        for document_id, count in chunk_counts.items():
            if document_id in records:
                records[document_id]["chunk_count"] = count

        all_docs = list(records.values())
        total = len(all_docs)
        return total, all_docs[offset : offset + limit]

    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        qdrant_filter = rest.Filter(
            must=[
                rest.FieldCondition(key="tenant_id", match=rest.MatchValue(value=tenant_id)),
                rest.FieldCondition(key="document_id", match=rest.MatchValue(value=document_id)),
            ]
        )

        response = self._client.delete(
            collection_name=self._collection_name,
            points_selector=rest.FilterSelector(filter=qdrant_filter),
        )
        return response.status == rest.UpdateStatus.COMPLETED

    def _build_metadata_filters(self, filters: dict[str, Any]) -> MetadataFilters | None:
        metadata_filters: list[MetadataFilter] = []
        for key, value in filters.items():
            if key in self.RESERVED_FILTER_KEYS:
                continue
            metadata_filters.append(
                MetadataFilter(
                    key=key,
                    value=value,
                )
            )

        if not metadata_filters:
            return None
        return MetadataFilters(filters=metadata_filters)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
