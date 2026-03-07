from __future__ import annotations

from typing import Any

from app.config.settings import Settings
from app.models.schemas import QueryResponse, SourceChunk
from app.services.reranker_service import RerankerService
from app.vector_store.base import VectorStoreAdapter


class QueryService:
    def __init__(self, vector_store: VectorStoreAdapter, reranker: RerankerService, settings: Settings) -> None:
        self._vector_store = vector_store
        self._reranker = reranker
        self._settings = settings

    def query(self, question: str, tenant_id: str, filters: dict[str, Any], top_k: int) -> QueryResponse:
        effective_filters = dict(filters)
        # Security boundary: tenant_id is always enforced regardless of incoming filters.
        effective_filters["tenant_id"] = tenant_id

        candidate_pool = self._reranker.candidate_pool_size(top_k=top_k, question=question)
        nodes = self._vector_store.query(question=question, filters=effective_filters, top_k=candidate_pool)
        sources = [
            SourceChunk(
                chunk_id=node.node.node_id,
                score=float(node.score or 0.0),
                text=node.node.get_content(),
                metadata=node.node.metadata,
            )
            for node in nodes
        ]
        if self._settings.rerank_enabled:
            sources = self._reranker.rerank(question=question, chunks=sources, final_k=top_k, filters=effective_filters)
        else:
            sources = sources[:top_k]

        answer = self._build_answer(question, sources)
        return QueryResponse(answer=answer, sources=sources)

    @staticmethod
    def _build_answer(question: str, sources: list[SourceChunk]) -> str:
        if not sources:
            return "No matching information was found for the given tenant and filters."

        context = "\n\n".join(source.text.strip() for source in sources[:3] if source.text.strip())
        if not context:
            return "Relevant chunks were found, but they do not contain extractable text."

        return (
            f"Question: {question}\n"
            f"Based on the most relevant retrieved chunks:\n{context[:1800]}"
        )
