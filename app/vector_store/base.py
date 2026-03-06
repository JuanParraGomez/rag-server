from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.schema import BaseNode, NodeWithScore


class VectorStoreAdapter(ABC):
    @abstractmethod
    def ensure_collection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_nodes(self, nodes: list[BaseNode]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, question: str, filters: dict[str, Any], top_k: int) -> list[NodeWithScore]:
        raise NotImplementedError

    @abstractmethod
    def list_documents(self, tenant_id: str, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        raise NotImplementedError
