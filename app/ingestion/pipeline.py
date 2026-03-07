from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding

from app.config.settings import Settings


class LlamaIngestionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embed_model = self._build_embed_model(settings)
        parser = self._build_parser(settings)
        self._pipeline = IngestionPipeline(
            transformations=[
                parser,
                self._embed_model,
            ]
        )

    @property
    def embed_model(self) -> Any:
        return self._embed_model

    def run(self, text: str, metadata: dict[str, Any], document_id: str) -> list[BaseNode]:
        doc = Document(text=text, metadata=metadata, id_=document_id)
        try:
            nodes = self._pipeline.run(documents=[doc])
        except Exception:
            # Keep ingestion available even if semantic parsing fails for edge-case content.
            fallback = IngestionPipeline(
                transformations=[
                    SentenceSplitter(
                        chunk_size=self._settings.chunk_size,
                        chunk_overlap=self._settings.chunk_overlap,
                    ),
                    self._embed_model,
                ]
            )
            nodes = fallback.run(documents=[doc])
        for idx, node in enumerate(nodes):
            node.metadata["document_id"] = document_id
            node.metadata["chunk_index"] = idx
        return nodes

    def _build_parser(self, settings: Settings) -> Any:
        if settings.chunking_strategy == "semantic":
            return SemanticSplitterNodeParser.from_defaults(
                embed_model=self._embed_model,
                buffer_size=settings.semantic_buffer_size,
                breakpoint_percentile_threshold=settings.semantic_breakpoint_percentile,
            )
        return SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    @staticmethod
    def _build_embed_model(settings: Settings) -> Any:
        if settings.embedding_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
            return OpenAIEmbedding(
                model=settings.embedding_model_name,
                api_key=settings.openai_api_key,
            )

        raise ValueError(
            "Only EMBEDDING_PROVIDER=openai is supported in this deployment profile"
        )
