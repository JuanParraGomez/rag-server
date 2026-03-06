from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.ingestion.extractors import TextExtractor
from app.ingestion.pipeline import LlamaIngestionService
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.vector_store.qdrant_adapter import QdrantAdapter


@lru_cache(maxsize=1)
def get_vector_store_adapter() -> QdrantAdapter:
    settings = get_settings()
    ingestion = LlamaIngestionService(settings)
    adapter = QdrantAdapter(settings=settings, embed_model=ingestion.embed_model)
    adapter.ensure_collection()
    return adapter


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    settings: Settings = get_settings()
    ingestion = LlamaIngestionService(settings)
    vector_store = get_vector_store_adapter()
    extractor = TextExtractor()
    return DocumentService(extractor=extractor, ingestion=ingestion, vector_store=vector_store)


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    vector_store = get_vector_store_adapter()
    return QueryService(vector_store=vector_store)
