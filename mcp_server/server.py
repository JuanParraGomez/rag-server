from __future__ import annotations

import os
from typing import Any

import httpx
from fastmcp import FastMCP

RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://api:8000").rstrip("/")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8081"))
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "30"))

mcp = FastMCP("rag-mcp-gateway")


def _client() -> httpx.Client:
    return httpx.Client(timeout=REQUEST_TIMEOUT)


@mcp.tool()
def rag_health() -> dict[str, Any]:
    """Returns health status from the RAG API."""
    with _client() as client:
        response = client.get(f"{RAG_API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_upload_text(text: str, tenant_id: str, metadata: dict[str, Any] | None = None, title: str | None = None) -> dict[str, Any]:
    """Uploads plain text into RAG with dynamic metadata and mandatory tenant_id."""
    payload = {
        "text": text,
        "title": title,
        "metadata": {
            **(metadata or {}),
            "tenant_id": tenant_id,
        },
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/upload-text", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_ingest_text(
    text: str,
    tenant_id: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    memory: dict[str, Any] | None = None,
    canonical_document_id: str | None = None,
) -> dict[str, Any]:
    """Canonical ingest: persists in CanonDock first, then indexes in RAG."""
    payload = {
        "content_type": "text",
        "text": text,
        "title": title,
        "metadata": {**(metadata or {}), "tenant_id": tenant_id},
        "tags": tags or [],
        "memory": memory,
        "canonical_document_id": canonical_document_id,
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/ingest/canonical", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_ingest_file(
    file_name: str,
    file_base64: str,
    tenant_id: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    memory: dict[str, Any] | None = None,
    canonical_document_id: str | None = None,
) -> dict[str, Any]:
    """Canonical file ingest: file bytes as base64, CanonDock-first orchestration."""
    payload = {
        "content_type": "file",
        "file_name": file_name,
        "file_base64": file_base64,
        "title": title,
        "metadata": {**(metadata or {}), "tenant_id": tenant_id},
        "tags": tags or [],
        "memory": memory,
        "canonical_document_id": canonical_document_id,
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/ingest/canonical", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_query(question: str, tenant_id: str, filters: dict[str, Any] | None = None, top_k: int = 5) -> dict[str, Any]:
    """Runs semantic query with tenant isolation and optional metadata filters."""
    payload = {
        "question": question,
        "tenant_id": tenant_id,
        "filters": filters or {},
        "top_k": top_k,
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_query_with_filters(
    question: str,
    tenant_id: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    include_context: bool = True,
    memory_limit: int = 10,
) -> dict[str, Any]:
    """Query with metadata filters and optional CanonDock context enrichment."""
    payload = {
        "question": question,
        "tenant_id": tenant_id,
        "filters": filters or {},
        "top_k": top_k,
        "include_context": include_context,
        "memory_limit": memory_limit,
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/query/with-filters", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_get_document_context(
    question: str,
    tenant_id: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    memory_limit: int = 10,
) -> dict[str, Any]:
    """Returns retrieval answer plus CanonDock versions/memories for matched documents."""
    payload = {
        "question": question,
        "tenant_id": tenant_id,
        "filters": filters or {},
        "top_k": top_k,
        "include_context": True,
        "memory_limit": memory_limit,
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/document-context", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_sync_document(document_id: str, tenant_id: str, text: str, metadata: dict[str, Any] | None = None, title: str | None = None) -> dict[str, Any]:
    """Manual sync helper: re-indexes text for an existing CanonDock document_id."""
    payload = {
        "content_type": "text",
        "text": text,
        "title": title,
        "metadata": {**(metadata or {}), "tenant_id": tenant_id},
        "tags": [],
        "canonical_document_id": document_id,
    }
    with _client() as client:
        response = client.post(f"{RAG_API_BASE_URL}/ingest/canonical", json=payload)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_list_documents(tenant_id: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """Lists indexed documents for a tenant."""
    with _client() as client:
        response = client.get(
            f"{RAG_API_BASE_URL}/documents",
            params={"tenant_id": tenant_id, "limit": limit, "offset": offset},
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
def rag_delete_document(document_id: str, tenant_id: str) -> dict[str, Any]:
    """Deletes a document and all chunks associated with it in the tenant scope."""
    with _client() as client:
        response = client.delete(
            f"{RAG_API_BASE_URL}/documents/{document_id}",
            params={"tenant_id": tenant_id},
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT, host=MCP_HOST, port=MCP_PORT)
