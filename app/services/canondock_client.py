from __future__ import annotations

from typing import Any

import httpx

from app.config.settings import Settings


class CanonDockClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.canondock_api_base_url.rstrip("/")
        self._timeout = settings.canondock_timeout_seconds
        self._service_token = settings.canondock_service_token

    def _headers(self, request_id: str | None = None) -> dict[str, str]:
        headers = {
            "X-Source-Service": "rag-server",
        }
        if request_id:
            headers["X-Request-Id"] = request_id
            headers["X-Correlation-Id"] = request_id
        if self._service_token:
            headers["Authorization"] = f"Bearer {self._service_token}"
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout)

    def resolve_tag(self, name: str, category: str = "tema", description: str | None = None, request_id: str | None = None) -> dict[str, Any]:
        payload = {
            "name": name,
            "category": category,
            "description": description,
        }
        with self._client() as client:
            response = client.post(f"{self._base_url}/tags/resolve", json=payload, headers=self._headers(request_id))
            response.raise_for_status()
            return response.json()

    def register_text_document(self, payload: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(f"{self._base_url}/documents/register-text", json=payload, headers=self._headers(request_id))
            response.raise_for_status()
            return response.json()

    def create_document_version(self, document_id: str, payload: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(
                f"{self._base_url}/documents/{document_id}/versions",
                json=payload,
                headers=self._headers(request_id),
            )
            response.raise_for_status()
            return response.json()

    def get_document(self, document_id: str, request_id: str | None = None) -> dict[str, Any]:
        with self._client() as client:
            response = client.get(f"{self._base_url}/documents/{document_id}", headers=self._headers(request_id))
            response.raise_for_status()
            return response.json()

    def list_document_versions(self, document_id: str, request_id: str | None = None) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get(f"{self._base_url}/documents/{document_id}/versions", headers=self._headers(request_id))
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            return []

    def create_memory(self, payload: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(f"{self._base_url}/memories", json=payload, headers=self._headers(request_id))
            response.raise_for_status()
            return response.json()

    def list_memories(self, document_id: str | None = None, project_id: str | None = None, limit: int = 20, request_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if document_id:
            params["document_id"] = document_id
        if project_id:
            params["project_id"] = project_id
        with self._client() as client:
            response = client.get(f"{self._base_url}/memories", params=params, headers=self._headers(request_id))
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("items", [])
            return []
