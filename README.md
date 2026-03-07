# rag-server

Servicio RAG con FastAPI + Qdrant, expuesto para agentes por FastMCP.

## Arquitectura
- `api` (FastAPI): ingesta, query, filtros, contexto.
- `mcp` (FastMCP): gateway de herramientas para IA.
- `qdrant` (vector DB): persistencia de embeddings.
- Integración canónica: CanonDock para documentos, versiones, tags y memoria.

## Flujo
1. IA -> MCP (`rag_query`, `rag_ingest_text`, etc.).
2. MCP -> API RAG.
3. API RAG -> CanonDock (persistencia canónica) + Qdrant (index/retrieval).
4. RAG aplica reranker y devuelve `answer` + `sources`.

## Stack
- Python 3.11+
- FastAPI
- FastMCP
- Qdrant
- OpenAI embeddings (`text-embedding-3-small`)
- LlamaIndex ingestion
- Docker Compose

## Configuración clave
- `OPENAI_API_KEY`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `CANONDOCK_API_BASE_URL`
- `CHUNKING_STRATEGY=semantic`
- `SEMANTIC_BUFFER_SIZE=1`
- `SEMANTIC_BREAKPOINT_PERCENTILE=90`

## Levantar local
```bash
cd /home/juan/Documents/rag-server
cp -n .env.example .env
sudo docker compose up --build -d
sudo docker compose ps
```

## Verificación rápida
```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8081/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"cli","version":"1.0"}}}'
```
