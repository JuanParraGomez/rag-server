# RAG Server (FastAPI + Qdrant + FastMCP)

Arquitectura recomendada para que sobreviva en producción sin sobreescrituras:

- `qdrant` = base de datos vectorial persistente (no tocar directo desde IAs)
- `api` = servicio RAG (ingesta, embeddings, query, metadata)
- `mcp` = gateway FastMCP que consumen las IAs

## Recomendación clave: Docker vs ejecutar en equipo

Para tu caso, **Docker** es la mejor opción:

- Evita que un cambio local rompa producción.
- Mantiene versiones fijas y reproducibles.
- Separa responsabilidades por contenedor (`qdrant`, `api`, `mcp`).
- Facilita backup/restore de Qdrant por volumen.

Ejecutar directamente en el equipo solo para desarrollo rápido.

## Topología segura (la que ya te configuré)

1. Las IAs llaman solo al `mcp`.
2. `mcp` llama al `api`.
3. `api` usa `qdrant`.
4. Nadie más escribe directo en `qdrant`.

Esto evita sobreescrituras accidentales y centraliza control en FastAPI.

## Servicios y puertos

- FastAPI RAG: `http://localhost:8000`
- Qdrant: `http://localhost:6333`
- FastMCP gateway: `http://localhost:8081`

## Estructura relevante

```text
rag-server/
├── app/                  # FastAPI RAG
├── mcp_server/           # FastMCP gateway
├── docker-compose.yml    # qdrant + api + mcp
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Variables

Copia:

```bash
cp .env.example .env
```

Variables críticas:

- `OPENAI_API_KEY`
- `QDRANT_HOST=qdrant`
- `QDRANT_PORT=6333`
- `RAG_API_BASE_URL=http://api:8000`

## Levantar todo

```bash
docker compose up --build -d
```

## FastMCP tools expuestos

El gateway `mcp_server.server` publica:

- `rag_health`
- `rag_upload_text`
- `rag_query`
- `rag_list_documents`
- `rag_delete_document`

## Endpoints FastAPI

- `POST /upload-file`
- `POST /upload-text`
- `POST /query`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `GET /health`

## Operación para que no se sobreescriba nada

- No montes el código con bind mount en producción.
- Mantén `qdrant_data` como volumen Docker persistente.
- Haz deploy por imagen (`docker compose up -d --build`).
- Usa siempre `tenant_id` y filtros por metadata.
- Haz backup de Qdrant (snapshots) y sube al VPS backup.

## Curls rápidos

### Health

```bash
curl -s http://localhost:8000/health
```

### Upload text

```bash
curl -X POST http://localhost:8000/upload-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contrato de prueba ACME",
    "title": "contrato_acme",
    "metadata": {
      "tenant_id": "empresa_7",
      "cliente": "acme",
      "categoria": "legal",
      "proyecto": "onboarding_2026"
    }
  }'
```

### Query con filtros

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Que dice el contrato?",
    "tenant_id": "empresa_7",
    "filters": {
      "categoria": "legal",
      "cliente": "acme"
    },
    "top_k": 5
  }'
```
