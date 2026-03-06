# RAG Server (FastAPI + LlamaIndex + Qdrant)

Open source, modular, production-ready baseline for document RAG with dynamic metadata filters and multi-tenant isolation.

## 1. Resumen de la arquitectura

- API HTTP en FastAPI para ingesta, consulta y gestión de documentos.
- Pipeline de ingestion con LlamaIndex para chunking + embeddings.
- Qdrant como vector database para almacenar vectores + payload (metadata dinámica).
- Filtros semánticos por metadata en consultas.
- Diseño preparado para multi-tenant con `tenant_id` obligatorio.

Flujo:
1. `upload-file` o `upload-text` recibe contenido + metadata JSON dinámica.
2. Se extrae texto, se chunkea, se generan embeddings, se indexa en Qdrant.
3. `query` recupera chunks relevantes aplicando filtros de metadata y responde con fuentes.

## 2. Patrones de diseño utilizados

- **Adapter pattern**: `VectorStoreAdapter` permite reemplazar Qdrant por Weaviate/Pinecone sin reescribir servicios.
- **Service layer**: `DocumentService` y `QueryService` centralizan reglas de negocio.
- **Repository-ish approach**: acceso a Qdrant encapsulado en `QdrantAdapter`.
- **Dependency Injection básica**: `app/dependencies.py` construye dependencias por capa.
- **Config por entorno**: `pydantic-settings` + `.env`.

## 3. Estructura de carpetas

```text
rag-server/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── config/
│   │   └── settings.py
│   ├── ingestion/
│   │   ├── extractors.py
│   │   └── pipeline.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── document_service.py
│   │   └── query_service.py
│   ├── vector_store/
│   │   ├── base.py
│   │   └── qdrant_adapter.py
│   ├── utils/
│   │   └── logging.py
│   ├── dependencies.py
│   └── main.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## 4. Endpoints

- `POST /upload-file` (multipart): archivo + `metadata` JSON string dinámica.
- `POST /upload-text` (json): texto + metadata dinámica.
- `POST /query` (json): pregunta + `tenant_id` + filtros metadata opcionales.
- `GET /documents?tenant_id=...`: lista documentos indexados por tenant.
- `DELETE /documents/{document_id}?tenant_id=...`: borra documento y todos sus chunks.
- `GET /health`: health check.

## 5. Metadata personalizada dinámica

La metadata es un `dict[str, Any]` no hardcodeado. Ejemplo:

```json
{
  "cliente": "acme",
  "categoria": "contrato",
  "proyecto": "onboarding_2026",
  "usuario_id": "123",
  "tenant_id": "empresa_7",
  "confidencial": true,
  "fecha": "2026-03-06"
}
```

Se persiste como payload en Qdrant y se usa para filtros en query.

## 6. Multi-tenant (aislamiento)

- `tenant_id` es obligatorio al indexar (`metadata.tenant_id`).
- En `/query`, el `tenant_id` se fuerza en filtros internos (siempre se aplica).
- Recomendación de hardening para producción:
  - extraer `tenant_id` de JWT/claims (no del body)
  - middleware de autorización por tenant
  - colecciones separadas por tenant para aislamiento fuerte si es requerido

## 7. Docker y ejecución local

### 7.1 Crear `.env`

```bash
cp .env.example .env

# set OPENAI_API_KEY en .env para embeddings OpenAI
```

### 7.2 Levantar servicios

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Qdrant: `http://localhost:6333`

## 8. Ejemplos con curl

### 8.1 Health

```bash
curl -s http://localhost:8000/health
```

### 8.2 Upload de texto con metadata dinámica

```bash
curl -X POST http://localhost:8000/upload-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Este contrato establece condiciones de servicio para ACME.",
    "title": "contrato_acme",
    "metadata": {
      "tenant_id": "empresa_7",
      "cliente": "acme",
      "categoria": "legal",
      "proyecto": "onboarding_2026",
      "usuario_id": "123",
      "tipo_documento": "contrato",
      "confidencial": true,
      "fecha": "2026-03-06"
    }
  }'
```

### 8.3 Upload de archivo (PDF/TXT/DOCX)

```bash
curl -X POST http://localhost:8000/upload-file \
  -F 'file=@/ruta/documento.pdf' \
  -F 'metadata={"tenant_id":"empresa_7","cliente":"acme","categoria":"legal","proyecto":"onboarding_2026"}'
```

### 8.4 Query semántica con filtros metadata

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

### 8.5 Listar documentos indexados por tenant

```bash
curl -s "http://localhost:8000/documents?tenant_id=empresa_7&limit=20&offset=0"
```

### 8.6 Eliminar documento por id

```bash
curl -X DELETE "http://localhost:8000/documents/<document_id>?tenant_id=empresa_7"
```

## 9. Decisiones técnicas

- **Extracción PDF/DOCX**: `pypdf` y `python-docx` por ser open source, estables y simples.
- ****Embeddings**: por defecto `text-embedding-3-small` (OpenAI) para mejor precision en busqueda; fallback open source con `EMBEDDING_PROVIDER=huggingface`.
- **Chunking**: `SentenceSplitter` de LlamaIndex con `chunk_size=800`, `chunk_overlap=120`.
- **Persistencia de document_id/metadata**: payload en cada chunk de Qdrant (`document_id`, `tenant_id`, tags dinámicos).
- **Borrado por documento**: `delete` con filtro payload (`tenant_id` + `document_id`) en Qdrant.
- **Filtros en query**: `MetadataFilters` de LlamaIndex, traducidos a filtro de vector store.
- **Cambio futuro de vector store**: interfaz `VectorStoreAdapter` evita acoplamiento a Qdrant.

## 10. Adaptación para Google Cloud / Vertex AI / Cloud Run

Manteniendo base open source:

- Despliegue API en Cloud Run usando el mismo `Dockerfile`.
- Qdrant:
  - opción A: Qdrant gestionado externo
  - opción B: VM / GKE con volumen persistente
- Secretos (`QDRANT_API_KEY`) en Secret Manager.
- Escalado horizontal de API sin estado.
- Vertex AI puede integrarse luego solo para LLM de respuesta (opcional), manteniendo embeddings + vector store OSS.

## 11. Mejoras futuras recomendadas

- Agregar auth (JWT/OIDC) y tenant extraído de token.
- Agregar endpoint de reindexación y versionado de documentos.
- Agregar tests (unit + integration con Qdrant en docker).
- Implementar response synthesis con LLM open source (Ollama/vLLM) configurable.
- Incorporar colas (Celery/RQ) para ingesta asíncrona de archivos grandes.
