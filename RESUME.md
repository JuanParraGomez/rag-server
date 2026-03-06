# RESUME - RAG + MCP + Qdrant Backup

## Objetivo
Plataforma RAG modular con metadata dinamica para embeddings/documentos, consumida por IAs via MCP, con Qdrant persistente y backup remoto al VPS.

## Arquitectura final
- `qdrant` (base vectorial persistente)
- `api` FastAPI (ingesta, query, filtros metadata, multi-tenant)
- `mcp` FastMCP (gateway que consumen las IAs)

Flujo recomendado:
1. IA -> MCP
2. MCP -> FastAPI
3. FastAPI -> Qdrant

Las IAs no escriben directo en Qdrant.

## Estado actual
- Repo: `rag-server`
- Branch: `main`
- MCP tools disponibles:
  - `rag_health`
  - `rag_upload_text`
  - `rag_query`
  - `rag_list_documents`
  - `rag_delete_document`

## Backup Qdrant (manual por ahora)
Script local:
- `/home/juan/Documents/rag_backup_manual.sh`

Configuracion global:
- `/home/juan/Documents/.env`

Que hace el script:
1. Crea snapshot de la coleccion local en Qdrant
2. Descarga el snapshot
3. Lo sube al VPS en `/opt/qdrant-backup/incoming`
4. (Opcional) borra snapshot local temporal

## Variables clave para labels/metadata (futuro)
La app ya soporta metadata dinamica por documento y filtros en query.
Campos sugeridos para labels:
- `tenant_id`
- `cliente`
- `categoria`
- `proyecto`
- `usuario_id`
- `tipo_documento`
- `confidencial`
- `fecha`

## Reglas para estabilidad (no sobreescritura)
- Mantener `qdrant` con volumen persistente.
- Desplegar por imagen/version (`docker compose up -d --build`).
- No exponer escritura directa a Qdrant para IAs.
- Siempre filtrar por `tenant_id`.
- Hacer backup por snapshot y subida remota.

## VPS
- Solo backup Qdrant activo:
  - contenedor: `qdrant-backup`
  - endpoint: `72.61.2.9:6333`
