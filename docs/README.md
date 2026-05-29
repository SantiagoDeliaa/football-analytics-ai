# Documentación del Proyecto

Esta carpeta contiene documentación viva para entender el producto, su arquitectura y las reglas de colaboración técnica.

## Qué contiene `docs/`
- Contexto de producto y visión.
- Arquitectura principal (`React + FastAPI + src`) con `Streamlit` archivado como legacy.
- Diseño funcional de Vertical 2 (Event Data).
- Capa de capacidades por provider para mantener la plataforma desacoplada del origen de datos.
- Modelo canónico de eventos.
- Modelos canónicos de entidades para contexto, timeline, stats y disponibilidad de datos.
- Documentación específica de providers e integraciones externas.
- Métricas tácticas (básicas y propietarias).
- Persistencia local (SQLite + JSON).
- Política de uso de datos.
- Deuda técnica y roadmap de evolución.
- Flujo de trabajo para agentes/desarrolladores.
- Validación integrada de frontend + backend.
- Guía concreta de despliegue en Hugging Face Spaces.

## Orden recomendado de lectura
1. `PRODUCT_CONTEXT.md`
2. `ARCHITECTURE.md`
3. `EVENT_DATA_VERTICAL.md`
4. `CANONICAL_EVENT_MODEL.md`
5. `CANONICAL_ENTITY_MODELS.md`
6. `PROVIDER_CAPABILITIES.md`
7. `providers/SPORTMONKS_INTEGRATION.md`
8. `TACTICAL_METRICS.md`
9. `LOCAL_PERSISTENCE.md`
10. `DATA_USAGE_POLICY.md`
11. `TECH_DEBT_AND_REFACTORING.md`
12. `AGENT_WORKFLOW.md`
13. `INTEGRATION_VALIDATION.md`
14. `HUGGINGFACE_DEPLOY.md`

## Cómo usar esta documentación
- Antes de implementar cambios, leer al menos `PRODUCT_CONTEXT.md` y `ARCHITECTURE.md`.
- Si cambia el modelo de eventos, actualizar `CANONICAL_EVENT_MODEL.md`.
- Si cambia la capa de entidades canónicas para providers y match center, actualizar `CANONICAL_ENTITY_MODELS.md`.
- Si cambia la estrategia multi-provider o la habilitacion de modulos por fuente, actualizar `PROVIDER_CAPABILITIES.md`.
- Si cambia la integracion base de Sportmonks, actualizar `providers/SPORTMONKS_INTEGRATION.md`.
- Si cambian métricas o insights, actualizar `TACTICAL_METRICS.md`.
- Si cambia almacenamiento local, actualizar `LOCAL_PERSISTENCE.md`.
- Si cambia el estado de integración frontend/backend, actualizar `INTEGRATION_VALIDATION.md`.
- Si cambia la estrategia de despliegue, actualizar `HUGGINGFACE_DEPLOY.md`.
