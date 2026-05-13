# Documentación del Proyecto

Esta carpeta contiene documentación viva para entender el producto, su arquitectura y las reglas de colaboración técnica.

## Qué contiene `docs/`
- Contexto de producto y visión.
- Arquitectura actual híbrida (`React + FastAPI + Streamlit legado`).
- Diseño funcional de Vertical 2 (Event Data).
- Modelo canónico de eventos.
- Métricas tácticas (básicas y propietarias).
- Persistencia local (SQLite + JSON).
- Política de uso de datos.
- Deuda técnica y roadmap de evolución.
- Flujo de trabajo para agentes/desarrolladores.
- Validación integrada de frontend + backend.

## Orden recomendado de lectura
1. `PRODUCT_CONTEXT.md`
2. `ARCHITECTURE.md`
3. `EVENT_DATA_VERTICAL.md`
4. `CANONICAL_EVENT_MODEL.md`
5. `TACTICAL_METRICS.md`
6. `LOCAL_PERSISTENCE.md`
7. `DATA_USAGE_POLICY.md`
8. `TECH_DEBT_AND_REFACTORING.md`
9. `AGENT_WORKFLOW.md`
10. `INTEGRATION_VALIDATION.md`

## Cómo usar esta documentación
- Antes de implementar cambios, leer al menos `PRODUCT_CONTEXT.md` y `ARCHITECTURE.md`.
- Si cambia el modelo de eventos, actualizar `CANONICAL_EVENT_MODEL.md`.
- Si cambian métricas o insights, actualizar `TACTICAL_METRICS.md`.
- Si cambia almacenamiento local, actualizar `LOCAL_PERSISTENCE.md`.
- Si cambia el estado de integración frontend/backend, actualizar `INTEGRATION_VALIDATION.md`.
