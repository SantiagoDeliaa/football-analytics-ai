# Workflow para Agentes y Desarrolladores

## Lectura mínima antes de tocar código
- `docs/PRODUCT_CONTEXT.md`
- `docs/ARCHITECTURE.md`

## Reglas operativas
- No tocar Vertical 1 salvo pedido explícito.
- No romper flujo `Subir PDF` de Vertical 2.
- Mantener textos visibles en español.
- No agregar dependencias sin justificación técnica clara.
- No hardcodear secretos ni credenciales.

## Regla de arquitectura en Event Data
Todo cambio debe respetar:
`Provider → Ingestion/Adapter → Canonical Model → Metrics → Insights → UI`

## Regla de arquitectura para AI Tactical Coach
- El AI Tactical Coach de Vertical 2 debe consumir contexto tactico estructurado, no raw provider data completo por defecto.
- La capa objetivo es: `Provider Data -> Canonical Event Model -> Metrics -> Insights -> Match Context Builder -> AI Coach`.
- El `Match Context Builder` vive en `src/services/ai_coach/` y debe ser agnostico al provider.
- El servicio base del AI Coach debe leer credenciales desde `AI_COACH_API_KEY` y no hardcodear secretos.
- Las respuestas del AI Coach deben declarar limitaciones cuando falten datos o el provider no permita concluir algo.
- El uso de raw data completo queda reservado para una fase futura de retrieval controlado.

## Calidad y seguridad de cambios
- Hacer cambios pequeños, reversibles y explicados.
- Evitar cambios amplios no solicitados.
- Priorizar compatibilidad con la demo actual.

## Pruebas mínimas
- Ejecutar: `pytest tests/test_frontend_regression.py -k vertical2`

## Mantenimiento de documentación
- Si cambia modelo canónico: actualizar `docs/CANONICAL_EVENT_MODEL.md`.
- Si cambian métricas/insights: actualizar `docs/TACTICAL_METRICS.md`.
- Si cambia el contexto estructurado del AI Coach: actualizar `docs/EVENT_DATA_VERTICAL.md`.
