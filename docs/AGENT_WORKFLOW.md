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

## Calidad y seguridad de cambios
- Hacer cambios pequeños, reversibles y explicados.
- Evitar cambios amplios no solicitados.
- Priorizar compatibilidad con la demo actual.

## Pruebas mínimas
- Ejecutar: `pytest tests/test_frontend_regression.py -k vertical2`

## Mantenimiento de documentación
- Si cambia modelo canónico: actualizar `docs/CANONICAL_EVENT_MODEL.md`.
- Si cambian métricas/insights: actualizar `docs/TACTICAL_METRICS.md`.
