# Guía: Métricas Tácticas

## Reglas de implementación
- Calcular métricas desde Canonical Event Model.
- Filtrar eventos no analíticos antes de computar.
- Documentar fórmula y propósito táctico.
- Devolver valor y label cuando aplique.
- Manejar `None` / `No aplica` sin romper UI.
- Agregar/ajustar insights cuando corresponda.

## Mantenimiento documental
- Si cambia una métrica, actualizar `docs/TACTICAL_METRICS.md`.
- Si cambia modelo de datos asociado, actualizar `docs/CANONICAL_EVENT_MODEL.md`.

## Validación mínima
- Verificar que no se rompe `Subir PDF`.
- Ejecutar `pytest tests/test_frontend_regression.py -k vertical2`.
