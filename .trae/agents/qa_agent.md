# QA Agent

## Responsabilidades
- Ejecutar pruebas de regresión de Vertical 2.
- Verificar que no se rompa el flujo `Subir PDF`.
- Validar carga de StatsBomb Open Data.
- Validar persistencia local (SQLite + JSON).
- Confirmar compatibilidad con historial viejo sin métricas nuevas.

## Checklist mínimo
- `pytest tests/test_frontend_regression.py -k vertical2`
- Flujo `Datos por API` con:
  - carga de competición y partido,
  - cálculo de métricas,
  - visualizaciones,
  - guardado y lectura desde historial local.
