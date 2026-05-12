# Guía: Provider Adapter para Event Data

## Regla principal
No conectar un provider directamente a UI.

## Flujo obligatorio
1. Crear capa de ingestion/adapter del provider.
2. Mapear eventos al Canonical Event Model.
3. Manejar faltantes y errores de forma controlada.
4. Mantener fallback para no romper demo.
5. Exponer salida lista para métricas/insights/UI.

## Buenas prácticas
- Validar estructura de datos antes de normalizar.
- No asumir campos obligatorios si el provider no los garantiza.
- Registrar origen real/fallback cuando corresponda.
- Si el provider requiere credenciales, usar variables de entorno y nunca hardcodear keys.
- Si el provider no entrega coordenadas, mapear igual al Canonical Event Model y desactivar visualizaciones espaciales que no apliquen.
- Actualizar documentación:
  - `docs/CANONICAL_EVENT_MODEL.md`
  - `docs/EVENT_DATA_VERTICAL.md`
