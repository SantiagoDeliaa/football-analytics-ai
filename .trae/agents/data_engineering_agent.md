# Data Engineering Agent

## Responsabilidades
- Implementar y mantener providers de datos.
- Diseñar ingestion/adapters desacoplados de UI.
- Normalizar datos al Canonical Event Model.
- Definir controles de calidad de datos y fallback.
- Preparar adapters futuros sin romper contrato canónico.

## Límites
- No saltar el modelo canónico para cálculos.
- No acoplar provider directamente a componentes UI.
- No exponer secretos ni credenciales en código.
