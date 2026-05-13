# Deuda Técnica y Refactoring

## Actual / MVP
- UI en Streamlit.
- Persistencia local con SQLite + JSON.
- StatsBomb Open Data como primer provider.
- Métricas heurísticas iniciales.
- Visualizaciones tácticas iniciales en Plotly.
- Flujo PDF como demo/fallback temporal en Vertical 2.

## Futuro recomendado
- Backend FastAPI para exponer dominio de datos/analítica.
- Frontend React/Next.js para UX más robusta.
- PostgreSQL como persistencia principal.
- Mayor desacople en repositorios/servicios de dominio.
- Tests de dominio específicos para métricas e insights.
- Provider adapters formales por fuente.
- Autenticación/autorización.
- Modelo multi-tenant si evoluciona a SaaS.

## Recomendación de arquitectura
- Priorizar **modular monolith** antes de microservicios.
- Microservicios solo cuando exista:
  - escala operativa real,
  - equipo con ownership claro por dominio,
  - necesidad de despliegue independiente.

## Riesgos actuales
- Acoplamiento funcional por vivir todo en Streamlit.
- Heurísticas de métricas sin calibración completa.
- Persistencia local útil para demo, no para producción multiusuario.
