# Política de Uso de Datos

## Principio general
El sistema transforma datos crudos en inteligencia táctica. El producto comercial debe centrarse en transformación analítica, no en redistribución de datos originales.

## Reglas
- No redistribuir data cruda como producto comercial.
- No exponer endpoints que devuelvan eventos originales completos.
- El output comercial debe ser:
  - métricas
  - insights
  - visualizaciones transformadas
- Las métricas no deberían permitir reconstruir íntegramente la data original.

## Persistencia local
- Los JSON crudos locales se usan como cache/desarrollo interno.
- No son formato de entrega final al cliente.

## Seguridad
- No hardcodear API keys ni secretos en código o docs.
- No guardar credenciales en repositorio.
- `API_FOOTBALL_KEY` debe manejarse por variable de entorno o `.env` local no versionado.
- `AI_COACH_API_KEY` debe manejarse por variable de entorno o `.env` local no versionado.

## AI Tactical Coach
- El AI Coach debe consumir `match_context` estructurado como entrada principal.
- No debe consumir raw provider data completo por defecto.
- Si faltan datos o el provider no ofrece cierta granularidad, la respuesta debe declarar esa limitacion de forma explicita.
- El uso futuro de raw data solo debe habilitarse mediante retrieval controlado y con cuidado de licencia/seguridad.

## Cumplimiento por provider
- Cualquier nuevo provider debe validarse contra términos de licencia y uso.
- El adapter debe respetar restricciones de atribución y distribución.
- API-Football / API-SPORTS debe revisarse según su licencia y plan contratado antes de exponer outputs comerciales.
