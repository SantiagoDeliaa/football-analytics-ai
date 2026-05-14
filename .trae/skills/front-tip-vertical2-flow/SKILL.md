---
name: "front-tip-vertical2-flow"
description: "Guia del flujo `Vertical 2` en `front-tip`. Invocá cuando cambies PDF, providers, historial, metricas, insights, AI Coach o endpoints de event-data."
---

# Front Tip Vertical 2 Flow

## Objetivo

Modificar `Vertical 2` sin romper la experiencia principal de la demo ni el contrato entre provider, modelo canonico, metricas e interfaz.

## Cuándo invocarla

Invocá esta skill cuando:

- trabajes en `src/pages/Vertical2Page.tsx`
- modifiques componentes de `src/components/vertical2`
- toques `src/services/eventDataApi.ts`
- agregues providers o variantes de ingestion
- cambies metricas, insights, historial o AI Coach
- ajustes la experiencia de PDF o API Event Data

## Flujo obligatorio

Mantener esta cadena:

`Provider -> ingestion/service -> Canonical Event Model -> metrics/selectors -> insights -> UI`

Nunca conectar un provider directamente a la UI.

## Reglas operativas

- Mantener `Vertical 2` como prioridad funcional de la demo.
- Preservar `Subir PDF`.
- Mantener fallbacks controlados para no romper UX si el backend falla.
- Usar el Canonical Event Model como contrato interno.
- Calcular metricas desde datos canonicos, no desde payloads crudos del provider.
- Persistir historial con el patron existente: almacenamiento local + historial backend cuando aplique.

## Patrones actuales a respetar

- `Vertical2Page` orquesta tabs, carga inicial, historial, filtros y seleccion.
- `eventDataApi.ts` concentra llamadas HTTP y estados de degradacion.
- `eventSelectors.ts`, `openEventAnalytics.ts`, `proprietaryMetrics.ts` y `formatters.ts` concentran logica derivada.
- `EventDataContext` mantiene provider, match, selecciones y resultado actual.
- `ApiFiltersPanel`, `ProcessedHistoryPanel`, `EventHistoryPanel`, `AiCoachPanel` y visualizaciones viven desacoplados de la capa HTTP.

## Si agregás un provider

- Crear o extender la capa de ingestion/adaptacion.
- Mapear todo al Canonical Event Model.
- Validar faltantes y errores sin asumir campos fijos.
- Exponer status claro: `api`, `mock` o `error`.
- Si no hay coordenadas o datos espaciales, degradar visualizaciones de forma explicita.

## Si tocás PDF

- No romper la lectura actual de `uploadPdfReport`.
- Mantener visible el estado de parser, extraccion y mensajes de ingestion.
- Si agregas nuevos datos normalizados, exponerlos sin acoplar la UI a estructuras inestables.
- Preservar radar/cancha tactica e insights iniciales.

## Si tocás historial

- Mantener compatibilidad entre historial local y persistido.
- No asumir que el resultado actual siempre viene del ultimo request.
- Respetar la diferenciacion de origen mostrada en `sourceLabel`.

## Si tocás metricas o insights

- Documentar formula y proposito tactico en el cambio si afecta comportamiento.
- Manejar `No aplica`, arreglos vacios y selecciones parciales sin romper UI.
- Verificar impacto en `MatchSummary`, `MetricsSection`, `PlayerSpotlight`, `VisualizationGrid` e `InsightsList`.

## Checklist antes de cerrar

- El flujo PDF sigue funcionando.
- El flujo API Event Data sigue funcionando.
- Los fallbacks siguen mostrando mensajes utiles.
- El contexto y las selecciones no quedan inconsistentes.
- Historial local y persistido siguen reabriendo partidos.
- AI Coach no se rompe por cambios laterales.

## Ejemplos de uso

- "Quiero sumar un provider nuevo en Vertical 2."
- "Necesito tocar el flujo de historial persistido."
- "Hay que ajustar metricas e insights del analisis por API."
