---
name: "front-tip-vertical1-flow"
description: "Guia del flujo `Vertical 1` en `front-tip`. Invocá cuando cambies Computer Vision, jobs, polling, configuracion avanzada, scouting o exportes."
---

# Front Tip Vertical 1 Flow

## Objetivo

Modificar `Vertical 1` manteniendo estable el flujo de procesamiento por jobs, la UX de configuracion avanzada y la salida demo/controlada cuando el backend no responde.

## Cuándo invocarla

Invocá esta skill cuando:

- trabajes en `src/pages/Vertical1Page.tsx`
- modifiques componentes de `src/components/vertical1`
- toques `src/services/computerVisionApi.ts`
- cambies validaciones de fuente, assets o configuracion
- ajustes polling, estado del job, timeline o exportes

## Reglas operativas

- No tocar `Vertical 1` salvo necesidad real o instruccion explicita.
- Mantener cambios acotados y reversibles.
- Preservar comportamiento funcional actual.
- Si el backend no esta disponible, mantener la salida demo estable donde ya exista fallback.

## Flujo actual

Mantener esta secuencia:

`Video source/config/assets -> create job -> poll job status -> completed result -> tabs y visuales`

## Patrones a respetar

- `Vertical1Page` concentra la orquestacion general.
- `VideoSourcePanel` y `ProcessingConfigPanel` manejan entrada y configuracion.
- Las validaciones de entrada usan `zod` y chequeos complementarios para assets custom.
- El polling consulta `getComputerVisionJob` cada 2 segundos mientras el job esta `queued` o `running`.
- La UI muestra `LoadingState` y `ErrorState` segun creacion del job, polling o errores finales.
- Los modulos pesados, como `TimelineCharts`, se cargan con `lazy` + `Suspense`.

## Si tocás fuentes o configuracion

- Mantener compatibilidad con `upload` y `soccernet`.
- No romper preview local del archivo subido.
- Si agregas una opcion de config, asegurate de que viaje dentro de `config` y no por props dispersas.
- Si el cambio requiere assets custom, validar presencia explicita antes del submit.

## Si tocás servicios de Computer Vision

- Mantener el contrato multipart actual para archivos, `soccernet_path`, `config` y modelos custom.
- No eliminar el fallback demo existente salvo instruccion explicita.
- Si ajustas errores HTTP, devolver mensajes claros para la UI.

## Si tocás tabs o resultados

- Preservar el orden mental del flujo: video, stats, charts, exports, scouting, interpretation, possession.
- No mostrar tabs dependientes de `result` antes de tener un resultado valido.
- Si agregas una nueva vista, ubicarla como derivacion del `result`, no de payloads intermedios.

## Checklist antes de cerrar

- Se puede crear el job correctamente.
- El polling se detiene al completar o fallar.
- Los errores de validacion siguen siendo claros.
- El preview local no filtra recursos; si tocas blobs, revocalos.
- Las tabs siguen dependiendo de `result`.
- El fallback demo sigue permitiendo recorrer la experiencia.

## Ejemplos de uso

- "Necesito sumar una opcion nueva al panel de configuracion."
- "Hay que tocar el polling del job de Computer Vision."
- "Quiero extender exports o scouting en Vertical 1."
