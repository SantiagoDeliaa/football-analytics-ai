---
name: "front-tip-ui-patterns"
description: "Guia para crear o refactorizar UI en `front-tip`. Invocá cuando trabajes en paginas/componentes React, rutas, estados de carga/error o look & feel."
---

# Front Tip UI Patterns

## Objetivo

Aplicar patrones consistentes de arquitectura, composicion de UI y experiencia de usuario dentro de `front-tip`.

## Cuándo invocarla

Invocá esta skill cuando:

- agregues o refactorices paginas en `src/pages`
- crees componentes en `src/components`
- toques layout, router, tabs, loading states o error states
- tengas que mantener el look & feel oscuro/premium del frontend
- quieras decidir en qué carpeta vive una pieza nueva

## Reglas del frontend

- Mantener la UI en español.
- Preservar cambios chicos y seguros.
- No mezclar una refactorizacion grande con una feature grande en el mismo cambio.
- No romper la demo actual.
- No tocar `Vertical 1` si la tarea es solo de `Vertical 2`, salvo instruccion explicita.

## Estructura esperada

Usar esta separacion:

- `src/pages`: orquestacion de pantallas
- `src/components/common`: piezas base reutilizables
- `src/components/layout`: layout principal
- `src/components/vertical1`: componentes de dominio de Computer Vision
- `src/components/vertical2`: componentes de dominio de Event Data
- `src/services`: acceso a API y adaptacion de payloads
- `src/utils`: helpers puros, calculos y selectores
- `src/hooks`: hooks reutilizables, como `useAsync`
- `src/app`: router, context global y preload de modulos

## Patrones a respetar

- Mantener las paginas como capas de orquestacion; mover UI reutilizable a componentes.
- Mantener logica de red fuera de componentes; usar `src/services`.
- Mantener calculos puros y selectores en `src/utils`.
- Reutilizar `LoadingState`, `ErrorState`, `Tabs`, `MetricCard` y layout existente antes de crear variantes nuevas.
- Si una seccion es pesada, preferir `lazy` + `Suspense` + prefetch desde `modulePreload`.
- Si una vista requiere estado compartido de Event Data, integrarla via `EventDataContext` y no via props drilling innecesario.

## Convenciones de UI

- Conservar el look oscuro con tarjetas, bordes suaves, gradientes sutiles y tipografia consistente.
- Priorizar lectura tactica rapida: resumen arriba, visuales despues, debug al final.
- Mostrar estados de carga y error explicitamente; no dejar pantallas vacias.
- En degradacion o fallback, informar el origen sin romper el flujo de la demo.
- Evitar acoplar decisiones del provider o del backend directamente a componentes visuales.

## Checklist antes de cerrar

- La pieza nueva esta en la carpeta correcta.
- Los textos visibles quedan en español.
- Los estados `loading`, `empty`, `error` y `success` estan cubiertos.
- No se duplico logica que ya exista en `services`, `utils` o `common`.
- La navegacion y el layout se mantienen coherentes con `Home`, `Vertical1` y `Vertical2`.

## Ejemplos de uso

- "Quiero agregar una nueva seccion visual en `Vertical2Page`."
- "Necesito refactorizar un componente de `src/components/common`."
- "Hay que sumar una ruta nueva sin romper lazy loading."
