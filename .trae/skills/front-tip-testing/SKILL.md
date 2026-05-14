---
name: "front-tip-testing"
description: "Guia de testing para `front-tip`. Invocá cuando agregues hooks, componentes con logica, paginas orquestadoras o necesites cobertura con Vitest y React Testing Library."
---

# Front Tip Testing

## Objetivo

Agregar pruebas utiles y enfocadas para `front-tip`, alineadas con `Vitest`, `React Testing Library` y el entorno `jsdom`.

## Cuándo invocarla

Invocá esta skill cuando:

- crees un hook reutilizable
- agregues un componente con logica de negocio, estado o efectos
- cambies una pagina orquestadora con comportamiento relevante
- necesites cubrir una regresion de frontend
- quieras decidir si un cambio requiere test o alcanza con validacion manual

## Stack de pruebas

- Runner: `Vitest`
- DOM: `jsdom`
- Utilidades: `@testing-library/react`, `@testing-library/user-event`, `@testing-library/jest-dom`
- Setup: `src/test/setup.ts`

## Reglas

- Si un componente u hook tiene logica real, agregar o ajustar tests.
- Evitar tests de bajo valor que solo repiten implementacion.
- Priorizar happy path, estados de error y transiciones relevantes.
- Mockear servicios o storage cuando el comportamiento que importa sea de la UI.
- Si el cambio esta concentrado en `utils`, preferir tests unitarios puros.

## Qué conviene testear en este repo

- Formularios y validaciones de upload/configuracion.
- Estados `loading`, `error`, `empty` y `success`.
- Interacciones de tabs, filtros y seleccion.
- Render condicional segun `result`, historial o fallback.
- Helpers puros de metricas, selectores y visualizaciones derivadas.

## Qué no conviene sobre-testear

- Clases utilitarias de Tailwind.
- Marcado estatico sin logica.
- Detalles internos de implementacion que el usuario no observa.
- Duplicacion de cobertura que ya exista en tests cercanos.

## Estrategia por tipo de cambio

- Para `services`: mockear red y validar contratos o degradacion.
- Para `pages`: cubrir orquestacion visible y decisiones de render.
- Para `components/vertical2`: cubrir filtros, historial, errores y callbacks.
- Para `components/vertical1`: cubrir validaciones, submit y estados del job cuando haya logica propia.
- Para `utils`: usar entradas pequeñas y expectativas claras.

## Comandos utiles

```bash
npm run test
npm run test -- --runInBand
```

Si el cambio fue puntual, correr el archivo de prueba mas cercano primero y despues el suite relevante.

## Checklist antes de cerrar

- El test cubre un riesgo real de regresion.
- Los mocks representan el contrato actual y no uno inventado.
- Las aserciones observan comportamiento visible o salidas publicas.
- No queda acoplado a detalles fragiles de implementacion.
- El archivo nuevo respeta el patron `*.test.ts` o `*.test.tsx`.

## Ejemplos de uso

- "Agregué logica nueva a `PdfUploadForm` y quiero cubrirla."
- "Necesito testear un hook de carga asincronica."
- "Quiero validar que una pagina muestre fallback y error correctamente."
