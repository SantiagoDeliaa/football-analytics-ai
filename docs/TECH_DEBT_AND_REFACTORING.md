# Deuda Tecnica y Refactoring

## Estado actual

- UI principal: `React + TypeScript + Vite` en `front-tip/`
- Backend principal: `FastAPI` en `api/`
- Core analitico: `src/`
- Legacy archivado: `legacy/streamlit/`

## Decision del refactor

- `Streamlit` deja de ser parte del flujo principal.
- Se elimina `streamlit` de `requirements.txt`.
- Se crea `requirements-legacy.txt` para ejecutar el legacy solo cuando haga falta.
- Se archivan los entrypoints legacy en `legacy/streamlit/`.
- Se extraen helpers puros desde la UI legacy a `src/services/presentation/`.

## Auditoria de archivos legacy

### A. Core analitico reusable

- `src/services/`
- `src/controllers/`
- `src/services/providers/`
- `src/services/canonical_models.py`
- `src/services/presentation/`
- `src/services/open_event_visualizations.py`

### B. UI legacy Streamlit

- `legacy/streamlit/app.py`
- `src/verticals/vertical2.py`
- `src/verticals/vertical2_api_event.py`
- `src/verticals/vertical1.py`
- `src/verticals/vertical1_legacy.py`
- `src/verticals/home.py`
- `src/utils/ui/`

### C. Documentacion a mantener actualizada

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/TECH_DEBT_AND_REFACTORING.md`
- `docs/AGENT_WORKFLOW.md`
- `docs/providers/SPORTMONKS_INTEGRATION.md`

### D. Tests legacy o de compatibilidad

- `tests/test_frontend_regression.py`
- `tests/test_vertical2_sportmonks_ui.py`

Estas pruebas siguen siendo utiles para compatibilidad de la capa legacy, pero no representan la arquitectura principal del frontend moderno.

### E. Dependencias requeridas

- `fastapi`, `uvicorn`, librerias analiticas y de datos: requeridas por el flujo principal
- `streamlit`: dependencia legacy, separada en `requirements-legacy.txt`

## Deuda tecnica vigente

- Aun existen modulos Streamlit en `src/verticals/*` y `src/utils/ui/*`.
- Parte de las regresiones Python siguen validando UI legacy y no componentes React.
- La experiencia provider-agnostic final de Sportmonks todavia no esta expuesta por FastAPI + React.
- Persistencia local y jobs en memoria siguen siendo suficientes para demo, no para produccion multiusuario.

## Riesgos actuales

- Mantener dos superficies de UI en paralelo aumenta costo de mantenimiento.
- Algunas rutas legacy todavia sirven como referencia funcional y no se pueden borrar de golpe sin perder cobertura.
- Computer Vision moderno depende de jobs en memoria; el historial persistido reduce el riesgo pero no reemplaza una cola real.

## Proximo objetivo de refactor

- Seguir moviendo cualquier helper reusable fuera de Streamlit.
- Reducir gradualmente `tests/test_frontend_regression.py` a compatibilidad minima.
- Exponer Match Center de Sportmonks por FastAPI y consumirlo desde React.
- Eliminar definitivamente el legacy cuando React + FastAPI cubran todo el alcance activo.
