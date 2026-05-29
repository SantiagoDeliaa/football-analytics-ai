# Validacion de Integracion

## Objetivo
Dejar trazable el estado de integracion entre:

- codigo heredado desde `main`
- backend FastAPI actual
- frontend React `front-tip`
- persistencia local y contexto operativo para agentes

## Alcance validado

### Contexto y documentacion
- `README.md` actualizado con arquitectura, arranque y puntos de entrada reales.
- `docs/ARCHITECTURE.md` actualizado a arquitectura principal `React + FastAPI + src`, con Streamlit archivado como legacy.
- `docs/LOCAL_PERSISTENCE.md` alineado con rutas reales de persistencia.
- `tests/conftest.py` agregado para bootstrap comun del repo en pruebas Python.
- Fixtures minimos restaurados en `data/` para conservar contexto de persistencia local, sin versionarlos en Git.

### Vertical 1
- El frontend nuevo consume `/api/v1/computer-vision/*`.
- El backend mantiene integracion con el dominio existente en `src/controllers/`.
- La suite incluye regresiones para render basico y estados seguros.

### Vertical 2
- `StatsBomb Open Data` integrado en backend y frontend.
- `API-Football` integrado en backend y frontend.
- Flujo PDF integrado con parser real y visualizacion tactica moderna.
- Historial local soportado en backend y frontend.
- El modelo canonico, metricas e insights siguen desacoplados del provider.

### UX y UI
- La UI nueva se apoya en el design language existente oscuro/premium.
- Se preserva consistencia visual entre Home, Vertical 1 y Vertical 2.
- Hay manejo de loading, error states, lazy loading y prefetch para mejorar navegacion.
- El frontend incluye componentes dedicados para visualizaciones, historial y estados vacios.

## Verificacion recomendada

### Backend
```bash
pytest
```

### Frontend
```bash
cd front-tip
npm run test
npm run build
```

### Integracion docker
```bash
docker compose up --build
```

## Checklist funcional

- Home renderiza y navega a ambas verticales.
- Vertical 1 renderiza paneles principales y tolera estados sin procesamiento.
- Vertical 2 renderiza filtros, resumen, metricas, insights y visualizaciones.
- Cambio de provider resetea filtros y recarga datos correctamente.
- Errores del backend se traducen a mensajes legibles en frontend.
- Si un provider no entrega coordenadas, la UI evita forzar mapas de cancha.
- El historial local permite reabrir partidos procesados.

## Checklist de responsividad

- Validar `Home`, `Vertical 1` y `Vertical 2` en anchos moviles, tablet y desktop.
- Verificar que grids, cards y charts no se superpongan.
- Confirmar que tabs, selects y tablas mantienen legibilidad en pantallas chicas.
- Revisar que los contenedores de Plotly/Chart.js respeten altura y overflow.

## Riesgos residuales

- La responsividad visual fina depende tambien de validacion en navegador real.
- El flujo de `API-Football` puede variar segun limites del plan y cobertura del provider.
- El pipeline real de Computer Vision puede requerir modelos/entorno local para validacion completa mas alla de tests.

## Criterio de aceptacion

La integracion se considera consistente si:

- frontend nuevo y backend actual levantan juntos sin errores de contrato
- Vertical 1 y Vertical 2 comparten look & feel y navegacion clara
- las capas de dominio heredadas siguen accesibles y documentadas
- cualquier agente puede reconstruir el contexto del proyecto leyendo `README.md`, `docs/` y `.trae/`
