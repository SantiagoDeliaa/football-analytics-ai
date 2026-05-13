# Arquitectura Actual

## Estado general
La arquitectura actual es un prototipo modular sobre Streamlit. Es funcional para MVP/demo, pero no representa la arquitectura enterprise final.

## Entrypoint principal
- `app.py` es el punto de entrada principal actual.

## Routing actual
- Routing custom basado en `st.session_state`.
- Home selecciona vertical activa y redirige por estado.

## Estructura relevante
- `src/verticals/`: Home, Vertical 1 y Vertical 2.
- `src/services/`: ingesta, normalización, métricas, insights y visualizaciones.
- `src/services/storage/`: persistencia local SQLite + JSON.
- `src/utils/ui/`: helpers visuales y tema.
- `data/`: base SQLite y payloads persistidos.

## Arquitectura conceptual
```text
Home
├── Vertical 1 — Computer Vision
└── Vertical 2 — Event Data
    ├── Subir PDF
    └── Datos por API
        ├── Provider Ingestion
        ├── Canonical Event Model
        ├── Metrics Engine
        ├── Insights Engine
        ├── Tactical Visualizations
        └── Local Persistence
```

## Nota de alcance
- Es un monolito modular en Streamlit.
- A futuro puede migrar a arquitectura con frontend y backend desacoplados.
