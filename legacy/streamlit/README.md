# Streamlit Legacy

Este directorio conserva la UI historica basada en `Streamlit`.

## Estado

- No es el frontend principal del producto.
- No debe usarse para nuevas features.
- Se conserva solo como referencia temporal y compatibilidad limitada.
- La arquitectura oficial es `React + TypeScript + Vite` en `front-tip/` consumiendo `FastAPI` en `api/`.

## Uso

Si necesitás abrir la UI legacy de forma puntual:

```bash
python -m pip install -r requirements-legacy.txt
streamlit run legacy/streamlit/app.py
```

## Alcance

- Puede servir para comparar salidas durante refactors controlados.
- No define el flujo principal de producto.
- Las nuevas integraciones de providers deben exponerse por `FastAPI` y consumirse desde `React`.

## Retiro futuro

Este legacy debe eliminarse cuando `React + FastAPI` cubra por completo los flujos que aun se usan solo como referencia en Streamlit.
