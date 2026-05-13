# Contexto de Producto

## Qué es Tactical Intelligence Platform
Tactical Intelligence Platform es una plataforma de análisis táctico orientada a fútbol que transforma datos técnicos en lectura accionable para analistas, scouting y cuerpo técnico.

## Objetivo del producto
Reducir el tiempo entre "tener datos" y "tomar decisiones", entregando métricas tácticas, visualizaciones sobre cancha e insights comprensibles para demo y operación.

## Visión
Evolucionar de MVP en Streamlit hacia una plataforma de inteligencia táctica escalable, con frontend y backend desacoplados, persistencia robusta y motor analítico extensible por provider.

## Data provider vs Tactical Intelligence Layer
- **Data provider**: fuente de eventos crudos (ejemplo: StatsBomb Open Data).
- **Tactical Intelligence Layer**: transformación de esos eventos en modelo canónico, métricas propietarias, visualizaciones e insights.

## Enfoque actual MVP/demo
- Prioridad en velocidad de iteración y estabilidad de demo.
- Arquitectura modular en Streamlit.
- Persistencia local liviana para reutilizar partidos sin redescarga.

## Verticales principales
- **Vertical 1**: Tracking Intelligence / Computer Vision.
- **Vertical 2**: Event Intelligence / Event Data.

## Principio clave
El valor no está en la data cruda, sino en cómo se transforma en métricas tácticas, visualizaciones e insights accionables.
