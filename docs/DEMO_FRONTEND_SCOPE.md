# Demo Miércoles — Alcance Real

## Objetivo inmediato
Para la demo con el cliente, el objetivo principal es **migrar la experiencia visual** desde Streamlit hacia el nuevo frontend en React, manteniendo la lógica funcional y la integración actual lo más estables posible.

## Qué significa esto en términos técnicos
- El foco inmediato es **cambiar la UI**, no rediseñar toda la plataforma.
- El backend actual debe seguir funcionando con la lógica existente.
- No es requisito de demo terminar ahora una arquitectura enterprise completa.
- Cualquier cambio de infraestructura debe evaluarse por su impacto en estabilidad y tiempo.

## Decisión de producto para esta etapa
- **Sí** a la migración visual hacia React.
- **Sí** a exponer/adaptar endpoints HTTP necesarios para que React consuma la lógica actual.
- **No** a una reescritura completa del dominio analítico antes de la demo.
- **No** a introducir complejidad operativa innecesaria si no aporta valor visible para el cliente esta semana.

## Alcance recomendado para la demo
- Mostrar Home y navegación moderna en React.
- Mostrar `Vertical 1` y `Vertical 2` en frontend React.
- Mantener el backend actual como motor funcional.
- Priorizar estabilidad del flujo:
  - cargar partido
  - mostrar métricas
  - mostrar insights
  - mostrar visualizaciones
  - ejecutar procesamiento de video o fallback controlado

## Qué no debería bloquear la demo
- Persistencia enterprise de jobs.
- Worker distribuido real.
- Redis / broker de colas.
- PostgreSQL obligatorio.
- Nginx productivo.
- Object storage.

## Riesgos a evitar antes de la demo
- Cambiar demasiadas capas al mismo tiempo.
- Reemplazar componentes del dominio que hoy funcionan.
- Introducir infraestructura nueva sin tiempo suficiente de prueba.
- Mover archivos, jobs y persistencia a una arquitectura nueva sin validación punta a punta.

## Criterio práctico
Si una mejora:
- mejora la UI o la percepción del cliente en la demo, priorizar;
- mejora la robustez sin cambiar demasiado la arquitectura, considerar;
- agrega complejidad operativa para escalar pero no aporta valor visible esta semana, documentar y postergar.
