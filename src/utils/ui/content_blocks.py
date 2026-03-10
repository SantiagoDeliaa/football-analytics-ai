INTERPRETATION_MARKDOWN = """
# Interpretación de Métricas (Beta)

**Nota:** el sistema está en beta. Cuando la señal base (homografía/tracking) baja, algunas métricas pueden mostrarse como **N/A** o como **aproximadas**. La idea es mostrar tendencias y patrones tácticos, no un dato “perfecto” frame a frame.

---

## 1) Compactación (Profundidad y Ancho)

**Qué mide**
- **Profundidad (m):** largo del bloque del equipo (distancia entre la última línea y el jugador más adelantado).
- **Ancho (m):** apertura lateral del equipo.

**Cómo leer el gráfico**
- **Picos** de profundidad = el equipo **se estira** (transición / ataque rápido).
- **Valles** de profundidad = equipo **compacto** (bloque defensivo).
- **Ancho alto** = ocupa bandas / abre el campo.
- **Ancho bajo** = cierra espacios interiores / defensa más junta.

**Ejemplos futboleros**
- Profundidad ↑ + Ancho ↑ → **contraataque / equipo lanzado**
- Profundidad ↓ + Ancho ↓ → **repliegue + bloque compacto**
- Ancho ↑ pero profundidad estable → **circulación y amplitud** (ataque posicional)
- Profundidad ↑ con ancho ↓ → **juego directo por carril central**

**Uso en Scouting (qué podés detectar)**
- Equipos **verticales** vs **posicionales**
- Nivel de **disciplina táctica** (oscilación vs estabilidad)
- Momentos de **transición** (cuándo se estiran y por qué)
- “Firma” de estilo: bloque alto/medio/bajo y amplitud habitual

---

## 2) Línea Defensiva

**Qué mide**
- Altura del bloque defensivo (metros). Indica si el equipo defiende alto o bajo.

**Cómo leer**
- Línea alta sostenida → **presión alta** / achique / intención de recuperar rápido.
- Línea baja sostenida → **bloque bajo** / repliegue / protección del área.
- Subidas y bajadas bruscas → equipo **reactivo**, o partido “partido”.

**Ejemplos futboleros**
- Línea defensiva sube mientras compactación baja → **presión + equipo junto**
- Línea defensiva baja y profundidad alta → **equipo largo** (riesgo entre líneas)

**Uso en Scouting**
- Detectar si el equipo juega al **offside** o defiende en **bloque bajo**
- Identificar vulnerabilidad a **pelotas largas**
- Analizar coherencia: línea alta sin compactación suele ser riesgo

---

## 3) Formación Detectada (aproximada)

**Qué representa**
- La formación “más frecuente” estimada por clustering de posiciones.

**Cómo leer**
- Se muestra como **aprox** porque en broadcast:
  - faltan jugadores por oclusión,
  - el equipo cambia de fase (defiende 4-4-2 y ataca 2-3-5, por ejemplo).

**Uso en Scouting**
- Sistema base (4-3-3 / 4-2-3-1 / 3-5-2)
- Cambios estructurales: “¿se transforma en ataque?”
- Señales de entrenador: extremos altos, carrileros, doble 5, etc.

---

## 4) Velocidad y Distancia

**Qué mide**
- Distancia total (y promedio por jugador).
- Velocidad máxima y sprints (aproximados).

**Cómo leer**
- Distancia ↑ → equipo con más **actividad** (presión, ida y vuelta).
- Sprints ↑ → partido más **vertical** o presión intensa.

**Nota Beta**
- Picos extremos pueden venir de tracking. En demo la velocidad máxima se muestra dentro de rangos humanos plausibles y se marca si fue “cappeada”.

**Uso en Scouting**
- Intensidad del equipo y del ritmo del partido
- Comparar estilos: presión constante vs bloque y salida
- Detectar tramos de alta exigencia (minutos “calientes”)

---

## 5) Heatmap (Mapa de Calor)

**Qué muestra**
- Zonas del campo donde el equipo **más ocupó** (por presencia de jugadores).

**Cómo leer**
- Zonas “calientes” = mayor permanencia / control territorial
- Banda cargada = preferencia por sector / ataques repetidos
- Distribución alta = presión territorial
- Distribución baja = repliegue / defensa cerca del área

**Uso en Scouting**
- Identificar banda preferida y patrones de ocupación
- Ver si el equipo juega ancho o se centraliza
- Contextualizar: “¿dónde intenta progresar y dónde recupera?”
"""
