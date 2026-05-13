from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT_AI_COACH = """
Sos un analista tactico de futbol orientado a usuario final.
Responde siempre en espanol.
Basate solo en el match_context disponible.
No inventes datos ni concluyas cosas que no puedan inferirse del contexto.
Si falta informacion, decilo de forma explicita.
No afirmes cosas que el provider no permite saber.
No expongas raw data ni payloads originales del provider.
No menciones nombres de variables internas, nombres de indices tecnicos, nombres de campos del sistema ni estructura del match_context.
Traduci cualquier metrica interna a conceptos de futbol comprensibles para el usuario final.
Prioriza interpretacion tactica antes que nomenclatura tecnica.
Si una metrica aparece en el contexto, explicala con lenguaje natural. Por ejemplo:
- field_tilt_index -> dominio territorial
- directness_index -> verticalidad del juego
- progressive_threat_index -> amenaza ofensiva progresiva
- recovery_height_index -> altura de recuperacion
- shot_quality_index -> calidad de remate
- player_influence_score -> influencia del jugador
Si faltan coordenadas u otras capacidades del provider, explicalo de forma natural. Por ejemplo:
- "Este proveedor no aporta coordenadas detalladas de los eventos, por lo que el analisis espacial es mas limitado."
Se claro, profesional, prudente, accionable y natural.
Responde como asistente tactico, no como documentacion tecnica ni como desarrollador.
Evita recomendaciones absolutas o deterministas.
Cuando corresponda, usa formulaciones como "segun los datos disponibles", "los indicadores sugieren" o "se observa que".
Evita enumerar demasiados numeros. Si incluis un numero, acompanalo con una interpretacion simple.
""".strip()

MAX_HISTORY_MESSAGES = 6


def _serialize_match_context(match_context: dict[str, Any]) -> str:
    return json.dumps(match_context or {}, ensure_ascii=False, indent=2, sort_keys=True)


def _normalize_history(conversation_history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    history = conversation_history or []
    normalized: list[dict[str, str]] = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if role == "system":
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def build_diagnosis_prompt(match_context: dict[str, Any]) -> list[dict[str, str]]:
    context_payload = _serialize_match_context(match_context)
    user_prompt = f"""
Usa exclusivamente el siguiente contexto tactico estructurado para generar un diagnostico tactico claro, amigable y orientado a usuario final.

Contexto tactico disponible:
{context_payload}

Entrega la respuesta con estas secciones exactas:
1. Resumen general del partido
2. Aspectos positivos
3. Puntos a mejorar
4. Jugadores destacados
5. Recomendaciones para el cuerpo técnico
6. Limitaciones de los datos

Reglas:
- No inventes eventos ni stats no presentes.
- Si una seccion no tiene suficiente evidencia, decilo claramente.
- Usa referencias prudentes como "segun los datos disponibles".
- No copies raw data completo; sintetiza el contexto.
- No menciones nombres de variables internas ni etiquetas tecnicas como field_tilt_index o directness_index.
- Si hablas de metricas, traducilas a conceptos de futbol comprensibles.
- Prioriza interpretacion clara y comercial antes que terminologia tecnica.
""".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT_AI_COACH},
        {"role": "user", "content": user_prompt},
    ]


def build_question_prompt(
    match_context: dict[str, Any],
    user_question: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    context_payload = _serialize_match_context(match_context)
    normalized_question = str(user_question or "").strip()
    history_messages = _normalize_history(conversation_history)
    current_prompt = f"""
Responde la siguiente pregunta usando exclusivamente el contexto tactico disponible.

Pregunta del usuario:
{normalized_question}

Contexto tactico disponible:
{context_payload}

Indicaciones:
- Responde en español.
- Usa solo la informacion disponible en el contexto.
- Responde con lenguaje claro, profesional y simple.
- No menciones nombres de variables internas, indices tecnicos ni nombres de campos del sistema.
- Si una metrica interna aparece en el contexto, explicala como concepto futbolistico comprensible.
- Si el usuario pregunta por metricas propietarias, explicalas como dominio territorial, verticalidad del juego, amenaza ofensiva progresiva, altura de recuperacion, calidad de remate e influencia del jugador.
- Si no alcanza para responder con certeza, aclaralo explicitamente.
- Declara limitaciones del provider o del contexto cuando afecten la respuesta en lenguaje natural.
- No expongas raw data completo.
- Prioriza interpretacion tactica antes que nomenclatura tecnica.
""".strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT_AI_COACH}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": current_prompt})
    return messages
