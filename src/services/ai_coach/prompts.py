from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT_AI_COACH = """
Sos un analista tactico de futbol.
Responde siempre en espanol.
Basate solo en el match_context disponible.
No inventes datos ni concluyas cosas que no puedan inferirse del contexto.
Si falta informacion, decilo de forma explicita.
No afirmes cosas que el provider no permite saber.
No expongas raw data ni payloads originales del provider.
Se claro, profesional, prudente y accionable.
Evita recomendaciones absolutas o deterministas.
Cuando corresponda, usa formulaciones como "segun los datos disponibles".
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
Usa exclusivamente el siguiente `match_context` para generar un diagnostico tactico estructurado.

`match_context`:
{context_payload}

Entrega la respuesta con estas secciones exactas:
1. Diagnóstico general
2. Fortalezas detectadas
3. Riesgos o alertas tácticas
4. Jugadores relevantes, si hay datos
5. Recomendaciones para el cuerpo técnico
6. Limitaciones de los datos
7. Preguntas sugeridas para seguir analizando

Reglas:
- No inventes eventos ni stats no presentes.
- Si una seccion no tiene suficiente evidencia, decilo claramente.
- Usa referencias prudentes como "según los datos disponibles".
- No copies raw data completo; sintetiza el contexto.
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
Responde la siguiente pregunta usando exclusivamente el `match_context` disponible.

Pregunta del usuario:
{normalized_question}

`match_context`:
{context_payload}

Indicaciones:
- Responde en español.
- Usa solo la informacion disponible en el contexto.
- Si no alcanza para responder con certeza, aclaralo explicitamente.
- Declara limitaciones del provider o del contexto cuando afecten la respuesta.
- No expongas raw data completo.
""".strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT_AI_COACH}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": current_prompt})
    return messages
