from __future__ import annotations

import re
from typing import Any

from src.services.ai_coach.llm_client import call_llm
from src.services.ai_coach.llm_client import get_ai_coach_config_status
from src.services.ai_coach.prompts import build_diagnosis_prompt
from src.services.ai_coach.prompts import build_question_prompt

FALLBACK_SUGGESTED_QUESTIONS = [
    "¿Cómo estuvo el equipo en términos generales?",
    "¿Dónde generó más peligro?",
    "¿Qué debería corregir el cuerpo técnico?",
    "¿Qué jugador fue más influyente?",
    "¿Qué limitaciones tienen estos datos?",
]

VISIBLE_TERM_REPLACEMENTS = {
    r"`?field_tilt_index`?": "dominio territorial",
    r"`?directness_index`?": "verticalidad del juego",
    r"`?progressive_threat_index`?": "amenaza ofensiva progresiva",
    r"`?recovery_height_index`?": "altura de recuperación",
    r"`?shot_quality_index`?": "calidad de remate",
    r"`?player_influence_score`?": "influencia del jugador",
    r"`?match_context`?": "contexto táctico disponible",
    r"`?provider_context`?": "contexto del proveedor",
}


def _validate_match_context(match_context: dict[str, Any] | None) -> str | None:
    if not isinstance(match_context, dict) or not match_context:
        return "No hay match_context disponible para el AI Tactical Coach."
    return None


def _validate_ai_coach_configuration() -> str | None:
    config_status = get_ai_coach_config_status()
    if not bool(config_status.get("configured")):
        return str(config_status.get("message") or "El AI Tactical Coach no está configurado.")
    return None


def _make_ai_coach_response_user_friendly(content: Any) -> str:
    normalized = str(content or "").strip()
    if not normalized:
        return ""

    for pattern, replacement in VISIBLE_TERM_REPLACEMENTS.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(
        r"`?has_event_coordinates`?\s*=\s*false",
        "este proveedor no aporta coordenadas detalladas de los eventos",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"`?has_event_coordinates`?\s*:\s*false",
        "este proveedor no aporta coordenadas detalladas de los eventos",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def generate_tactical_diagnosis(match_context: dict[str, Any] | None) -> dict[str, Any]:
    context_error = _validate_match_context(match_context)
    if context_error:
        return {"ok": False, "diagnosis": "", "error": context_error}

    config_error = _validate_ai_coach_configuration()
    if config_error:
        return {"ok": False, "diagnosis": "", "error": config_error}

    messages = build_diagnosis_prompt(match_context)
    response = call_llm(messages)
    if not response.get("ok"):
        return {
            "ok": False,
            "diagnosis": "",
            "error": str(response.get("error") or "No se pudo generar el diagnóstico táctico."),
        }

    return {
        "ok": True,
        "diagnosis": _make_ai_coach_response_user_friendly(response.get("content")),
        "error": "",
    }


def answer_coach_question(
    match_context: dict[str, Any] | None,
    user_question: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_question = str(user_question or "").strip()
    if not normalized_question:
        return {
            "ok": False,
            "answer": "",
            "error": "La pregunta del usuario no puede estar vacía.",
        }

    context_error = _validate_match_context(match_context)
    if context_error:
        return {"ok": False, "answer": "", "error": context_error}

    config_error = _validate_ai_coach_configuration()
    if config_error:
        return {"ok": False, "answer": "", "error": config_error}

    messages = build_question_prompt(match_context, normalized_question, conversation_history=conversation_history)
    response = call_llm(messages)
    if not response.get("ok"):
        return {
            "ok": False,
            "answer": "",
            "error": str(response.get("error") or "No se pudo responder la pregunta del AI Coach."),
        }

    return {
        "ok": True,
        "answer": _make_ai_coach_response_user_friendly(response.get("content")),
        "error": "",
    }


def get_suggested_questions(match_context: dict[str, Any] | None) -> list[str]:
    if isinstance(match_context, dict):
        questions = match_context.get("suggested_questions")
        if isinstance(questions, list):
            normalized = [str(question).strip() for question in questions if str(question).strip()]
            if normalized:
                return normalized

    return list(FALLBACK_SUGGESTED_QUESTIONS)
