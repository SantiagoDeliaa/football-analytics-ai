from __future__ import annotations

import re
from typing import Any

import streamlit as st

from src.services.ai_coach import answer_coach_question
from src.services.ai_coach import build_match_context
from src.services.ai_coach import generate_tactical_diagnosis
from src.services.ai_coach import get_ai_coach_config_status
from src.services.ai_coach import get_suggested_questions

MAX_CONVERSATION_HISTORY = 6


def _sanitize_key_fragment(value: Any) -> str:
    raw_value = str(value or "none").strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", raw_value)
    sanitized = sanitized.strip("_")
    return sanitized or "none"


def build_ai_coach_state_key(
    prefix: str,
    provider: Any,
    match_id: Any,
    selected_team: str | None = None,
    selected_player: str | None = None,
) -> str:
    return "_".join(
        [
            prefix,
            _sanitize_key_fragment(provider),
            _sanitize_key_fragment(match_id),
            _sanitize_key_fragment(selected_team),
            _sanitize_key_fragment(selected_player),
        ]
    )


def _has_pitch_coordinates(canonical_events: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(event.get("x"), (int, float)) and isinstance(event.get("y"), (int, float))
        for event in canonical_events
    )


def _has_xg_data(canonical_events: list[dict[str, Any]], metrics: dict[str, Any]) -> bool:
    if any((event.get("xG") or 0) not in {0, 0.0, None} for event in canonical_events):
        return True
    return bool((metrics or {}).get("total_xg"))


def build_provider_capabilities(
    provider: Any,
    canonical_events: list[dict[str, Any]] | None,
    metrics: dict[str, Any] | None,
    raw_payload: Any,
) -> dict[str, bool]:
    normalized_events = canonical_events or []
    normalized_metrics = metrics or {}
    provider_key = str(provider or "").strip().lower()
    has_coordinates = _has_pitch_coordinates(normalized_events)
    has_xg = _has_xg_data(normalized_events, normalized_metrics)

    if provider_key == "statsbomb":
        return {
            "has_event_coordinates": has_coordinates,
            "has_lineups": False,
            "has_team_stats": False,
            "has_player_stats": False,
            "has_xg": has_xg,
            "has_event_timeline": bool(normalized_events),
        }

    if provider_key == "api_football":
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        return {
            "has_event_coordinates": has_coordinates,
            "has_lineups": bool(payload.get("lineups")),
            "has_team_stats": bool(payload.get("statistics")),
            "has_player_stats": bool(payload.get("players")),
            "has_xg": has_xg,
            "has_event_timeline": bool(payload.get("events")) or bool(normalized_events),
        }

    return {
        "has_event_coordinates": has_coordinates,
        "has_lineups": False,
        "has_team_stats": False,
        "has_player_stats": False,
        "has_xg": has_xg,
        "has_event_timeline": bool(normalized_events),
    }


def build_match_metadata_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": result.get("provider"),
        "match_id": result.get("match_id"),
        "competition_name": result.get("competition_name"),
        "season_name": result.get("season_name"),
        "home_team": result.get("home_team"),
        "away_team": result.get("away_team"),
        "match_date": result.get("match_date"),
    }


def _limit_conversation_history(chat_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in chat_history[-MAX_CONVERSATION_HISTORY:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _append_chat_history(chat_key: str, user_question: str, answer: str) -> None:
    history = list(st.session_state.get(chat_key, []))
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": answer})
    st.session_state[chat_key] = history


def _run_ai_coach_question(
    match_context: dict[str, Any],
    user_question: str,
    chat_key: str,
) -> str | None:
    normalized_question = str(user_question or "").strip()
    if not normalized_question:
        return "La pregunta del usuario no puede estar vacía."

    conversation_history = _limit_conversation_history(list(st.session_state.get(chat_key, [])))
    with st.spinner("Analizando el partido..."):
        response = answer_coach_question(
            match_context,
            normalized_question,
            conversation_history=conversation_history,
        )
    if not response.get("ok"):
        return str(response.get("error") or "No se pudo responder la pregunta del AI Coach.")

    _append_chat_history(chat_key, normalized_question, str(response.get("answer") or ""))
    return None


def render_ai_coach_panel(
    result: dict[str, Any],
    metrics: dict[str, Any],
    insights: list[str],
    selected_team: str,
    selected_player: str,
    show_technical_info: bool = False,
) -> None:
    provider = str(result.get("provider") or "unknown")
    match_id = result.get("match_id")
    canonical_events = result.get("canonical_events", []) or []
    raw_payload = result.get("raw_payload", {})
    provider_capabilities = build_provider_capabilities(
        provider=provider,
        canonical_events=canonical_events,
        metrics=metrics,
        raw_payload=raw_payload,
    )
    match_context = build_match_context(
        provider=provider,
        match_metadata=build_match_metadata_from_result(result),
        canonical_events=canonical_events,
        metrics=metrics,
        insights=insights,
        selected_team=selected_team,
        selected_player=selected_player,
        provider_capabilities=provider_capabilities,
        raw_summary=None,
    )
    state_scope = {
        "provider": provider,
        "match_id": match_id,
        "selected_team": selected_team,
        "selected_player": selected_player,
    }
    diagnosis_key = build_ai_coach_state_key("ai_coach_diagnosis", **state_scope)
    chat_key = build_ai_coach_state_key("ai_coach_chat", **state_scope)
    input_key = build_ai_coach_state_key("ai_coach_input", **state_scope)
    clear_input_key = build_ai_coach_state_key("ai_coach_clear_input", **state_scope)
    config_status = get_ai_coach_config_status()
    is_configured = bool(config_status.get("configured"))

    st.markdown("### AI Tactical Coach")
    st.caption("Consultá el partido con contexto táctico generado por TIP.")
    if not is_configured:
        st.warning("Falta configurar AI_COACH_API_KEY para activar el AI Tactical Coach.")

    if st.button(
        "Generar diagnóstico táctico",
        key=build_ai_coach_state_key("ai_coach_generate", **state_scope),
        use_container_width=True,
        disabled=not is_configured,
    ):
        if not is_configured:
            st.warning("Falta configurar AI_COACH_API_KEY para activar el AI Tactical Coach.")
        else:
            with st.spinner("Analizando el partido..."):
                response = generate_tactical_diagnosis(match_context)
            if response.get("ok"):
                st.session_state[diagnosis_key] = str(response.get("diagnosis") or "")
            else:
                st.error(str(response.get("error") or "No se pudo generar el diagnóstico táctico."))

    saved_diagnosis = str(st.session_state.get(diagnosis_key, "") or "").strip()
    if saved_diagnosis:
        st.markdown("#### Diagnóstico táctico")
        st.markdown(saved_diagnosis)

    st.markdown("#### Preguntas sugeridas")
    suggested_questions = get_suggested_questions(match_context)
    for index, question in enumerate(suggested_questions):
        if st.button(
            question,
            key=build_ai_coach_state_key(f"ai_coach_question_{index}", **state_scope),
            use_container_width=True,
            disabled=not is_configured,
        ):
            if not is_configured:
                st.warning("Falta configurar AI_COACH_API_KEY para activar el AI Tactical Coach.")
            else:
                error_message = _run_ai_coach_question(match_context, question, chat_key)
                if error_message:
                    st.error(error_message)

    st.markdown("#### Chat contextual")
    if st.session_state.pop(clear_input_key, False):
        st.session_state[input_key] = ""
    question_input = st.text_input(
        "Preguntale algo al AI Coach...",
        key=input_key,
        placeholder="Ejemplo: ¿Dónde generó más peligro el equipo?",
    )
    if st.button(
        "Preguntar",
        key=build_ai_coach_state_key("ai_coach_submit", **state_scope),
        use_container_width=True,
        disabled=not is_configured,
    ):
        if not is_configured:
            st.warning("Falta configurar AI_COACH_API_KEY para activar el AI Tactical Coach.")
        else:
            error_message = _run_ai_coach_question(match_context, question_input, chat_key)
            if error_message:
                st.error(error_message)
            else:
                st.session_state[clear_input_key] = True
                st.rerun()

    chat_history = list(st.session_state.get(chat_key, []))
    if chat_history:
        for message in chat_history:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role == "user":
                st.markdown(f"**Usuario:** {content}")
            elif role == "assistant":
                st.markdown(f"**AI Coach:** {content}")
    elif not is_configured:
        st.info("Configurá AI_COACH_API_KEY para habilitar el diagnóstico y el chat contextual.")
    else:
        st.info("Generá un diagnóstico o hacé una pregunta para comenzar el análisis contextual.")

    if show_technical_info:
        with st.expander("Match Context enviado al AI Coach"):
            st.json(match_context)
