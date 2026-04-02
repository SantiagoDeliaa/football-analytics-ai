from __future__ import annotations

import re
from typing import Any


def _to_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def _category(score: int) -> str:
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def _get_signal(normalized_payload: dict[str, Any], section: str, key: str) -> float:
    section_data = normalized_payload.get(section, {})
    signals = section_data.get("signals", {})
    try:
        return float(signals.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _get_meta_hits(normalized_payload: dict[str, Any], section: str) -> float:
    meta = normalized_payload.get("meta", {})
    sections_detected = meta.get("sections_detected", {})
    try:
        return float(sections_detected.get(section, 0.0) or 0.0)
    except Exception:
        return 0.0


def _extract_percent(raw_text: str, keyword: str) -> float:
    match = re.search(rf"{keyword}[^\n\r%]{{0,25}}(\d{{1,3}}(?:[.,]\d+)?)\s*%", raw_text, flags=re.IGNORECASE)
    if not match:
        return 50.0
    try:
        value = float(match.group(1).replace(",", "."))
    except Exception:
        return 50.0
    return max(0.0, min(100.0, value))


def calculate_proprietary_metrics(
    normalized_payload: dict[str, Any],
    raw_payload: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    raw_text = str((raw_payload or {}).get("raw_text", "") or "")

    possession = _extract_percent(raw_text, "possession|posesión")
    final_third = _get_signal(normalized_payload, "attack", "final third")
    box_entries = _get_signal(normalized_payload, "attack", "box entries")
    progression = _get_signal(normalized_payload, "build_up", "progression")
    build_up_possession = _get_signal(normalized_payload, "build_up", "possession")
    direct_attack = _get_signal(normalized_payload, "transitions", "direct attack")
    counter = _get_signal(normalized_payload, "transitions", "counter")
    pressing = _get_signal(normalized_payload, "defense", "pressing")
    recoveries = _get_signal(normalized_payload, "defense", "recoveries")
    regain = _get_signal(normalized_payload, "transitions", "regain")
    shots = _get_signal(normalized_payload, "attack", "shots")
    on_target = _get_signal(normalized_payload, "finishing", "on target")
    turnover = _get_signal(normalized_payload, "transitions", "turnover")
    sections_attack = _get_meta_hits(normalized_payload, "attack")
    sections_build_up = _get_meta_hits(normalized_payload, "build_up")

    field_tilt_index = _to_score(
        0.42 * possession
        + 8.0 * final_third
        + 7.0 * box_entries
        + 4.0 * progression
        + 1.5 * sections_attack
    )

    directness_index = _to_score(
        40.0
        + 8.5 * progression
        + 9.0 * direct_attack
        + 7.0 * counter
        + 2.0 * final_third
        - 1.3 * build_up_possession
    )

    pressing_efficiency = _to_score(
        28.0
        + 8.0 * pressing
        + 7.0 * recoveries
        + 8.0 * regain
        + 5.0 * shots
        + 5.0 * on_target
    )

    risk_exposure_score = _to_score(
        24.0
        + 11.0 * turnover
        + 4.0 * build_up_possession
        + 2.0 * sections_build_up
        - 2.2 * pressing
    )

    metrics = {
        "field_tilt_index": {
            "label": "Territorial Control",
            "score": field_tilt_index,
            "description": "Presencia territorial y acciones en zonas avanzadas.",
            "category": _category(field_tilt_index),
        },
        "directness_index": {
            "label": "Verticality",
            "score": directness_index,
            "description": "Qué tan directo progresa el equipo hacia el arco rival.",
            "category": _category(directness_index),
        },
        "pressing_efficiency": {
            "label": "High Press Impact",
            "score": pressing_efficiency,
            "description": "Peligro generado después de recuperaciones altas.",
            "category": _category(pressing_efficiency),
        },
        "risk_exposure_score": {
            "label": "Build-Up Risk",
            "score": risk_exposure_score,
            "description": "Exposición tras pérdidas en zonas sensibles.",
            "category": _category(risk_exposure_score),
        },
    }
    return metrics
