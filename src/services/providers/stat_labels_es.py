from __future__ import annotations

import re

SPORTMONKS_STAT_LABELS_ES = {
    "passes": "Pases",
    "accurate-passes": "Pases precisos",
    "accurate-passes-percentage": "Precision de pase",
    "touches": "Toques",
    "rating": "Rating",
    "minutes-played": "Minutos jugados",
    "possession-lost": "Perdidas de posesion",
    "interceptions": "Intercepciones",
    "long-balls": "Pases largos",
    "long-balls-won": "Pases largos completados",
    "ball-recovery": "Recuperaciones",
    "passes-in-final-third": "Pases al ultimo tercio",
    "total-duels": "Duelos totales",
    "duels-won": "Duelos ganados",
    "duels-lost": "Duelos perdidos",
    "duels-won-percentage": "Porcentaje de duelos ganados",
    "tackles": "Entradas",
    "tackles-won": "Entradas ganadas",
    "clearances": "Despejes",
    "aerials": "Duelos aereos",
    "aerials-won": "Duelos aereos ganados",
    "shots-total": "Remates totales",
    "shots-on-target": "Remates al arco",
    "shots-off-target": "Remates desviados",
    "fouls": "Faltas cometidas",
    "fouls-drawn": "Faltas recibidas",
    "total-crosses": "Centros totales",
    "accurate-crosses": "Centros precisos",
    "dribble-attempts": "Regates intentados",
    "successful-dribbles": "Regates exitosos",
    "dispossessed": "Desposesiones",
    "big-chances-created": "Grandes chances creadas",
    "chances-created": "Chances creadas",
    "key-passes": "Pases clave",
    "assists": "Asistencias",
    "captain": "Capitan",
}

SPORTMONKS_EVENT_LABELS_ES = {
    "goal": "Gol",
    "yellowcard": "Tarjeta amarilla",
    "redcard": "Tarjeta roja",
    "substitution": "Sustitucion",
    "var": "VAR",
    "VAR": "VAR",
    "penalty": "Penal",
    "own_goal": "Gol en contra",
    "unknown": "Evento",
}

SPORTMONKS_EXPECTED_LABELS_ES = {
    "expected-goals": "Goles esperados",
    "expected-goals-on-target": "Goles esperados al arco",
    "expected-points": "Puntos esperados",
    "expected-goals-free-kicks": "xG de tiros libres",
    "expected-non-penalty-goals": "xG sin penales",
    "expected-goals-set-play": "xG de pelota parada",
    "expected-goals-open-play": "xG en jugada",
    "shooting-performance": "Rendimiento de remate",
    "expected-goals-against": "xG concedido",
}


def _humanize_key(key: str) -> str:
    normalized = str(key or "").strip()
    if not normalized:
        return ""
    normalized = re.sub(r"[_-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]


def get_stat_label_es(stat_key: str) -> str:
    normalized_key = str(stat_key or "").strip()
    return SPORTMONKS_STAT_LABELS_ES.get(normalized_key, _humanize_key(normalized_key) or normalized_key)


def get_event_label_es(event_type: str) -> str:
    normalized_key = str(event_type or "").strip()
    return SPORTMONKS_EVENT_LABELS_ES.get(normalized_key, _humanize_key(normalized_key) or normalized_key)


def get_expected_metric_label_es(metric_key: str) -> str:
    normalized_key = str(metric_key or "").strip()
    return SPORTMONKS_EXPECTED_LABELS_ES.get(
        normalized_key,
        _humanize_key(normalized_key) or normalized_key,
    )
