from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


def build_canonical_id(*parts: Any) -> str:
    normalized_parts: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip().lower()
        if not text:
            continue
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace(" ", "_")
        text = re.sub(r"[^a-z0-9_.-]+", "_", text)
        text = re.sub(r"_+", "_", text)
        text = text.strip("._-")
        if text:
            normalized_parts.append(text)
    return "__".join(normalized_parts) or "unknown"


def safe_get(data: dict[str, Any] | None, path: list[Any] | str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default

    if isinstance(path, str):
        return data.get(path, default)

    current: Any = data
    for key in path:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if isinstance(current, list) and isinstance(key, int):
            if key < 0 or key >= len(current):
                return default
            current = current[key]
            continue
        return default

    return default if current is None else current


def normalize_optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return default

    normalized = text.replace("%", "").replace(",", ".")
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return default


def normalize_optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    if not text:
        return default

    normalized = text.replace(",", ".")
    try:
        return int(float(normalized))
    except (TypeError, ValueError):
        return default


@dataclass
class CanonicalCompetition:
    canonical_competition_id: str
    provider: str
    provider_competition_id: str
    name: str
    country: str | None = None
    type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalSeason:
    canonical_season_id: str
    provider: str
    provider_season_id: str
    competition_id: str | None
    name: str
    start_date: str | None = None
    end_date: str | None = None
    current: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalTeam:
    canonical_team_id: str
    provider: str
    provider_team_id: str
    name: str
    short_name: str | None = None
    country: str | None = None
    logo_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalPlayer:
    canonical_player_id: str
    provider: str
    provider_player_id: str
    name: str
    display_name: str | None = None
    team_id: str | None = None
    position: str | None = None
    nationality: str | None = None
    birthdate: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalMatch:
    canonical_match_id: str
    provider: str
    provider_match_id: str
    competition_id: str | None = None
    competition_name: str | None = None
    season_id: str | None = None
    season_name: str | None = None
    home_team_id: str | None = None
    away_team_id: str | None = None
    home_team_name: str | None = None
    away_team_name: str | None = None
    match_date: str | None = None
    status: str | None = None
    venue_name: str | None = None
    venue_city: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    winner_team_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalLineup:
    canonical_match_id: str
    provider: str
    team_id: str | None = None
    team_name: str | None = None
    formation: str | None = None
    starters: list[dict[str, Any]] = field(default_factory=list)
    substitutes: list[dict[str, Any]] = field(default_factory=list)
    coach: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalEventTimelineItem:
    canonical_event_id: str
    canonical_match_id: str
    provider: str
    provider_event_id: str | None = None
    team_id: str | None = None
    team_name: str | None = None
    player_id: str | None = None
    player_name: str | None = None
    related_player_id: str | None = None
    related_player_name: str | None = None
    minute: int | None = None
    extra_minute: int | None = None
    period: str | None = None
    event_type: str = "unknown"
    event_label: str = "Evento"
    result: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalTeamExpectedMetrics:
    canonical_match_id: str
    provider: str
    team_id: str | None = None
    team_name: str | None = None
    location: str | None = None
    xg: float | None = None
    xgot: float | None = None
    xpts: float | None = None
    npxg: float | None = None
    xg_open_play: float | None = None
    xg_set_play: float | None = None
    xg_free_kicks: float | None = None
    shooting_performance: float | None = None
    xga: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalTeamMatchStats:
    canonical_match_id: str
    provider: str
    team_id: str | None = None
    team_name: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalPlayerMatchStats:
    canonical_match_id: str
    provider: str
    player_id: str | None = None
    player_name: str | None = None
    team_id: str | None = None
    team_name: str | None = None
    position: str | None = None
    jersey_number: int | None = None
    is_starter: bool | None = None
    minutes_played: int | None = None
    rating: float | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalEventAvailability:
    canonical_match_id: str
    provider: str
    has_events: bool
    event_count: int
    has_event_timeline: bool
    has_coordinates: bool
    has_xg: bool
    has_xgot: bool
    has_xpts: bool
    has_lineups: bool
    has_team_stats: bool
    has_player_stats: bool
    has_tracking: bool
    quality_notes: list[str] = field(default_factory=list)


@dataclass
class CanonicalDataAvailability:
    canonical_match_id: str
    available_sources: dict[str, Any] = field(default_factory=dict)
    recommended_sources: dict[str, Any] = field(default_factory=dict)
    enabled_modules: dict[str, bool] = field(default_factory=dict)
    data_quality_score: int = 0
    data_quality_label: str = "Sin evaluar"
    warnings: list[str] = field(default_factory=list)
