from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from src.services.canonical_models import (
    CanonicalCompetition,
    CanonicalEventAvailability,
    CanonicalEventTimelineItem,
    CanonicalLineup,
    CanonicalMatch,
    CanonicalPlayer,
    CanonicalPlayerMatchStats,
    CanonicalSeason,
    CanonicalTeam,
    CanonicalTeamExpectedMetrics,
    CanonicalTeamMatchStats,
    build_canonical_id,
    normalize_optional_float,
    normalize_optional_int,
    safe_get,
)

from .stat_labels_es import get_event_label_es, get_stat_label_es

SPORTMONKS_PROVIDER = "sportmonks"
SPORTMONKS_QUALITY_NOTE = (
    "Sportmonks se usa inicialmente para contexto, lineups, estadísticas, "
    "timeline y expected metrics. No se asumen coordenadas de eventos."
)

_EXPECTED_METRIC_MAP = {
    "expected-goals": "xg",
    "expected_goals": "xg",
    "expected-goals-on-target": "xgot",
    "expected_goals_on_target": "xgot",
    "expected-points": "xpts",
    "expected_points": "xpts",
    "expected-non-penalty-goals": "npxg",
    "expected_non_penalty_goals": "npxg",
    "expected-goals-open-play": "xg_open_play",
    "expected_goals_open_play": "xg_open_play",
    "expected-goals-set-play": "xg_set_play",
    "expected_goals_set_play": "xg_set_play",
    "expected-goals-free-kicks": "xg_free_kicks",
    "expected_goals_free_kicks": "xg_free_kicks",
    "shooting-performance": "shooting_performance",
    "shooting_performance": "shooting_performance",
    "expected-goals-against": "xga",
    "expected_goals_against": "xga",
}


def _to_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _snake_case(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return normalized


def _safe_metadata(raw: Any, allowed_keys: list[str] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    metadata: dict[str, Any] = {}
    keys = allowed_keys or []
    for key in keys:
        value = raw.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
            continue
        if isinstance(value, dict):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    nested[str(nested_key)] = nested_value
            if nested:
                metadata[key] = nested
    return metadata


def _extract_country_name(raw: dict[str, Any]) -> str | None:
    return (
        _string(safe_get(raw, ["country", "name"]))
        or _string(raw.get("country_name"))
        or _string(raw.get("country"))
        or _string(safe_get(raw, ["location", "country"]))
    )


def _extract_relation(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        candidate = raw.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _extract_id(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _find_team_by_location(raw_fixture: dict[str, Any], location: str) -> dict[str, Any]:
    participants = _to_list(raw_fixture.get("participants"))
    expected = location.strip().lower()
    for participant in participants:
        participant_dict = _to_dict(participant)
        participant_location = (
            _string(safe_get(participant_dict, ["meta", "location"]))
            or _string(participant_dict.get("location"))
            or _string(safe_get(participant_dict, ["pivot", "location"]))
        )
        if participant_location and participant_location.strip().lower() == expected:
            return participant_dict
    return {}


def _extract_team_identity(raw_team: dict[str, Any]) -> tuple[str | None, str | None]:
    team = _to_dict(raw_team.get("team")) or raw_team
    team_id = _extract_id(team, "id", "team_id", "participant_id")
    team_name = (
        _string(team.get("name"))
        or _string(team.get("short_name"))
        or _string(team.get("display_name"))
    )
    return team_id, team_name


def _extract_score_from_entry(score_entry: dict[str, Any]) -> int | None:
    for key in ("score", "goals", "value", "total"):
        parsed = normalize_optional_int(score_entry.get(key))
        if parsed is not None:
            return parsed
    return None


def _extract_fixture_scores(
    raw_fixture: dict[str, Any],
    home_team_id: str | None,
    away_team_id: str | None,
) -> tuple[int | None, int | None]:
    direct_home = normalize_optional_int(raw_fixture.get("home_score"))
    direct_away = normalize_optional_int(raw_fixture.get("away_score"))
    if direct_home is not None or direct_away is not None:
        return direct_home, direct_away

    home_score: int | None = None
    away_score: int | None = None
    for score_entry in _to_list(raw_fixture.get("scores")):
        entry = _to_dict(score_entry)
        participant_id = _extract_id(entry, "participant_id", "team_id")
        description = _snake_case(entry.get("description") or entry.get("type") or entry.get("name"))
        if description and description not in {"current", "score", "ft_score", "full_time", "total"}:
            continue
        score_value = _extract_score_from_entry(entry)
        if participant_id and participant_id == home_team_id and home_score is None:
            home_score = score_value
        if participant_id and participant_id == away_team_id and away_score is None:
            away_score = score_value

    return home_score, away_score


def _infer_winner_team_id(
    raw_fixture: dict[str, Any],
    home_team_id: str | None,
    away_team_id: str | None,
    home_score: int | None,
    away_score: int | None,
) -> str | None:
    explicit_winner_id = _extract_id(raw_fixture, "winner_team_id", "winning_team_id")
    if explicit_winner_id:
        return explicit_winner_id

    winner = _to_dict(raw_fixture.get("winner"))
    winner_id = _extract_id(winner, "id", "team_id", "participant_id")
    if winner_id:
        return winner_id

    if home_score is not None and away_score is not None:
        if home_score > away_score:
            return home_team_id
        if away_score > home_score:
            return away_team_id
    return None


def _normalize_event_type(raw_event: dict[str, Any]) -> str:
    raw_type = (
        _string(raw_event.get("type"))
        or _string(raw_event.get("event_type"))
        or _string(raw_event.get("type_name"))
        or _string(safe_get(raw_event, ["type", "name"]))
        or _string(raw_event.get("name"))
        or _string(raw_event.get("code"))
        or _string(raw_event.get("detail"))
        or ""
    )
    normalized = _snake_case(raw_type)
    if "own" in normalized and "goal" in normalized:
        return "own_goal"
    if "yellow" in normalized:
        return "yellowcard"
    if "red" in normalized:
        return "redcard"
    if "sub" in normalized:
        return "substitution"
    if normalized in {"var", "video_assistant_referee"} or "var" in normalized:
        return "var"
    if "penalty" in normalized:
        return "penalty"
    if "goal" in normalized:
        return "goal"
    return "unknown"


def _extract_period(raw_event: dict[str, Any]) -> str | None:
    return (
        _string(raw_event.get("period"))
        or _string(safe_get(raw_event, ["time", "period"]))
        or _string(safe_get(raw_event, ["period", "name"]))
    )


def _extract_result(raw_event: dict[str, Any]) -> str | None:
    return (
        _string(raw_event.get("result"))
        or _string(raw_event.get("detail"))
        or _string(raw_event.get("outcome"))
        or _string(safe_get(raw_event, ["type", "name"]))
    )


def _normalize_lineup_player(raw_player: dict[str, Any], is_starter: bool) -> dict[str, Any]:
    player = _to_dict(raw_player.get("player")) or raw_player
    player_stats_source = raw_player.get("stats") or raw_player.get("statistics") or {}
    return {
        "player_id": _extract_id(player, "id", "player_id"),
        "player_name": _string(player.get("name")) or _string(player.get("display_name")),
        "position": _string(raw_player.get("position")) or _string(player.get("position")),
        "jersey_number": normalize_optional_int(raw_player.get("jersey_number") or raw_player.get("number")),
        "is_starter": is_starter,
        "minutes_played": normalize_optional_int(
            raw_player.get("minutes_played") or raw_player.get("minutes")
        ),
        "rating": normalize_optional_float(raw_player.get("rating")),
        "stats": _normalize_stat_entries(player_stats_source),
    }


def _normalize_stat_entries(raw_stats: Any) -> dict[str, Any]:
    if isinstance(raw_stats, dict):
        normalized: dict[str, Any] = {}
        for key, value in raw_stats.items():
            normalized_key = _snake_case(key)
            if not normalized_key:
                continue
            if isinstance(value, dict) and "value" in value:
                label = get_stat_label_es(str(key))
                normalized[normalized_key] = {
                    "value": value.get("value"),
                    "label": label,
                }
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[normalized_key] = value
        return normalized

    normalized_list: dict[str, Any] = {}
    for item in _to_list(raw_stats):
        entry = _to_dict(item)
        stat_key = (
            _string(entry.get("type"))
            or _string(entry.get("code"))
            or _string(entry.get("name"))
            or _string(safe_get(entry, ["type", "name"]))
            or _string(safe_get(entry, ["type", "code"]))
        )
        if not stat_key:
            continue
        normalized_key = _snake_case(stat_key)
        stat_value = entry.get("value")
        if stat_value is None and "data" in entry:
            stat_value = entry.get("data")
        normalized_list[normalized_key] = {
            "value": stat_value,
            "label": get_stat_label_es(str(stat_key)),
        }
    return normalized_list


def _extract_expected_metric_entries(raw_expected: Any) -> list[dict[str, Any]]:
    if isinstance(raw_expected, dict):
        if isinstance(raw_expected.get("data"), list):
            return [item for item in raw_expected["data"] if isinstance(item, dict)]
        if isinstance(raw_expected.get("statistics"), list):
            return [item for item in raw_expected["statistics"] if isinstance(item, dict)]
        if isinstance(raw_expected.get("expected"), list):
            return [item for item in raw_expected["expected"] if isinstance(item, dict)]
        return [_to_dict(raw_expected)]
    return [item for item in _to_list(raw_expected) if isinstance(item, dict)]


def _extract_expected_metrics_map(entry: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    raw_stats = entry.get("stats") or entry.get("statistics") or entry.get("values") or entry

    if isinstance(raw_stats, dict):
        iterable = raw_stats.items()
        for key, value in iterable:
            metric_name = _EXPECTED_METRIC_MAP.get(_snake_case(key))
            if metric_name:
                values[metric_name] = normalize_optional_float(value)
        return values

    for item in _to_list(raw_stats):
        stat_item = _to_dict(item)
        key = (
            _string(stat_item.get("type"))
            or _string(stat_item.get("code"))
            or _string(stat_item.get("name"))
            or _string(safe_get(stat_item, ["type", "name"]))
            or _string(safe_get(stat_item, ["type", "code"]))
        )
        if not key:
            continue
        metric_name = _EXPECTED_METRIC_MAP.get(_snake_case(key))
        if not metric_name:
            continue
        values[metric_name] = normalize_optional_float(stat_item.get("value"))
    return values


def _extract_statistics_groups(raw_fixture: dict[str, Any]) -> list[dict[str, Any]]:
    statistics = raw_fixture.get("statistics")
    return [item for item in _to_list(statistics) if isinstance(item, dict)]


def normalize_sportmonks_league(raw_league: dict[str, Any]) -> CanonicalCompetition:
    league = _to_dict(raw_league)
    provider_competition_id = _extract_id(league, "id", "league_id") or "unknown-league"
    name = _string(league.get("name")) or "Competicion Sportmonks"
    return CanonicalCompetition(
        canonical_competition_id=build_canonical_id(SPORTMONKS_PROVIDER, "competition", provider_competition_id or name),
        provider=SPORTMONKS_PROVIDER,
        provider_competition_id=provider_competition_id,
        name=name,
        country=_extract_country_name(league),
        type=_string(league.get("type")),
        metadata=_safe_metadata(league, ["id", "name", "type", "country_id", "short_code"]),
    )


def normalize_sportmonks_season(raw_season: dict[str, Any]) -> CanonicalSeason:
    season = _to_dict(raw_season)
    provider_season_id = _extract_id(season, "id", "season_id") or "unknown-season"
    competition_id = _extract_id(season, "league_id", "competition_id") or _extract_id(
        _extract_relation(season, "league", "competition"),
        "id",
    )
    return CanonicalSeason(
        canonical_season_id=build_canonical_id(SPORTMONKS_PROVIDER, "season", provider_season_id or season.get("name")),
        provider=SPORTMONKS_PROVIDER,
        provider_season_id=provider_season_id,
        competition_id=competition_id,
        name=_string(season.get("name")) or _string(season.get("display_name")) or provider_season_id,
        start_date=_string(season.get("starting_at")) or _string(season.get("start_date")),
        end_date=_string(season.get("ending_at")) or _string(season.get("end_date")),
        current=season.get("is_current") if isinstance(season.get("is_current"), bool) else season.get("current"),
        metadata=_safe_metadata(season, ["id", "name", "league_id", "starting_at", "ending_at"]),
    )


def normalize_sportmonks_team(raw_team: dict[str, Any]) -> CanonicalTeam:
    team = _to_dict(raw_team.get("team")) or _to_dict(raw_team.get("participant")) or _to_dict(raw_team)
    provider_team_id = _extract_id(team, "id", "team_id", "participant_id") or "unknown-team"
    name = _string(team.get("name")) or _string(team.get("short_name")) or "Equipo Sportmonks"
    return CanonicalTeam(
        canonical_team_id=build_canonical_id(SPORTMONKS_PROVIDER, "team", provider_team_id or name),
        provider=SPORTMONKS_PROVIDER,
        provider_team_id=provider_team_id,
        name=name,
        short_name=_string(team.get("short_name")),
        country=_extract_country_name(team),
        logo_url=_string(team.get("image_path")) or _string(team.get("logo_url")),
        metadata=_safe_metadata(team, ["id", "name", "short_name", "country_id", "image_path"]),
    )


def normalize_sportmonks_player(raw_player: dict[str, Any]) -> CanonicalPlayer:
    player = _to_dict(raw_player.get("player")) or _to_dict(raw_player)
    provider_player_id = _extract_id(player, "id", "player_id") or "unknown-player"
    name = _string(player.get("name")) or _string(player.get("display_name")) or "Jugador Sportmonks"
    position_value = player.get("position")
    position_name = _string(position_value.get("name")) if isinstance(position_value, dict) else _string(position_value)
    return CanonicalPlayer(
        canonical_player_id=build_canonical_id(SPORTMONKS_PROVIDER, "player", provider_player_id or name),
        provider=SPORTMONKS_PROVIDER,
        provider_player_id=provider_player_id,
        name=name,
        display_name=_string(player.get("display_name")) or name,
        team_id=_extract_id(player, "team_id", "participant_id"),
        position=position_name,
        nationality=_extract_country_name(player) or _string(player.get("nationality")),
        birthdate=_string(player.get("date_of_birth")) or _string(player.get("birthdate")),
        metadata=_safe_metadata(player, ["id", "name", "display_name", "team_id", "date_of_birth"]),
    )


def normalize_sportmonks_fixture(raw_fixture: dict[str, Any]) -> CanonicalMatch:
    fixture = _to_dict(raw_fixture)
    provider_match_id = _extract_id(fixture, "id", "fixture_id", "match_id") or "unknown-fixture"
    league = _extract_relation(fixture, "league", "competition")
    season = _extract_relation(fixture, "season")
    venue = _extract_relation(fixture, "venue")
    state = _extract_relation(fixture, "state", "status")

    home_participant = _find_team_by_location(fixture, "home")
    away_participant = _find_team_by_location(fixture, "away")
    home_team_id, home_team_name = _extract_team_identity(home_participant)
    away_team_id, away_team_name = _extract_team_identity(away_participant)

    if not home_team_name:
        home_team_name = _string(fixture.get("home_team_name")) or _string(safe_get(fixture, ["home_team", "name"]))
    if not away_team_name:
        away_team_name = _string(fixture.get("away_team_name")) or _string(safe_get(fixture, ["away_team", "name"]))

    if not home_team_id:
        home_team_id = _extract_id(_to_dict(fixture.get("home_team")), "id", "team_id")
    if not away_team_id:
        away_team_id = _extract_id(_to_dict(fixture.get("away_team")), "id", "team_id")

    home_score, away_score = _extract_fixture_scores(fixture, home_team_id, away_team_id)
    winner_team_id = _infer_winner_team_id(fixture, home_team_id, away_team_id, home_score, away_score)

    return CanonicalMatch(
        canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", provider_match_id),
        provider=SPORTMONKS_PROVIDER,
        provider_match_id=provider_match_id,
        competition_id=_extract_id(league, "id", "league_id"),
        competition_name=_string(league.get("name")),
        season_id=_extract_id(season, "id", "season_id"),
        season_name=_string(season.get("name")),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        match_date=_string(fixture.get("starting_at")) or _string(fixture.get("match_date")),
        status=_string(state.get("name")) or _string(state.get("state")) or _string(fixture.get("status")),
        venue_name=_string(venue.get("name")) or _string(fixture.get("venue_name")),
        venue_city=_string(venue.get("city_name")) or _string(venue.get("city")),
        home_score=home_score,
        away_score=away_score,
        winner_team_id=winner_team_id,
        metadata=_safe_metadata(fixture, ["id", "starting_at", "result_info", "leg", "round_id"]),
    )


def normalize_sportmonks_timeline_event(
    raw_event: dict[str, Any],
    match_id: str | None = None,
) -> CanonicalEventTimelineItem:
    event = _to_dict(raw_event)
    event_type = _normalize_event_type(event)
    provider_event_id = _extract_id(event, "id", "event_id")
    team = _extract_relation(event, "team", "participant")
    player = _extract_relation(event, "player")
    related_player = _extract_relation(event, "related_player", "assist_player", "substitute")
    resolved_match_id = _string(match_id) or _extract_id(event, "fixture_id", "match_id") or "unknown-match"
    minute = normalize_optional_int(event.get("minute") or safe_get(event, ["time", "minute"]))
    extra_minute = normalize_optional_int(
        event.get("extra_minute")
        or event.get("injury_time")
        or safe_get(event, ["time", "extra_minute"])
    )
    description = _string(event.get("description")) or _string(event.get("comment"))
    result = _extract_result(event)

    return CanonicalEventTimelineItem(
        canonical_event_id=build_canonical_id(
            SPORTMONKS_PROVIDER,
            "event",
            provider_event_id or event_type,
            resolved_match_id,
            minute,
        ),
        canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", resolved_match_id),
        provider=SPORTMONKS_PROVIDER,
        provider_event_id=provider_event_id,
        team_id=_extract_id(team, "id", "team_id", "participant_id") or _extract_id(event, "participant_id", "team_id"),
        team_name=_string(team.get("name")) or _string(event.get("team_name")),
        player_id=_extract_id(player, "id", "player_id") or _extract_id(event, "player_id"),
        player_name=_string(player.get("name")) or _string(event.get("player_name")),
        related_player_id=_extract_id(related_player, "id", "player_id") or _extract_id(event, "related_player_id"),
        related_player_name=_string(related_player.get("name")) or _string(event.get("related_player_name")),
        minute=minute,
        extra_minute=extra_minute,
        period=_extract_period(event),
        event_type=event_type,
        event_label=get_event_label_es(event_type),
        result=result,
        description=description,
        metadata=_safe_metadata(event, ["id", "type", "detail", "minute", "result", "participant_id", "player_id"]),
    )


def normalize_sportmonks_lineup(
    raw_lineup: dict[str, Any],
    match_id: str | None = None,
) -> CanonicalLineup:
    lineup = _to_dict(raw_lineup)
    team = _extract_relation(lineup, "team", "participant")
    players_section = _to_list(lineup.get("players"))
    starters_source = _to_list(lineup.get("starting_lineup")) or [
        item for item in players_section if _to_dict(item).get("starter") is True
    ]
    substitutes_source = _to_list(lineup.get("bench")) or [
        item for item in players_section if _to_dict(item).get("starter") is False
    ]

    coach_raw = _extract_relation(lineup, "coach")
    coach_name = _string(coach_raw.get("name")) or _string(lineup.get("coach_name"))
    resolved_match_id = _string(match_id) or _extract_id(lineup, "fixture_id", "match_id") or "unknown-match"

    return CanonicalLineup(
        canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", resolved_match_id),
        provider=SPORTMONKS_PROVIDER,
        team_id=_extract_id(team, "id", "team_id", "participant_id") or _extract_id(lineup, "team_id", "participant_id"),
        team_name=_string(team.get("name")) or _string(lineup.get("team_name")),
        formation=_string(lineup.get("formation")),
        starters=[_normalize_lineup_player(_to_dict(item), True) for item in starters_source],
        substitutes=[_normalize_lineup_player(_to_dict(item), False) for item in substitutes_source],
        coach=coach_name,
        metadata=_safe_metadata(lineup, ["team_id", "formation", "fixture_id"]),
    )


def normalize_sportmonks_expected_metrics(
    raw_expected: dict[str, Any] | list[dict[str, Any]],
    match_id: str | None = None,
) -> list[CanonicalTeamExpectedMetrics]:
    resolved_match_id = _string(match_id) or _extract_id(_to_dict(raw_expected), "fixture_id", "match_id") or "unknown-match"
    normalized: list[CanonicalTeamExpectedMetrics] = []

    for entry in _extract_expected_metric_entries(raw_expected):
        metrics_map = _extract_expected_metrics_map(entry)
        if not metrics_map:
            continue
        participant = _extract_relation(entry, "team", "participant")
        normalized.append(
            CanonicalTeamExpectedMetrics(
                canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", resolved_match_id),
                provider=SPORTMONKS_PROVIDER,
                team_id=_extract_id(participant, "id", "team_id", "participant_id")
                or _extract_id(entry, "team_id", "participant_id"),
                team_name=_string(participant.get("name")) or _string(entry.get("team_name")),
                location=_string(safe_get(entry, ["meta", "location"])) or _string(entry.get("location")),
                xg=metrics_map.get("xg"),
                xgot=metrics_map.get("xgot"),
                xpts=metrics_map.get("xpts"),
                npxg=metrics_map.get("npxg"),
                xg_open_play=metrics_map.get("xg_open_play"),
                xg_set_play=metrics_map.get("xg_set_play"),
                xg_free_kicks=metrics_map.get("xg_free_kicks"),
                shooting_performance=metrics_map.get("shooting_performance"),
                xga=metrics_map.get("xga"),
                metadata=_safe_metadata(entry, ["team_id", "participant_id", "location"]),
            )
        )
    return normalized


def normalize_sportmonks_team_stats(
    raw_stats: dict[str, Any],
    match_id: str | None = None,
) -> CanonicalTeamMatchStats:
    stats_entry = _to_dict(raw_stats)
    participant = _extract_relation(stats_entry, "team", "participant")
    resolved_match_id = _string(match_id) or _extract_id(stats_entry, "fixture_id", "match_id") or "unknown-match"
    return CanonicalTeamMatchStats(
        canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", resolved_match_id),
        provider=SPORTMONKS_PROVIDER,
        team_id=_extract_id(participant, "id", "team_id", "participant_id")
        or _extract_id(stats_entry, "team_id", "participant_id"),
        team_name=_string(participant.get("name")) or _string(stats_entry.get("team_name")),
        stats=_normalize_stat_entries(stats_entry.get("stats") or stats_entry.get("statistics") or stats_entry),
        metadata=_safe_metadata(stats_entry, ["team_id", "participant_id", "fixture_id"]),
    )


def normalize_sportmonks_player_stats(
    raw_stats: dict[str, Any],
    match_id: str | None = None,
) -> CanonicalPlayerMatchStats:
    stats_entry = _to_dict(raw_stats)
    player = _extract_relation(stats_entry, "player")
    participant = _extract_relation(stats_entry, "team", "participant")
    position_value = stats_entry.get("position") or player.get("position")
    position_name = _string(position_value.get("name")) if isinstance(position_value, dict) else _string(position_value)
    resolved_match_id = _string(match_id) or _extract_id(stats_entry, "fixture_id", "match_id") or "unknown-match"
    normalized_stats = _normalize_stat_entries(stats_entry.get("stats") or stats_entry.get("statistics") or stats_entry)

    return CanonicalPlayerMatchStats(
        canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", resolved_match_id),
        provider=SPORTMONKS_PROVIDER,
        player_id=_extract_id(player, "id", "player_id") or _extract_id(stats_entry, "player_id"),
        player_name=_string(player.get("name")) or _string(stats_entry.get("player_name")),
        team_id=_extract_id(participant, "id", "team_id", "participant_id")
        or _extract_id(stats_entry, "team_id", "participant_id"),
        team_name=_string(participant.get("name")) or _string(stats_entry.get("team_name")),
        position=position_name,
        jersey_number=normalize_optional_int(stats_entry.get("jersey_number") or stats_entry.get("number")),
        is_starter=stats_entry.get("is_starter") if isinstance(stats_entry.get("is_starter"), bool) else stats_entry.get("starter"),
        minutes_played=normalize_optional_int(
            stats_entry.get("minutes_played") or normalized_stats.get("minutes_played", {}).get("value")
            if isinstance(normalized_stats.get("minutes_played"), dict)
            else stats_entry.get("minutes_played")
        ),
        rating=normalize_optional_float(
            stats_entry.get("rating") or (
                normalized_stats.get("rating", {}).get("value")
                if isinstance(normalized_stats.get("rating"), dict)
                else normalized_stats.get("rating")
            )
        ),
        stats=normalized_stats,
        metadata=_safe_metadata(stats_entry, ["player_id", "team_id", "fixture_id", "is_starter"]),
    )


def build_sportmonks_event_availability(
    match_id: str,
    timeline_events: list[Any] | None = None,
    lineups: list[Any] | None = None,
    team_stats: list[Any] | None = None,
    player_stats: list[Any] | None = None,
    expected_metrics: list[CanonicalTeamExpectedMetrics] | None = None,
) -> CanonicalEventAvailability:
    normalized_expected = expected_metrics or []
    return CanonicalEventAvailability(
        canonical_match_id=build_canonical_id(SPORTMONKS_PROVIDER, "match", match_id),
        provider=SPORTMONKS_PROVIDER,
        has_events=bool(timeline_events),
        event_count=len(timeline_events or []),
        has_event_timeline=bool(timeline_events),
        has_coordinates=False,
        has_xg=any(item.xg is not None for item in normalized_expected),
        has_xgot=any(item.xgot is not None for item in normalized_expected),
        has_xpts=any(item.xpts is not None for item in normalized_expected),
        has_lineups=bool(lineups),
        has_team_stats=bool(team_stats),
        has_player_stats=bool(player_stats),
        has_tracking=False,
        quality_notes=[SPORTMONKS_QUALITY_NOTE],
    )


def normalize_sportmonks_full_context(raw_fixture_response: dict[str, Any] | Any) -> dict[str, Any]:
    response = _to_dict(raw_fixture_response)
    fixture_payload = response.get("data") if "data" in response else raw_fixture_response
    raw_fixture = _to_dict(fixture_payload[0]) if isinstance(fixture_payload, list) and fixture_payload else _to_dict(fixture_payload)

    match = normalize_sportmonks_fixture(raw_fixture) if raw_fixture else None
    provider_match_id = match.provider_match_id if match else "unknown-match"

    lineups = [
        normalize_sportmonks_lineup(item, match_id=provider_match_id)
        for item in _to_list(raw_fixture.get("lineups"))
        if isinstance(item, dict)
    ]
    timeline_events = [
        normalize_sportmonks_timeline_event(item, match_id=provider_match_id)
        for item in _to_list(raw_fixture.get("events"))
        if isinstance(item, dict)
    ]

    statistics_groups = _extract_statistics_groups(raw_fixture)
    expected_metrics = normalize_sportmonks_expected_metrics(statistics_groups or raw_fixture.get("statistics"), match_id=provider_match_id)

    team_stats: list[CanonicalTeamMatchStats] = []
    player_stats: list[CanonicalPlayerMatchStats] = []
    for stats_entry in statistics_groups:
        if _extract_relation(stats_entry, "player") or _extract_id(stats_entry, "player_id"):
            player_stats.append(normalize_sportmonks_player_stats(stats_entry, match_id=provider_match_id))
        else:
            team_stats.append(normalize_sportmonks_team_stats(stats_entry, match_id=provider_match_id))

    availability = build_sportmonks_event_availability(
        match_id=provider_match_id,
        timeline_events=timeline_events,
        lineups=lineups,
        team_stats=team_stats,
        player_stats=player_stats,
        expected_metrics=expected_metrics,
    )

    return {
        "match": match,
        "lineups": lineups,
        "timeline_events": timeline_events,
        "expected_metrics": expected_metrics,
        "team_stats": team_stats,
        "player_stats": player_stats,
        "availability": availability,
        "source": SPORTMONKS_PROVIDER,
    }
