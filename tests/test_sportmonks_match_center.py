import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.canonical_models import CanonicalEventAvailability
from src.services.canonical_models import CanonicalEventTimelineItem
from src.services.canonical_models import CanonicalLineup
from src.services.canonical_models import CanonicalMatch
from src.services.canonical_models import CanonicalPlayerMatchStats
from src.services.canonical_models import CanonicalTeamExpectedMetrics
from src.services.canonical_models import CanonicalTeamMatchStats
from src.services.providers import sportmonks_match_center as service


def _full_context_response():
    match = CanonicalMatch(
        canonical_match_id="sportmonks:match:19636404",
        provider="sportmonks",
        provider_match_id="19636404",
        competition_id="55",
        competition_name="Liga Profesional de Futbol",
        season_id="2026",
        season_name="2026",
        home_team_id="14212",
        away_team_id="587",
        home_team_name="Central Cordoba SdE",
        away_team_name="Boca Juniors",
        match_date="2026-05-02 19:15:00",
        status="FT",
        venue_name="Estadio Unico Madre de Ciudades",
        venue_city="Santiago del Estero",
        home_score=1,
        away_score=2,
        winner_team_id="587",
    )
    expected_metrics = [
        CanonicalTeamExpectedMetrics(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="14212",
            team_name="Central Cordoba SdE",
            xg=1.46,
            xgot=2.43,
            xpts=1.39,
            npxg=1.46,
            xga=1.44,
        ),
        CanonicalTeamExpectedMetrics(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="587",
            team_name="Boca Juniors",
            xg=1.44,
            xgot=1.60,
            xpts=1.37,
            npxg=1.44,
            xga=1.46,
        ),
    ]
    timeline = [
        CanonicalEventTimelineItem(
            canonical_event_id="event-1",
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="587",
            team_name="Boca Juniors",
            player_id="10",
            player_name="Alan Velasco",
            related_player_id="20",
            related_player_name="Williams Alarcon",
            minute=43,
            event_type="goal",
            event_label="Gol",
            result="0-1",
            description="Gol de Boca Juniors",
        )
    ]
    lineups = [
        CanonicalLineup(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="14212",
            team_name="Central Cordoba SdE",
            formation="4-4-2",
            coach="Omar De Felippe",
            starters=[{"player_id": "1", "player_name": "Mansilla", "position": "GK", "jersey_number": 1}],
            substitutes=[{"player_id": "2", "player_name": "Suplente", "position": "FW", "jersey_number": 18}],
        ),
        CanonicalLineup(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="587",
            team_name="Boca Juniors",
            formation="4-3-3",
            coach="Diego Martinez",
            starters=[{"player_id": "10", "player_name": "Alan Velasco", "position": "FW", "jersey_number": 10}],
            substitutes=[],
        ),
    ]
    team_stats = [
        CanonicalTeamMatchStats(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="14212",
            team_name="Central Cordoba SdE",
            stats={"passes": {"value": 320, "label": "Pases"}},
        ),
        CanonicalTeamMatchStats(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_id="587",
            team_name="Boca Juniors",
            stats={"passes": {"value": 410, "label": "Pases"}},
        ),
    ]
    player_stats = [
        CanonicalPlayerMatchStats(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            player_id="10",
            player_name="Alan Velasco",
            team_id="587",
            team_name="Boca Juniors",
            position="FW",
            jersey_number=10,
            is_starter=True,
            minutes_played=87,
            rating=7.8,
            stats={
                "passes": {"value": 22, "label": "Pases"},
                "accurate_passes_percentage": {"value": 81, "label": "Precision de pase"},
                "duels_won": {"value": 4, "label": "Duelos ganados"},
                "total_duels": {"value": 7, "label": "Duelos totales"},
            },
        )
    ]
    availability = CanonicalEventAvailability(
        canonical_match_id=match.canonical_match_id,
        provider="sportmonks",
        has_events=True,
        event_count=12,
        has_event_timeline=True,
        has_coordinates=False,
        has_xg=True,
        has_xgot=True,
        has_xpts=True,
        has_lineups=True,
        has_team_stats=True,
        has_player_stats=True,
        has_tracking=False,
        quality_notes=["Cobertura parcial sin coordenadas"],
    )
    return {
        "ok": True,
        "data": {
            "match": match,
            "expected_metrics": expected_metrics,
            "timeline_events": timeline,
            "lineups": lineups,
            "team_stats": team_stats,
            "player_stats": player_stats,
            "availability": availability,
        },
    }


def test_build_sportmonks_match_center_happy_path(monkeypatch):
    monkeypatch.setattr(service, "get_sportmonks_match_context", lambda match_id: _full_context_response())

    result = service.build_sportmonks_match_center("19636404")

    assert result["provider"] == "sportmonks"
    assert result["match"]["home_team"]["name"] == "Central Cordoba SdE"
    assert result["match"]["away_team"]["score"] == 2
    assert result["expected_metrics"]["home"]["xg"] == 1.46
    assert result["timeline"][0]["event_label"] == "Gol"
    assert result["lineups"]["away"]["formation"] == "4-3-3"
    assert result["team_stats"]["home"]["stats"][0]["label"] == "Pases"
    assert result["player_stats"][0]["insights"]
    assert result["derived_metrics"]["away"]["eficacia_ofensiva"] == 1.39
    assert result["data_quality"]["level"] == "media"
    assert result["data_quality"]["enabled_modules"]["event_maps"] is False
    assert "raw_payload" not in result
    assert any("coordenadas" in insight.lower() for insight in result["insights"])


def test_build_sportmonks_match_center_handles_missing_optional_sections(monkeypatch):
    match = CanonicalMatch(
        canonical_match_id="sportmonks:match:1",
        provider="sportmonks",
        provider_match_id="1",
        home_team_id="1",
        away_team_id="2",
        home_team_name="Local",
        away_team_name="Visitante",
    )
    availability = CanonicalEventAvailability(
        canonical_match_id=match.canonical_match_id,
        provider="sportmonks",
        has_events=False,
        event_count=0,
        has_event_timeline=False,
        has_coordinates=False,
        has_xg=False,
        has_xgot=False,
        has_xpts=False,
        has_lineups=False,
        has_team_stats=False,
        has_player_stats=False,
        has_tracking=False,
        quality_notes=[],
    )
    monkeypatch.setattr(
        service,
        "get_sportmonks_match_context",
        lambda match_id: {
            "ok": True,
            "data": {
                "match": match,
                "expected_metrics": [],
                "timeline_events": [],
                "lineups": [],
                "team_stats": [],
                "player_stats": [],
                "availability": availability,
            },
        },
    )

    result = service.build_sportmonks_match_center("1")

    assert result["expected_metrics"]["home"]["xg"] is None
    assert result["lineups"]["home"]["starters"] == []
    assert result["player_stats"] == []
    assert result["data_quality"]["level"] == "baja"


def test_build_sportmonks_match_center_raises_not_found(monkeypatch):
    monkeypatch.setattr(service, "get_sportmonks_match_context", lambda match_id: {"ok": True, "data": {"match": None}})

    with pytest.raises(service.SportmonksMatchCenterNotFoundError):
        service.build_sportmonks_match_center("404")
