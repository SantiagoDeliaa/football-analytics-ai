import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from src.services.canonical_models import (
    CanonicalEventAvailability,
    CanonicalEventTimelineItem,
    CanonicalLineup,
    CanonicalMatch,
    CanonicalPlayerMatchStats,
    CanonicalTeamExpectedMetrics,
    CanonicalTeamMatchStats,
)
from test_frontend_regression import load_vertical2_api_event


def _build_match():
    return CanonicalMatch(
        canonical_match_id="sportmonks__match__999",
        provider="sportmonks",
        provider_match_id="999",
        competition_name="Liga Profesional",
        season_name="2026",
        home_team_name="Boca Juniors",
        away_team_name="Central Córdoba",
        match_date="2026-05-28 20:00:00",
        status="FT",
        venue_name="La Bombonera",
        venue_city="Buenos Aires",
        home_score=2,
        away_score=1,
    )


def _build_match_context():
    match = _build_match()
    expected_metrics = [
        CanonicalTeamExpectedMetrics(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_name="Boca Juniors",
            xg=1.8,
            xgot=1.2,
            xpts=2.3,
            npxg=1.5,
            xg_open_play=1.1,
            xg_set_play=0.4,
            xg_free_kicks=0.1,
            shooting_performance=0.2,
            xga=0.9,
        ),
        CanonicalTeamExpectedMetrics(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_name="Central Córdoba",
            xg=0.9,
            xgot=0.6,
            xpts=0.7,
            npxg=0.8,
            xg_open_play=0.5,
            xg_set_play=0.2,
            xg_free_kicks=0.0,
            shooting_performance=-0.1,
            xga=1.8,
        ),
    ]
    timeline_events = [
        CanonicalEventTimelineItem(
            canonical_event_id="evt-1",
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            provider_event_id="5001",
            team_name="Boca Juniors",
            player_name="Cavani",
            minute=17,
            event_type="goal",
            event_label="Gol",
            result="1-0",
        )
    ]
    lineups = [
        CanonicalLineup(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_name="Boca Juniors",
            formation="4-4-2",
            coach="Diego Martínez",
            starters=[
                {
                    "player_id": "10",
                    "player_name": "Cavani",
                    "position": "FW",
                    "jersey_number": 10,
                    "is_starter": True,
                    "minutes_played": 90,
                    "rating": 7.8,
                    "stats": {},
                }
            ],
            substitutes=[
                {
                    "player_id": "20",
                    "player_name": "Merentiel",
                    "position": "FW",
                    "jersey_number": 16,
                    "is_starter": False,
                    "minutes_played": 20,
                    "rating": 6.9,
                    "stats": {},
                }
            ],
        )
    ]
    team_stats = [
        CanonicalTeamMatchStats(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            team_name="Boca Juniors",
            stats={"passes": {"value": 320, "label": "Pases"}},
        )
    ]
    player_stats = [
        CanonicalPlayerMatchStats(
            canonical_match_id=match.canonical_match_id,
            provider="sportmonks",
            player_id="10",
            player_name="Cavani",
            team_name="Boca Juniors",
            position="FW",
            jersey_number=10,
            minutes_played=90,
            rating=7.8,
            stats={
                "passes": {"value": 22, "label": "Pases"},
                "accurate_passes_percentage": {"value": 81, "label": "Precisión de pase"},
                "duels_won": {"value": 5, "label": "Duelos ganados"},
                "total_duels": {"value": 8, "label": "Duelos totales"},
            },
        )
    ]
    availability = CanonicalEventAvailability(
        canonical_match_id=match.canonical_match_id,
        provider="sportmonks",
        has_events=True,
        event_count=1,
        has_event_timeline=True,
        has_coordinates=False,
        has_xg=True,
        has_xgot=True,
        has_xpts=True,
        has_lineups=True,
        has_team_stats=True,
        has_player_stats=True,
        has_tracking=False,
        quality_notes=["Sin coordenadas"],
    )
    return {
        "match": match,
        "lineups": lineups,
        "timeline_events": timeline_events,
        "expected_metrics": expected_metrics,
        "team_stats": team_stats,
        "player_stats": player_stats,
        "availability": availability,
    }


def test_vertical2_sportmonks_without_api_key_shows_warning(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {},
            "sidebar": {"selectbox": {"Proveedor de datos": "Sportmonks"}},
        },
    )
    monkeypatch.setattr(module, "is_sportmonks_configured", lambda: False)
    monkeypatch.setattr(
        module,
        "initialize_event_data_persistence",
        lambda: {"persistence_backend": "local", "storage_backend": "local"},
    )

    module.render_vertical2_api_event()

    assert any("Sportmonks no está configurado" in item for item in recorder.warning_messages)


def test_vertical2_sportmonks_can_list_fixtures(monkeypatch):
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {
                "vertical2_sportmonks_fixture_search": {
                    "date": "2026-05-28",
                    "response": {"ok": True, "data": [_build_match()]},
                }
            },
            "sidebar": {"selectbox": {"Proveedor de datos": "Sportmonks"}},
            "text_input": {"Fecha del partido": "2026-05-28"},
        },
    )
    monkeypatch.setattr(module, "is_sportmonks_configured", lambda: True)
    monkeypatch.setattr(
        module,
        "initialize_event_data_persistence",
        lambda: {"persistence_backend": "local", "storage_backend": "local"},
    )

    module.render_vertical2_api_event()

    assert any("Fixture interno Sportmonks" in item for item in recorder.captions)


def test_vertical2_sportmonks_renders_match_center_from_canonical_context(monkeypatch):
    context = _build_match_context()
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {
            "session_state": {
                "vertical2_sportmonks_fixture_search": {
                    "date": "2026-05-28",
                    "response": {"ok": True, "data": [_build_match()]},
                },
                "vertical2_api_event_result": {
                    "provider": "sportmonks",
                    "match_id": "999",
                    "sportmonks_context": context,
                },
            },
            "sidebar": {"selectbox": {"Proveedor de datos": "Sportmonks"}},
            "text_input": {"Fecha del partido": "2026-05-28"},
        },
    )
    monkeypatch.setattr(module, "is_sportmonks_configured", lambda: True)
    monkeypatch.setattr(
        module,
        "initialize_event_data_persistence",
        lambda: {"persistence_backend": "local", "storage_backend": "local"},
    )

    module.render_vertical2_api_event()

    assert "Boca Juniors 2 - 1 Central Córdoba" in recorder.subheaders
    metric_labels = [label for label, _ in recorder.metrics]
    assert "Resultado" in metric_labels
    assert "Estado" in metric_labels
    assert "Competición" in metric_labels
    assert "Estadio" in metric_labels
    assert "Fecha" in metric_labels
    assert any("Fuentes y calidad de datos" in item for item in recorder.expander_labels)
    assert recorder.dataframe_calls >= 3


def test_vertical2_sportmonks_without_coordinates_does_not_render_maps(monkeypatch):
    context = _build_match_context()
    module, recorder = load_vertical2_api_event(
        monkeypatch,
        {"session_state": {}},
    )

    module._render_sportmonks_match_center(
        {
            "provider": "sportmonks",
            "match_id": "999",
            "sportmonks_context": context,
        },
        show_technical_info=False,
    )

    assert recorder.plotly_calls == 0
    assert any("no están disponibles" in item.lower() for item in recorder.info_messages)
