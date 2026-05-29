import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.canonical_models import CanonicalMatch
from src.services.providers.sportmonks_normalizer import (
    build_sportmonks_event_availability,
    normalize_sportmonks_expected_metrics,
    normalize_sportmonks_fixture,
    normalize_sportmonks_full_context,
    normalize_sportmonks_player_stats,
    normalize_sportmonks_timeline_event,
)


def _fixture_mock():
    return {
        "id": 999,
        "starting_at": "2026-05-28 20:00:00",
        "league": {"id": 55, "name": "Liga Profesional"},
        "season": {"id": 2026, "name": "2026"},
        "venue": {"name": "Monumental", "city_name": "Buenos Aires"},
        "state": {"name": "FT"},
        "participants": [
            {"id": 1, "name": "River Plate", "meta": {"location": "home"}},
            {"id": 2, "name": "Boca Juniors", "meta": {"location": "away"}},
        ],
        "scores": [
            {"participant_id": 1, "score": 2, "description": "current"},
            {"participant_id": 2, "score": 1, "description": "current"},
        ],
        "events": [
            {
                "id": 5001,
                "fixture_id": 999,
                "type": "goal",
                "minute": 18,
                "participant": {"id": 1, "name": "River Plate"},
                "player": {"id": 10, "name": "Colidio"},
            }
        ],
        "lineups": [
            {
                "fixture_id": 999,
                "team_id": 1,
                "team_name": "River Plate",
                "formation": "4-3-3",
                "starting_lineup": [
                    {"player": {"id": 10, "name": "Colidio"}, "position": "FW", "jersey_number": 11}
                ],
                "bench": [
                    {"player": {"id": 20, "name": "Borja"}, "position": "FW", "jersey_number": 9}
                ],
                "coach": {"name": "Marcelo Gallardo"},
            }
        ],
        "statistics": [
            {
                "fixture_id": 999,
                "team_id": 1,
                "team_name": "River Plate",
                "stats": [
                    {"type": "passes", "value": 410},
                    {"type": "expected-goals", "value": 1.8},
                    {"type": "expected-goals-on-target", "value": 1.4},
                    {"type": "expected-points", "value": 2.1},
                    {"type": "expected-goals-open-play", "value": 1.3},
                    {"type": "expected-goals-set-play", "value": 0.3},
                    {"type": "expected-goals-against", "value": 0.9},
                ],
            },
            {
                "fixture_id": 999,
                "player_id": 10,
                "player_name": "Colidio",
                "team_id": 1,
                "team_name": "River Plate",
                "minutes_played": 90,
                "rating": 7.6,
                "stats": [
                    {"type": "passes", "value": 22},
                    {"type": "shots-total", "value": 3},
                ],
            },
        ],
    }


def test_normalize_sportmonks_fixture_maps_home_away_and_score():
    fixture = normalize_sportmonks_fixture(_fixture_mock())

    assert isinstance(fixture, CanonicalMatch)
    assert fixture.home_team_name == "River Plate"
    assert fixture.away_team_name == "Boca Juniors"
    assert fixture.home_score == 2
    assert fixture.away_score == 1


def test_normalize_sportmonks_fixture_does_not_break_without_participants():
    raw_fixture = {"id": 1000, "starting_at": "2026-06-01", "home_score": 0, "away_score": 0}

    fixture = normalize_sportmonks_fixture(raw_fixture)

    assert fixture.provider_match_id == "1000"
    assert fixture.home_score == 0
    assert fixture.away_score == 0


def test_normalize_sportmonks_timeline_event_goal():
    event = normalize_sportmonks_timeline_event(_fixture_mock()["events"][0], match_id="999")

    assert event.event_type == "goal"
    assert event.event_label == "Gol"
    assert event.minute == 18
    assert event.player_name == "Colidio"
    assert event.team_name == "River Plate"


def test_normalize_sportmonks_timeline_event_unknown():
    event = normalize_sportmonks_timeline_event({"id": 1, "type": "strange-event"}, match_id="999")

    assert event.event_type == "unknown"
    assert event.event_label == "Evento"


def test_normalize_sportmonks_expected_metrics_maps_expected_fields():
    expected = normalize_sportmonks_expected_metrics(_fixture_mock()["statistics"], match_id="999")

    assert expected
    assert expected[0].xg == 1.8
    assert expected[0].xgot == 1.4
    assert expected[0].xpts == 2.1
    assert expected[0].xg_open_play == 1.3
    assert expected[0].xg_set_play == 0.3
    assert expected[0].xga == 0.9


def test_normalize_sportmonks_player_stats_maps_core_fields():
    player_stats = normalize_sportmonks_player_stats(_fixture_mock()["statistics"][1], match_id="999")

    assert player_stats.player_id == "10"
    assert player_stats.player_name == "Colidio"
    assert player_stats.minutes_played == 90
    assert player_stats.rating == 7.6
    assert isinstance(player_stats.stats, dict)


def test_build_sportmonks_event_availability():
    expected = normalize_sportmonks_expected_metrics(_fixture_mock()["statistics"], match_id="999")
    availability = build_sportmonks_event_availability(
        match_id="999",
        timeline_events=[{"id": 1}],
        lineups=[{"id": 1}],
        team_stats=[{"id": 1}],
        player_stats=[{"id": 1}],
        expected_metrics=expected,
    )

    assert availability.has_event_timeline is True
    assert availability.has_lineups is True
    assert availability.has_xg is True
    assert availability.has_coordinates is False
    assert availability.has_tracking is False


def test_normalize_sportmonks_full_context_with_partial_data():
    context = normalize_sportmonks_full_context({"data": _fixture_mock()})

    assert context["source"] == "sportmonks"
    assert context["match"] is not None
    assert context["availability"] is not None
