import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.open_event_metrics import calculate_open_event_metrics
from src.services.open_event_metrics import filter_analytical_events


def _canonical_events():
    return [
        {
            "event_id": "ignored-1",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Lionel Messi",
            "event_type": "Starting XI",
            "x": None,
            "y": None,
            "end_x": None,
            "end_y": None,
            "outcome": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "1",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Lionel Messi",
            "event_type": "Pass",
            "x": 50.0,
            "y": 30.0,
            "end_x": 70.0,
            "end_y": 32.0,
            "outcome": "Complete",
            "progressive": True,
            "under_pressure": True,
            "xG": 0.0,
        },
        {
            "event_id": "2",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Julian Alvarez",
            "event_type": "Pass",
            "x": None,
            "y": None,
            "end_x": None,
            "end_y": None,
            "outcome": "Complete",
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "3",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Alexis Mac Allister",
            "event_type": "Carry",
            "x": 82.0,
            "y": 28.0,
            "end_x": 92.0,
            "end_y": 30.0,
            "outcome": None,
            "progressive": True,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "4",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Julian Alvarez",
            "event_type": "Shot",
            "x": 101.0,
            "y": 35.0,
            "end_x": None,
            "end_y": None,
            "outcome": "Goal",
            "progressive": False,
            "under_pressure": True,
            "xG": 0.4,
        },
        {
            "event_id": "5",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Nahuel Molina",
            "event_type": "Ball Recovery",
            "x": 72.0,
            "y": 44.0,
            "end_x": None,
            "end_y": None,
            "outcome": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "6",
            "match_id": "m1",
            "team_name": "Francia",
            "player_name": "Kylian Mbappe",
            "event_type": "Pass",
            "x": 85.0,
            "y": 22.0,
            "end_x": 88.0,
            "end_y": 30.0,
            "outcome": "Complete",
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "7",
            "match_id": "m1",
            "team_name": "Francia",
            "player_name": "Antoine Griezmann",
            "event_type": "Interception",
            "x": 40.0,
            "y": 50.0,
            "end_x": None,
            "end_y": None,
            "outcome": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "8",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Cristian Romero",
            "event_type": "Duel",
            "x": 60.0,
            "y": 40.0,
            "end_x": None,
            "end_y": None,
            "outcome": "Won",
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
    ]


def test_filter_analytical_events_excludes_noise():
    filtered = filter_analytical_events(_canonical_events())
    assert all(event["event_type"] != "Starting XI" for event in filtered)
    assert len(filtered) == 8


def test_calculate_open_event_metrics_returns_basic_and_proprietary_metrics():
    metrics = calculate_open_event_metrics(_canonical_events())

    assert metrics["total_events"] == 8
    assert metrics["total_passes"] == 3
    assert metrics["total_shots"] == 1
    assert metrics["progressive_actions"] == 2
    assert metrics["final_third_actions"] == 3
    assert metrics["recoveries"] == 3
    assert metrics["total_carries"] == 1
    assert metrics["total_under_pressure"] == 2
    assert metrics["total_xg"] == 0.4
    assert metrics["field_tilt_index"] is None
    assert metrics["field_tilt_label"] == "No aplica"
    assert metrics["directness_index"] == 66.7
    assert metrics["recovery_height_index"] == 47.8
    assert metrics["shot_quality_index"] == 40.0
    assert isinstance(metrics["progressive_threat_index"], float)
    assert metrics["player_influence_score"] is None


def test_calculate_open_event_metrics_applies_team_filter():
    metrics = calculate_open_event_metrics(_canonical_events(), selected_team="Argentina")

    assert metrics["total_events"] == 6
    assert metrics["total_passes"] == 2
    assert metrics["total_shots"] == 1
    assert metrics["final_third_actions"] == 2
    assert metrics["recoveries"] == 2
    assert metrics["field_tilt_index"] == 66.7
    assert metrics["field_tilt_label"] == "Medio"


def test_calculate_open_event_metrics_applies_player_filter_and_returns_influence():
    metrics = calculate_open_event_metrics(
        _canonical_events(),
        selected_team="Argentina",
        selected_player="Lionel Messi",
    )

    assert metrics["total_events"] == 1
    assert metrics["total_passes"] == 1
    assert metrics["progressive_actions"] == 1
    assert metrics["field_tilt_index"] == 66.7
    assert metrics["player_influence_score"] is not None
    assert metrics["player_influence_label"] in {"Bajo", "Medio", "Alto"}


def test_metrics_handle_empty_or_non_applicable_cases_without_breaking():
    events = [
        {
            "event_id": "1",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Jugador desconocido",
            "event_type": "Pass",
            "x": None,
            "y": None,
            "end_x": None,
            "end_y": None,
            "outcome": "Complete",
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        }
    ]

    metrics = calculate_open_event_metrics(events, selected_team="Argentina")

    assert metrics["final_third_actions"] == 0
    assert metrics["recoveries"] == 0
    assert metrics["recovery_height_index"] is None
    assert metrics["recovery_height_label"] == "No aplica"
    assert metrics["shot_quality_index"] == 0.0


def test_metrics_can_be_recalculated_from_canonical_events_without_saved_scores():
    canonical_events = _canonical_events()

    metrics = calculate_open_event_metrics(canonical_events, selected_team="Argentina")

    assert "field_tilt_index" in metrics
    assert "directness_index" in metrics
    assert "progressive_threat_index" in metrics
