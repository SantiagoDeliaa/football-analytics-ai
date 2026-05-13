import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.open_event_insights import generate_open_event_insights


def test_generate_open_event_insights_returns_spanish_strings_with_basic_metrics():
    insights = generate_open_event_insights(
        {
            "total_events": 18,
            "total_passes": 12,
            "total_shots": 3,
            "progressive_actions": 5,
            "final_third_actions": 7,
            "recoveries": 4,
        },
        selected_team="Argentina",
    )

    assert isinstance(insights, list)
    assert insights
    assert all(isinstance(item, str) for item in insights)
    assert any("El equipo seleccionado" in item for item in insights)
    assert any("acciones progresivas" in item for item in insights)


def test_generate_open_event_insights_uses_proprietary_metrics_when_available():
    insights = generate_open_event_insights(
        {
            "total_events": 20,
            "total_passes": 10,
            "total_shots": 4,
            "progressive_actions": 7,
            "final_third_actions": 9,
            "recoveries": 3,
            "field_tilt_index": 75.0,
            "progressive_threat_index": 72.0,
        },
        selected_team="Argentina",
    )

    assert any("dominio territorial" in item.lower() for item in insights)
    assert any("amenaza progresiva" in item.lower() for item in insights)


def test_generate_open_event_insights_mentions_verticality_when_relevant():
    insights = generate_open_event_insights(
        {
            "total_events": 12,
            "total_passes": 8,
            "total_shots": 1,
            "progressive_actions": 4,
            "final_third_actions": 4,
            "recoveries": 1,
            "directness_index": 70.0,
        },
        selected_team="Argentina",
    )

    assert any("verticalidad" in item.lower() for item in insights)


def test_generate_open_event_insights_mentions_player_influence_when_relevant():
    insights = generate_open_event_insights(
        {
            "total_events": 6,
            "total_passes": 2,
            "total_shots": 1,
            "progressive_actions": 1,
            "final_third_actions": 2,
            "recoveries": 1,
            "player_influence_score": 82.0,
        },
        selected_player="Lionel Messi",
    )

    assert any("alta influencia" in item.lower() for item in insights)


def test_generate_open_event_insights_handles_missing_or_none_metrics():
    insights = generate_open_event_insights(
        {
            "total_events": 1,
            "total_passes": 0,
            "total_shots": 0,
            "progressive_actions": 0,
            "final_third_actions": 0,
            "recoveries": 0,
            "field_tilt_index": None,
            "recovery_height_index": None,
            "player_influence_score": None,
        }
    )

    assert len(insights) >= 1
    assert all(isinstance(item, str) for item in insights)


def test_generate_open_event_insights_returns_empty_selection_message():
    insights = generate_open_event_insights({"total_events": 0})
    assert insights == ["No se detectaron eventos para los filtros seleccionados."]
