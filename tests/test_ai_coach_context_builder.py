import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.ai_coach.context_builder import build_match_context


def _sample_events():
    return [
        {
            "event_id": "1",
            "match_id": "match-1",
            "team_name": "Argentina",
            "player_name": "Lionel Messi",
            "event_type": "Pass",
            "x": 55.0,
            "y": 30.0,
            "end_x": 82.0,
            "end_y": 33.0,
            "progressive": True,
            "under_pressure": True,
            "xG": 0.0,
        },
        {
            "event_id": "2",
            "match_id": "match-1",
            "team_name": "Argentina",
            "player_name": "Julian Alvarez",
            "event_type": "Shot",
            "x": 101.0,
            "y": 35.0,
            "end_x": None,
            "end_y": None,
            "progressive": False,
            "under_pressure": True,
            "xG": 0.35,
        },
        {
            "event_id": "3",
            "match_id": "match-1",
            "team_name": "Argentina",
            "player_name": "Rodrigo De Paul",
            "event_type": "Ball Recovery",
            "x": 72.0,
            "y": 41.0,
            "end_x": None,
            "end_y": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "4",
            "match_id": "match-1",
            "team_name": "Francia",
            "player_name": "Kylian Mbappe",
            "event_type": "Pass",
            "x": None,
            "y": None,
            "end_x": None,
            "end_y": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "5",
            "match_id": "match-1",
            "team_name": "Francia",
            "player_name": "Antoine Griezmann",
            "event_type": "Interception",
            "x": 40.0,
            "y": 50.0,
            "end_x": None,
            "end_y": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
    ]


def _sample_metadata():
    return {
        "match_id": "match-1",
        "competition_name": "Mundial",
        "season_name": "2022",
        "home_team": "Argentina",
        "away_team": "Francia",
        "match_date": "2022-12-18",
    }


def _sample_metrics():
    return {
        "total_events": 5,
        "total_passes": 2,
        "total_shots": 1,
        "progressive_actions": 1,
        "final_third_actions": 2,
        "recoveries": 2,
        "total_carries": 0,
        "total_under_pressure": 2,
        "total_xg": 0.35,
        "field_tilt_index": 68.0,
        "field_tilt_label": "Alto",
        "directness_index": 62.0,
        "directness_label": "Medio",
        "progressive_threat_index": 71.0,
        "progressive_threat_label": "Alto",
        "recovery_height_index": 64.0,
        "recovery_height_label": "Medio",
        "shot_quality_index": 18.0,
        "shot_quality_label": "Bajo",
        "player_influence_score": 74.0,
        "player_influence_label": "Alto",
    }


def test_build_match_context_handles_empty_events_list():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=[],
        metrics=_sample_metrics(),
    )

    assert context["event_summary"]["total_canonical_events"] == 0
    assert context["event_summary"]["event_type_counts"] == {}
    assert context["spatial_summary"]["has_spatial_data"] is False


def test_build_match_context_handles_incomplete_metrics():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=None,
        metrics={"total_events": 0, "total_shots": 0},
    )

    assert context["tactical_metrics"]["total_events"] == 0
    assert context["tactical_metrics"]["total_shots"] == 0
    assert context["tactical_metrics"]["field_tilt_index"] is None
    assert isinstance(context["tactical_summary"]["data_quality_note"], str)


def test_build_match_context_detects_provider_without_coordinates():
    context = build_match_context(
        provider="api_football",
        match_metadata=_sample_metadata(),
        canonical_events=[{"event_type": "Pass", "team_name": "Argentina", "player_name": "Lionel Messi"}],
        metrics={},
        provider_capabilities={"has_event_coordinates": False},
    )

    assert context["provider_context"]["capabilities"]["has_event_coordinates"] is False
    assert "Este provider no entrega coordenadas de eventos." in context["provider_context"]["limitations"]
    assert context["spatial_summary"]["has_spatial_data"] is False


def test_build_match_context_detects_events_with_coordinates():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=_sample_events(),
        metrics=_sample_metrics(),
    )

    assert context["provider_context"]["capabilities"]["has_event_coordinates"] is True
    assert context["spatial_summary"]["has_spatial_data"] is True
    assert context["spatial_summary"]["coordinate_system"] == "StatsBomb 120x80"
    assert context["spatial_summary"]["final_third_actions_count"] == 1


def test_build_match_context_calculates_event_type_counts():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=_sample_events(),
        metrics=_sample_metrics(),
    )

    assert context["event_summary"]["event_type_counts"] == {
        "Pass": 2,
        "Shot": 1,
        "Ball Recovery": 1,
        "Interception": 1,
    }


def test_build_match_context_generates_suggested_questions():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=[],
        metrics={},
    )

    assert isinstance(context["suggested_questions"], list)
    assert len(context["suggested_questions"]) >= 5
    assert all(isinstance(question, str) for question in context["suggested_questions"])
    assert any("métricas propietarias" in question.lower() for question in context["suggested_questions"])


def test_build_match_context_includes_proprietary_metrics_when_present():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=_sample_events(),
        metrics=_sample_metrics(),
    )

    assert context["tactical_metrics"]["field_tilt_index"] == 68.0
    assert context["tactical_metrics"]["progressive_threat_index"] == 71.0
    assert context["tactical_metrics"]["player_influence_score"] == 74.0


def test_build_match_context_does_not_include_raw_events():
    events = _sample_events()
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=events,
        metrics=_sample_metrics(),
    )

    serialized = json.dumps(context, ensure_ascii=False)

    assert "canonical_events" not in context
    assert "raw_events" not in context
    assert '"event_id": "1"' not in serialized
    assert str(events[0]) not in serialized


def test_build_match_context_works_with_selected_player():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=_sample_events(),
        metrics=_sample_metrics(),
        selected_team="Argentina",
        selected_player="Lionel Messi",
    )

    assert context["match"]["selected_team"] == "Argentina"
    assert context["match"]["selected_player"] == "Lionel Messi"
    assert "alta influencia" in context["tactical_summary"]["player_profile"].lower()


def test_build_match_context_returns_json_serializable_dict():
    context = build_match_context(
        provider="statsbomb",
        match_metadata=_sample_metadata(),
        canonical_events=_sample_events(),
        metrics=_sample_metrics(),
        insights=["Insight 1"],
        raw_summary={"provider_status": "ok"},
    )

    payload = json.dumps(context, ensure_ascii=False)

    assert isinstance(context, dict)
    assert '"provider_status": "ok"' in payload
