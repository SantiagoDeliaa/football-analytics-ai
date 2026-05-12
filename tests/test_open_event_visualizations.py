import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.open_event_visualizations import create_event_map
from src.services.open_event_visualizations import create_pitch_figure
from src.services.open_event_visualizations import create_player_action_map
from src.services.open_event_visualizations import create_progressive_actions_map
from src.services.open_event_visualizations import create_recoveries_map
from src.services.open_event_visualizations import create_shot_map


def _canonical_events():
    return [
        {
            "event_id": "1",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Lionel Messi",
            "minute": 10,
            "second": 5,
            "event_type": "Pass",
            "x": 42.0,
            "y": 30.0,
            "end_x": 62.0,
            "end_y": 34.0,
            "outcome": "Complete",
            "progressive": True,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "2",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Julian Alvarez",
            "minute": 20,
            "second": 15,
            "event_type": "Shot",
            "x": 101.0,
            "y": 37.0,
            "end_x": None,
            "end_y": None,
            "outcome": "Goal",
            "progressive": False,
            "under_pressure": True,
            "xG": 0.33,
        },
        {
            "event_id": "3",
            "match_id": "m1",
            "team_name": "Argentina",
            "player_name": "Nahuel Molina",
            "minute": 32,
            "second": 10,
            "event_type": "Ball Recovery",
            "x": 70.0,
            "y": 40.0,
            "end_x": None,
            "end_y": None,
            "outcome": None,
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
        {
            "event_id": "4",
            "match_id": "m1",
            "team_name": "Francia",
            "player_name": "Kylian Mbappe",
            "minute": 40,
            "second": 1,
            "event_type": "Duel",
            "x": None,
            "y": None,
            "end_x": None,
            "end_y": None,
            "outcome": "Won",
            "progressive": False,
            "under_pressure": False,
            "xG": 0.0,
        },
    ]


def test_create_pitch_figure_returns_plotly_figure():
    fig = create_pitch_figure("Cancha base")
    assert isinstance(fig, go.Figure)
    assert fig.layout.plot_bgcolor == "#14532d"


def test_create_event_map_handles_empty_events():
    fig = create_event_map([])
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1


def test_create_event_map_renders_traces_for_events_with_coordinates():
    fig = create_event_map(_canonical_events(), selected_team="Argentina")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3


def test_create_shot_map_does_not_break_with_missing_coordinates():
    fig = create_shot_map(_canonical_events())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_create_progressive_actions_map_renders_lines():
    fig = create_progressive_actions_map(_canonical_events(), selected_team="Argentina")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines+markers"


def test_create_recoveries_map_handles_selection():
    fig = create_recoveries_map(_canonical_events(), selected_team="Argentina")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_create_player_action_map_handles_todos_without_breaking():
    fig = create_player_action_map(_canonical_events(), selected_player="Todos")
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) >= 1
