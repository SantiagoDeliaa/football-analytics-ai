import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.open_event_normalizer import normalize_events_to_canonical


def test_normalize_pass_event_to_canonical():
    raw_events = [
        {
            "id": "pass-1",
            "match_id": 123,
            "minute": 12,
            "second": 18,
            "type": {"name": "Pass"},
            "team": {"id": 1, "name": "Argentina"},
            "player": {"id": 10, "name": "Lionel Messi"},
            "location": [45, 30],
            "pass": {"end_location": [65, 34]},
            "under_pressure": True,
        }
    ]

    normalized = normalize_events_to_canonical(raw_events)

    assert len(normalized) == 1
    event = normalized[0]
    assert event["event_id"] == "pass-1"
    assert event["match_id"] == "123"
    assert event["team_id"] == "1"
    assert event["team_name"] == "Argentina"
    assert event["player_id"] == "10"
    assert event["player_name"] == "Lionel Messi"
    assert event["event_type"] == "Pass"
    assert event["x"] == 45.0
    assert event["y"] == 30.0
    assert event["end_x"] == 65.0
    assert event["end_y"] == 34.0
    assert event["outcome"] == "Complete"
    assert event["progressive"] is True
    assert event["under_pressure"] is True


def test_normalize_shot_event_with_xg():
    raw_events = [
        {
            "id": "shot-1",
            "minute": 27,
            "second": 3,
            "type": {"name": "Shot"},
            "team": {"id": 2, "name": "Francia"},
            "player": {"id": 9, "name": "Kylian Mbappe"},
            "location": [101, 39],
            "shot": {
                "outcome": {"name": "Goal"},
                "statsbomb_xg": 0.42,
            },
        }
    ]

    normalized = normalize_events_to_canonical(raw_events, match_id="m-1")

    event = normalized[0]
    assert event["match_id"] == "m-1"
    assert event["event_type"] == "Shot"
    assert event["outcome"] == "Goal"
    assert event["xG"] == 0.42
    assert event["xA"] == 0.0
    assert event["end_x"] is None
    assert event["end_y"] is None


def test_normalize_carry_event_with_end_location():
    raw_events = [
        {
            "id": "carry-1",
            "type": {"name": "Carry"},
            "team": {"id": 1, "name": "Argentina"},
            "player": {"id": 7, "name": "Rodrigo De Paul"},
            "location": [40, 20],
            "carry": {"end_location": [59, 28]},
        }
    ]

    normalized = normalize_events_to_canonical(raw_events, match_id=77)

    event = normalized[0]
    assert event["event_type"] == "Carry"
    assert event["x"] == 40.0
    assert event["end_x"] == 59.0
    assert event["end_y"] == 28.0
    assert event["progressive"] is True


def test_normalize_event_without_player_uses_fallback():
    raw_events = [
        {
            "id": "recovery-1",
            "type": {"name": "Ball Recovery"},
            "team": {"id": 3, "name": "Brasil"},
            "location": [55, 44],
        }
    ]

    normalized = normalize_events_to_canonical(raw_events, match_id="xyz")

    event = normalized[0]
    assert event["player_id"] == "unknown-player"
    assert event["player_name"] == "Jugador desconocido"
    assert event["match_id"] == "xyz"


def test_normalize_event_without_location_does_not_break():
    raw_events = [
        {
            "id": "event-no-location",
            "type": {"name": "Pass"},
            "team": {"id": 1, "name": "Argentina"},
            "player": {"id": 8, "name": "Enzo Fernandez"},
            "pass": {"end_location": [30, 40]},
        }
    ]

    normalized = normalize_events_to_canonical(raw_events)

    event = normalized[0]
    assert event["x"] is None
    assert event["y"] is None
    assert event["end_x"] == 30.0
    assert event["end_y"] == 40.0
    assert event["progressive"] is False


def test_progressive_flag_is_false_below_threshold():
    raw_events = [
        {
            "id": "pass-short",
            "type": {"name": "Pass"},
            "team": {"id": 1, "name": "Argentina"},
            "player": {"id": 10, "name": "Lionel Messi"},
            "location": [50, 20],
            "pass": {"end_location": [64.9, 30]},
        }
    ]

    normalized = normalize_events_to_canonical(raw_events)

    assert normalized[0]["progressive"] is False
