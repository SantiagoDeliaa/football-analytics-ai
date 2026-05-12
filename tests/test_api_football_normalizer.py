import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.api_football_normalizer import normalize_api_football_events_to_canonical


def test_normalize_api_football_events_to_canonical_maps_core_fields():
    raw_events = [
        {
            "time": {"elapsed": 17},
            "team": {"id": 1, "name": "River Plate"},
            "player": {"id": 10, "name": "Esequiel Barco"},
            "type": "Goal",
            "detail": "Normal Goal",
        }
    ]

    canonical = normalize_api_football_events_to_canonical(raw_events, fixture_id=999)

    assert len(canonical) == 1
    event = canonical[0]
    assert event["match_id"] == "999"
    assert event["team_id"] == "1"
    assert event["team_name"] == "River Plate"
    assert event["player_id"] == "10"
    assert event["player_name"] == "Esequiel Barco"
    assert event["minute"] == 17
    assert event["event_type"] == "Goal"
    assert event["outcome"] == "Normal Goal"
    assert event["x"] is None
    assert event["progressive"] is False


def test_normalize_api_football_events_uses_fallbacks_when_player_missing():
    raw_events = [
        {
            "time": {"elapsed": 64},
            "team": {"id": 2, "name": "Boca Juniors"},
            "detail": "Yellow Card",
        }
    ]

    canonical = normalize_api_football_events_to_canonical(raw_events, fixture_id="abc")

    event = canonical[0]
    assert event["event_id"].startswith("abc-0-")
    assert event["player_id"] == "unknown-player"
    assert event["player_name"] == "Jugador desconocido"
    assert event["event_type"] == "Yellow Card"
