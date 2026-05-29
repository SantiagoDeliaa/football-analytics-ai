import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.canonical_models import (
    CanonicalMatch,
    CanonicalPlayerMatchStats,
    CanonicalTeamExpectedMetrics,
    build_canonical_id,
    normalize_optional_float,
    safe_get,
)
from src.services.providers.stat_labels_es import (
    get_event_label_es,
    get_expected_metric_label_es,
    get_stat_label_es,
)


def test_build_canonical_id_with_simple_strings():
    assert build_canonical_id("Sportmonks", "Liga Profesional", "2025") == "sportmonks__liga_profesional__2025"


def test_build_canonical_id_ignores_none_and_empty_values():
    assert build_canonical_id("Sportmonks", None, "", "Fixture 123") == "sportmonks__fixture_123"


def test_safe_get_supports_nested_dict_paths():
    payload = {"match": {"teams": {"home": {"name": "River Plate"}}}}

    assert safe_get(payload, ["match", "teams", "home", "name"]) == "River Plate"
    assert safe_get(payload, ["match", "teams", "away", "name"], default="Sin dato") == "Sin dato"


def test_normalize_optional_float_handles_supported_inputs():
    assert normalize_optional_float("1.25") == 1.25
    assert normalize_optional_float(2.5) == 2.5
    assert normalize_optional_float(None) is None
    assert normalize_optional_float("valor-invalido") is None


def test_can_create_minimal_canonical_match_instance():
    match = CanonicalMatch(
        canonical_match_id="sportmonks__fixture_123",
        provider="sportmonks",
        provider_match_id="123",
    )

    assert match.provider == "sportmonks"
    assert match.provider_match_id == "123"
    assert match.metadata == {}


def test_can_create_minimal_canonical_team_expected_metrics_instance():
    expected_metrics = CanonicalTeamExpectedMetrics(
        canonical_match_id="sportmonks__fixture_123",
        provider="sportmonks",
    )

    assert expected_metrics.canonical_match_id == "sportmonks__fixture_123"
    assert expected_metrics.xg is None
    assert expected_metrics.metadata == {}


def test_can_create_minimal_canonical_player_match_stats_instance():
    player_stats = CanonicalPlayerMatchStats(
        canonical_match_id="sportmonks__fixture_123",
        provider="sportmonks",
    )

    assert player_stats.player_id is None
    assert player_stats.stats == {}
    assert player_stats.metadata == {}


def test_metadata_default_factory_does_not_share_mutable_references():
    first_match = CanonicalMatch(
        canonical_match_id="match_1",
        provider="sportmonks",
        provider_match_id="1",
    )
    second_match = CanonicalMatch(
        canonical_match_id="match_2",
        provider="sportmonks",
        provider_match_id="2",
    )

    first_match.metadata["debug"] = "solo-primero"

    assert second_match.metadata == {}


def test_stat_labels_in_spanish_return_expected_values():
    assert get_stat_label_es("passes") == "Pases"
    assert get_event_label_es("goal") == "Gol"
    assert get_expected_metric_label_es("expected-goals") == "Goles esperados"
    assert get_stat_label_es("unknown_key") == "Unknown key"
