import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.providers.provider_capabilities import (
    get_enabled_modules_for_provider,
    get_provider_capabilities,
)
from src.services.providers.provider_registry import (
    get_available_providers,
    get_default_provider,
    get_provider_display_name,
    is_provider_supported,
)


def test_registered_providers_expose_capabilities():
    providers = get_available_providers()

    assert providers
    for provider in providers:
        capabilities = get_provider_capabilities(provider["name"])
        assert capabilities
        assert "has_event_data" in capabilities
        assert "has_event_coordinates" in capabilities


def test_statsbomb_open_data_enables_event_maps():
    enabled_modules = get_enabled_modules_for_provider("statsbomb_open_data")

    assert enabled_modules["event_maps"] is True
    assert enabled_modules["shot_map"] is True
    assert enabled_modules["progressive_actions_map"] is True


def test_sportmonks_does_not_enable_event_maps_without_true_coordinates():
    enabled_modules = get_enabled_modules_for_provider("sportmonks")

    assert enabled_modules["event_maps"] is False
    assert enabled_modules["shot_map"] is False
    assert enabled_modules["progressive_actions_map"] is False


def test_unknown_provider_returns_safe_defaults():
    capabilities = get_provider_capabilities("unknown_provider")
    enabled_modules = get_enabled_modules_for_provider("unknown_provider")

    assert capabilities == {}
    assert enabled_modules == {
        "match_context": False,
        "standings": False,
        "lineups": False,
        "team_stats": False,
        "player_stats": False,
        "event_maps": False,
        "shot_map": False,
        "progressive_actions_map": False,
        "ai_coach": False,
    }


def test_registry_helpers_return_expected_values():
    assert get_default_provider() == "statsbomb_open_data"
    assert is_provider_supported("statsbomb_open_data") is True
    assert is_provider_supported("unknown_provider") is False
    assert get_provider_display_name("api_football") == "API-Football"
