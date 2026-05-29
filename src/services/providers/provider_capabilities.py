from __future__ import annotations

from copy import deepcopy

CapabilityValue = bool | str
ProviderCapabilities = dict[str, CapabilityValue]

_CAPABILITY_DEFAULTS: ProviderCapabilities = {
    "has_fixtures": False,
    "has_standings": False,
    "has_lineups": False,
    "has_formations": False,
    "has_team_stats": False,
    "has_player_stats": False,
    "has_event_data": False,
    "has_event_coordinates": False,
    "has_xg": False,
    "has_xa": False,
    "has_pressures": False,
    "has_360_data": False,
    "has_tracking": False,
    "has_live_data": False,
    "has_historical_data": False,
    "has_current_seasons": False,
    "has_argentina_coverage": False,
    "has_south_america_coverage": False,
    "has_player_profiles": False,
    "has_transfers": False,
    "has_assets": False,
}


def _build_capabilities(**overrides: CapabilityValue) -> ProviderCapabilities:
    capabilities = deepcopy(_CAPABILITY_DEFAULTS)
    capabilities.update(overrides)
    return capabilities


PROVIDER_CAPABILITIES: dict[str, ProviderCapabilities] = {
    "statsbomb_open_data": _build_capabilities(
        has_event_data=True,
        has_event_coordinates=True,
        has_xg=True,
        has_pressures=True,
        has_historical_data=True,
        has_current_seasons=False,
        has_argentina_coverage="limited",
        has_south_america_coverage="limited",
        has_live_data=False,
        has_tracking=False,
    ),
    "api_football": _build_capabilities(
        has_fixtures=True,
        has_standings=True,
        has_lineups=True,
        has_formations=True,
        has_team_stats=True,
        has_player_stats=True,
        has_event_data=True,
        has_event_coordinates=False,
        has_xg="limited_or_unknown",
        has_live_data=True,
        has_historical_data=True,
        has_current_seasons="limited_by_plan",
        has_argentina_coverage=True,
        has_south_america_coverage=True,
        has_player_profiles=True,
        has_transfers=True,
        has_assets=True,
    ),
    "sportmonks": _build_capabilities(
        has_fixtures=True,
        has_standings=True,
        has_lineups=True,
        has_formations=True,
        has_team_stats=True,
        has_player_stats=True,
        has_event_data="limited_or_plan_dependent",
        has_event_coordinates="limited_or_unknown",
        has_xg="plan_dependent",
        has_xa="plan_dependent",
        has_live_data=True,
        has_historical_data=True,
        has_current_seasons=True,
        has_argentina_coverage=True,
        has_south_america_coverage=True,
        has_player_profiles=True,
        has_transfers=True,
        has_assets=True,
    ),
}


def _normalize_provider_name(provider_name: str | None) -> str:
    return str(provider_name or "").strip().lower()


def get_provider_capabilities(provider_name: str) -> ProviderCapabilities:
    normalized_name = _normalize_provider_name(provider_name)
    return deepcopy(PROVIDER_CAPABILITIES.get(normalized_name, {}))


def provider_has_capability(provider_name: str, capability: str) -> bool:
    normalized_name = _normalize_provider_name(provider_name)
    normalized_capability = str(capability or "").strip()
    if not normalized_name or not normalized_capability:
        return False
    return PROVIDER_CAPABILITIES.get(normalized_name, {}).get(normalized_capability) is True


def get_enabled_modules_for_provider(provider_name: str) -> dict[str, bool]:
    match_context_enabled = any(
        (
            provider_has_capability(provider_name, "has_fixtures"),
            provider_has_capability(provider_name, "has_historical_data"),
            provider_has_capability(provider_name, "has_current_seasons"),
        )
    )
    team_stats_enabled = provider_has_capability(provider_name, "has_team_stats")
    player_stats_enabled = provider_has_capability(provider_name, "has_player_stats")
    event_data_enabled = provider_has_capability(provider_name, "has_event_data")
    spatial_event_maps_enabled = provider_has_capability(provider_name, "has_event_coordinates")

    return {
        "match_context": match_context_enabled,
        "standings": provider_has_capability(provider_name, "has_standings"),
        "lineups": any(
            (
                provider_has_capability(provider_name, "has_lineups"),
                provider_has_capability(provider_name, "has_formations"),
            )
        ),
        "team_stats": team_stats_enabled,
        "player_stats": player_stats_enabled,
        "event_maps": spatial_event_maps_enabled,
        "shot_map": spatial_event_maps_enabled,
        "progressive_actions_map": spatial_event_maps_enabled,
        "ai_coach": any(
            (
                match_context_enabled,
                team_stats_enabled,
                player_stats_enabled,
                event_data_enabled,
            )
        ),
    }
