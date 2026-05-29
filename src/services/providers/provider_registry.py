from __future__ import annotations

from copy import deepcopy

from .provider_capabilities import PROVIDER_CAPABILITIES

_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "statsbomb_open_data": "StatsBomb Open Data",
    "api_football": "API-Football",
    "sportmonks": "Sportmonks",
}
_DEFAULT_PROVIDER = "statsbomb_open_data"


def _normalize_provider_name(provider_name: str | None) -> str:
    return str(provider_name or "").strip().lower()


def get_available_providers() -> list[dict[str, str]]:
    providers: list[dict[str, str]] = []
    for provider_name, display_name in _PROVIDER_DISPLAY_NAMES.items():
        if provider_name not in PROVIDER_CAPABILITIES:
            continue
        providers.append({"name": provider_name, "display_name": display_name})
    return deepcopy(providers)


def get_provider_display_name(provider_name: str) -> str:
    raw_name = str(provider_name or "").strip()
    normalized_name = _normalize_provider_name(provider_name)
    return _PROVIDER_DISPLAY_NAMES.get(normalized_name, raw_name or "Proveedor desconocido")


def get_default_provider() -> str:
    return _DEFAULT_PROVIDER


def is_provider_supported(provider_name: str) -> bool:
    normalized_name = _normalize_provider_name(provider_name)
    return normalized_name in _PROVIDER_DISPLAY_NAMES and normalized_name in PROVIDER_CAPABILITIES
