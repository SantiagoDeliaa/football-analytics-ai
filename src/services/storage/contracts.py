from __future__ import annotations

from typing import Any, Protocol


class EventDataRepository(Protocol):
    def initialize(self) -> None:
        ...

    def save_processed_match(
        self,
        *,
        provider: str,
        match_id: str,
        match_metadata: dict[str, Any],
        raw_events: Any,
        canonical_events: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def get_processed_matches(self, limit: int = 20) -> list[dict[str, Any]]:
        ...

    def get_processed_match(self, provider: str, match_id: str) -> dict[str, Any] | None:
        ...

    def load_processed_match_payloads(
        self,
        *,
        provider: str,
        match_id: str,
    ) -> dict[str, Any] | None:
        ...

    def has_processed_match(self, provider: str, match_id: str) -> bool:
        ...

    def delete_processed_match(self, provider: str, match_id: str) -> dict[str, Any]:
        ...


class StorageService(Protocol):
    def save_json(self, payload: Any, object_key: str) -> dict[str, Any]:
        ...

    def load_json(self, object_key: str) -> Any:
        ...

    def save_file(self, source_path: str, object_key: str) -> dict[str, Any]:
        ...

    def exists(self, object_key: str) -> bool:
        ...

    def get_uri(self, object_key: str) -> str:
        ...

    def build_object_key(self, *parts: str) -> str:
        ...

    def delete(self, object_key: str) -> None:
        ...
