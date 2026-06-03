from __future__ import annotations

from src.services.storage import event_data_repository as repository


class FakeRepository:
    def save_processed_match(self, **kwargs):
        return {"provider": kwargs["provider"], "match_id": kwargs["match_id"]}

    def get_processed_matches(self, limit: int = 20):
        return [{"match_id": "1", "provider": "statsbomb"}]

    def get_processed_match(self, provider: str, match_id: str):
        return {"provider": provider, "match_id": match_id}

    def load_processed_match_payloads(self, *, provider: str, match_id: str):
        return {"metadata": {"provider": provider, "match_id": match_id}}

    def has_processed_match(self, provider: str, match_id: str):
        return True

    def delete_processed_match(self, provider: str, match_id: str):
        return {"ok": True, "message": f"deleted {provider}:{match_id}"}


def test_event_data_repository_dispatches_to_selected_backend(monkeypatch):
    fake_repository = FakeRepository()
    monkeypatch.setattr(repository, "_get_event_data_repository", lambda: fake_repository)

    saved = repository.save_processed_match(
        provider="StatsBomb Open Data",
        match_id="3895302",
        match_metadata={},
        raw_events=[],
        canonical_events=[],
        metrics={},
    )
    rows = repository.get_processed_matches()
    row = repository.get_processed_match("StatsBomb Open Data", "3895302")
    payloads = repository.load_processed_match_payloads("StatsBomb Open Data", "3895302")
    has_match = repository.has_processed_match("StatsBomb Open Data", "3895302")
    deleted = repository.delete_processed_match("StatsBomb Open Data", "3895302")

    assert saved["match_id"] == "3895302"
    assert rows == [{"match_id": "1", "provider": "statsbomb"}]
    assert row == {"provider": "StatsBomb Open Data", "match_id": "3895302"}
    assert payloads == {"metadata": {"provider": "StatsBomb Open Data", "match_id": "3895302"}}
    assert has_match is True
    assert deleted["ok"] is True
