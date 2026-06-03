from __future__ import annotations

from src.services.storage import event_data_repository as repository


class FakeRepository:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True


def test_initialize_event_data_persistence_returns_active_backends(monkeypatch):
    fake_repository = FakeRepository()

    monkeypatch.setattr(repository, "_get_event_data_repository", lambda: fake_repository)
    monkeypatch.setattr(
        repository,
        "load_persistence_settings",
        lambda: type(
            "Settings",
            (),
            {
                "persistence_backend": "postgres",
                "storage_backend": "r2",
            },
        )(),
    )

    result = repository.initialize_event_data_persistence()

    assert fake_repository.initialized is True
    assert result == {
        "persistence_backend": "postgres",
        "storage_backend": "r2",
    }
