from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api import main


def test_initialize_application_state_delegates_to_event_data_persistence(monkeypatch):
    monkeypatch.setattr(
        main,
        "initialize_event_data_persistence",
        lambda: {"persistence_backend": "postgres", "storage_backend": "local"},
    )

    result = main.initialize_application_state()

    assert result == {
        "persistence_backend": "postgres",
        "storage_backend": "local",
    }
