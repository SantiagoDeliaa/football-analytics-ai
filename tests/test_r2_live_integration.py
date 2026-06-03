from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.services.storage.settings import load_persistence_settings
from src.services.storage.storage_service import build_match_object_key
from src.services.storage.storage_service import get_storage_service


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_R2_TESTS") != "1",
    reason="Set RUN_LIVE_R2_TESTS=1 para ejecutar pruebas reales contra Cloudflare R2.",
)


def test_r2_storage_service_live_roundtrip():
    settings = load_persistence_settings()
    if settings.storage_backend != "r2":
        pytest.skip("El backend de storage configurado no es R2.")

    service = get_storage_service(settings)
    object_key = build_match_object_key(
        organization_id="local_demo",
        match_id="pytest-r2-live-001",
        category="canonical",
        version="v1",
        file_name="events.json",
    )
    payload = {"source": "pytest", "ok": True}

    metadata = service.save_json(payload, object_key)
    assert metadata["bucket"] == settings.r2_bucket_name
    assert metadata["object_key"] == object_key
    assert service.exists(object_key) is True
    assert service.load_json(object_key) == payload

    service.delete(object_key)
    assert service.exists(object_key) is False
