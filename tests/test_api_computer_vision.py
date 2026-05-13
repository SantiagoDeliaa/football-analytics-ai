from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.main import app


client = TestClient(app)


def test_computer_vision_analyze_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.computer_vision.analyze_video",
        lambda **kwargs: {
            "source": "mock",
            "status_message": "demo",
            "video_name": "clip.mp4",
            "duration_seconds": 10.0,
            "total_frames": 250,
            "fps": 25.0,
            "health_summary": {"demo_mode": "stable"},
            "formations": {"team1": {"most_common": "4-3-3"}, "team2": {"most_common": "4-4-2"}},
            "metrics": {"team1": {}, "team2": {}},
            "timeline": {
                "pressure_height": {"frames": [], "team1": [], "team2": []},
                "compactness": {"frames": [], "team1": [], "team2": []},
                "offensive_width": {"frames": [], "team1": [], "team2": []},
            },
            "possession": None,
            "scouting": {
                "confidence": {
                    "team1": {"label": "Alta", "score": 80},
                    "team2": {"label": "Media", "score": 60},
                },
                "bullets": {"team1": ["A"], "team2": ["B"]},
            },
            "exports": {"json": True, "csv": True, "pdf": False},
            "warnings": [],
            "interpretation": ["ok"],
        },
    )

    response = client.post(
        "/api/v1/computer-vision/analyze",
        data={
            "source_mode": "soccernet",
            "soccernet_path": "videos/demo.mp4",
            "config": json.dumps(
                {
                    "model_name": "yolov8n.pt",
                    "confidence": 0.25,
                    "image_size": 640,
                    "only_person": True,
                    "segment_mode": False,
                    "start_seconds": 0,
                    "duration_seconds": 10,
                    "full_field_approx": False,
                    "enable_radar": True,
                    "enable_analytics": True,
                    "enable_possession": True,
                    "disable_inertia": False,
                    "export_profile": "debug_sampled",
                    "sample_stride": 10,
                    "topk_frames": 20,
                    "enable_compression": True,
                }
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["video_name"] == "clip.mp4"


def test_computer_vision_analyze_returns_400_with_invalid_config():
    response = client.post(
        "/api/v1/computer-vision/analyze",
        data={
            "source_mode": "soccernet",
            "soccernet_path": "videos/demo.mp4",
            "config": "{invalid json",
        },
    )

    assert response.status_code == 400


def test_computer_vision_job_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.computer_vision.create_job",
        lambda **kwargs: {
            "job_id": "job-1",
            "status": "queued",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "video_name": "clip.mp4",
            "result": None,
            "error": None,
        },
    )

    response = client.post(
        "/api/v1/computer-vision/jobs",
        data={
            "source_mode": "soccernet",
            "soccernet_path": "videos/demo.mp4",
            "config": json.dumps(
                {
                    "model_name": "yolov8n.pt",
                    "confidence": 0.25,
                    "image_size": 640,
                    "only_person": True,
                    "segment_mode": False,
                    "start_seconds": 0,
                    "duration_seconds": 10,
                    "full_field_approx": False,
                    "enable_radar": True,
                    "enable_analytics": True,
                    "enable_possession": True,
                    "disable_inertia": False,
                    "export_profile": "debug_sampled",
                    "sample_stride": 10,
                    "topk_frames": 20,
                    "enable_compression": True,
                }
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"


def test_computer_vision_job_accepts_custom_model_assets(monkeypatch):
    captured: dict = {}

    def _fake_create_job(**kwargs):
        captured.update(kwargs)
        return {
            "job_id": "job-custom",
            "status": "queued",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "video_name": "clip.mp4",
            "result": None,
            "error": None,
        }

    monkeypatch.setattr("api.routers.computer_vision.create_job", _fake_create_job)

    response = client.post(
        "/api/v1/computer-vision/jobs",
        data={
            "source_mode": "soccernet",
            "soccernet_path": "videos/demo.mp4",
            "config": json.dumps(
                {
                    "model_name": "yolov8m.pt",
                    "player_model_source": "custom",
                    "ball_model_source": "custom",
                    "pitch_source": "soccana",
                }
            ),
        },
        files={
            "player_model_file": ("players.pt", b"fake-players", "application/octet-stream"),
            "ball_model_file": ("ball.pt", b"fake-ball", "application/octet-stream"),
        },
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-custom"
    assert captured["config"].player_model_source == "custom"
    assert captured["config"].ball_model_source == "custom"
    assert captured["config"].pitch_source == "soccana"
    assert captured["player_model_file"].filename == "players.pt"
    assert captured["ball_model_file"].filename == "ball.pt"


def test_computer_vision_job_status_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routers.computer_vision.get_job",
        lambda job_id: {
            "job_id": job_id,
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:05+00:00",
            "video_name": "clip.mp4",
            "result": {
                "source": "api",
                "status_message": None,
                "video_name": "clip.mp4",
                "duration_seconds": 10,
                "total_frames": 250,
                "fps": 25,
                "health_summary": {},
                "formations": {"team1": {"most_common": "4-3-3"}, "team2": {"most_common": "4-4-2"}},
                "metrics": {"team1": {}, "team2": {}},
                "timeline": {
                    "pressure_height": {"frames": [], "team1": [], "team2": []},
                    "compactness": {"frames": [], "team1": [], "team2": []},
                    "offensive_width": {"frames": [], "team1": [], "team2": []},
                },
                "possession": None,
                "scouting": {
                    "confidence": {
                        "team1": {"label": "Alta", "score": 80},
                        "team2": {"label": "Media", "score": 60},
                    },
                    "bullets": {"team1": ["A"], "team2": ["B"]},
                },
                "exports": {"json": True, "csv": True, "pdf": False},
                "warnings": [],
                "interpretation": ["ok"],
            },
            "error": None,
        },
    )

    response = client.get("/api/v1/computer-vision/jobs/job-1")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
