import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.controllers.process_video import build_export_payload


def test_build_export_payload_debug_sampled_schema():
    stats_data = {
        "fps": 25,
        "duration_seconds": 10.0,
        "total_frames": 250,
        "metrics": {"team1": {}, "team2": {}},
        "formations": {
            "team1": {"most_common": "4-4-2", "timeline": ["4-4-2", "4-3-3", "4-4-2"]},
            "team2": {"most_common": "4-3-3", "timeline": ["4-3-3", "4-3-3"]},
        },
        "timeline": {
            "health": [
                {"frame_idx": 0, "reproj_error_m": 0.6, "delta_H": 0.05, "tracks_active": 10, "churn_ratio": 0.1, "ball_detected": True, "possession_state": "team1", "max_player_speed_mps": 5.0},
                {"frame_idx": 1, "reproj_error_m": 0.5, "delta_H": 0.04, "tracks_active": 11, "churn_ratio": 0.12, "ball_detected": True, "possession_state": "team2", "max_player_speed_mps": 5.2},
                {"frame_idx": 2, "reproj_error_m": 0.7, "delta_H": 0.03, "tracks_active": 9, "churn_ratio": 0.08, "ball_detected": False, "possession_state": "contested", "max_player_speed_mps": 4.9},
                {"frame_idx": 3, "reproj_error_m": 0.8, "delta_H": 0.06, "tracks_active": 12, "churn_ratio": 0.11, "ball_detected": True, "possession_state": "team1", "max_player_speed_mps": 5.4},
                {"frame_idx": 4, "reproj_error_m": 0.9, "delta_H": 0.07, "tracks_active": 8, "churn_ratio": 0.15, "ball_detected": False, "possession_state": "team2", "max_player_speed_mps": 4.7},
            ]
        },
        "possession": {"team1_possession_pct": 52.0, "team2_possession_pct": 48.0},
    }
    cfg = {
        "sample_stride": 2,
        "topk_frames": 3,
        "topk_formations": 2,
        "version": "telemetry_export_v1",
        "heatmap_downsample_shape": (5, 5),
    }
    payload = build_export_payload(stats_data, "debug_sampled", cfg)
    assert payload["export_profile"] == "debug_sampled"
    assert payload["version"] == "telemetry_export_v1"
    assert payload["formations"]["team1"]["most_common"] == "4-4-2"
    assert len(payload["formations"]["team1"]["top_k"]) <= 2
    assert payload["timeline"]["health_sampling_stride"] == 2
    assert len(payload["timeline"]["series_frames"]) == len(payload["timeline"]["health_sampled"])
