import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_homography_regression_fixture():
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "homography_sample.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    health = data.get("health_summary", {})
    fallback_ratio = health.get("fallback_ratio", 1.0)
    valid_frames = health.get("valid_frames", 0)
    total_frames = data.get("total_frames", 0)
    valid_ratio = valid_frames / total_frames if total_frames else 0.0
    assert fallback_ratio < 0.5
    assert valid_frames > 0
    assert valid_ratio >= 0.5


if __name__ == "__main__":
    test_homography_regression_fixture()
