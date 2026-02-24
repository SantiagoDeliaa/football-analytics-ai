import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.homography_manager import HomographyManager


def make_points(scale=1.0):
    src = np.array([
        [100.0, 50.0],
        [540.0, 50.0],
        [540.0, 370.0],
        [100.0, 370.0]
    ], dtype=np.float32)
    tgt = np.array([
        [0.0, 0.0],
        [105.0 * scale, 0.0],
        [105.0 * scale, 68.0 * scale],
        [0.0, 68.0 * scale]
    ], dtype=np.float32)
    return src, tgt


def test_warmup_anchor_and_track():
    m = HomographyManager(debug=False)
    m.warmup_frames = 3
    for _ in range(3):
        src, tgt = make_points(1.0)
        m.update(src, tgt, None, (640, 400))
    assert m.mode in ("TRACK", "INERTIA")
    assert m.get_transformer() is not None


def test_reacquire_on_cut_and_recover():
    m = HomographyManager(debug=False)
    m.warmup_frames = 2
    for _ in range(2):
        src, tgt = make_points(1.0)
        m.update(src, tgt, None, (640, 400))
    assert m.mode in ("TRACK", "INERTIA")
    src2, tgt2 = make_points(1.5)
    m.update(src2, tgt2, None, (640, 400))
    if m.cut_detected:
        m.start_reacquire()
    assert m.mode in ("REACQUIRE", "INERTIA")
    for _ in range(3):
        src3, tgt3 = make_points(1.0)
        m.update(src3, tgt3, None, (640, 400))
        m.tick_reacquire()
    assert m.mode in ("TRACK", "INERTIA")
    assert m.get_transformer() is not None


if __name__ == "__main__":
    test_warmup_anchor_and_track()
    test_reacquire_on_cut_and_recover()
