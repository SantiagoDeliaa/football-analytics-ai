import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.homography_manager import HomographyManager
from src.utils.quality_config import DELTA_H_CUT


def test_cut_detected_triggers_reacquire():
    manager = HomographyManager()
    manager.mode = "TRACK"
    manager.current_H = np.eye(3, dtype=np.float32)
    source = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [50.0, 60.0],
            [160.0, 55.0],
            [170.0, 180.0],
            [45.0, 170.0],
        ],
        dtype=np.float32,
    )
    ok = manager.update(source, target)
    assert ok is True
    assert manager.last_delta is not None
    assert manager.last_delta > DELTA_H_CUT
    assert manager.cut_detected is True
    assert manager.mode in {"TRACK", "REACQUIRE"}
    if manager.mode == "REACQUIRE":
        assert manager.homography_state == "REACQUIRE"
    else:
        assert manager.homography_state == "STABLE"
