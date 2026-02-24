import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.controllers.tactical_metrics import TacticalMetricsCalculator


def test_pressure_height_not_duplicate():
    rng = np.random.default_rng(42)
    calculator = TacticalMetricsCalculator()
    pressure_vals = []
    centroid_vals = []
    for _ in range(120):
        positions = rng.uniform(low=[0, 0], high=[105, 68], size=(10, 2))
        metrics = calculator.calculate_all_metrics(positions)
        pressure_vals.append(metrics["pressure_height"])
        centroid_vals.append(metrics["centroid"][0])
    pressure_arr = np.array(pressure_vals, dtype=np.float32)
    centroid_arr = np.array(centroid_vals, dtype=np.float32)
    identical_ratio = float(np.mean(np.isclose(pressure_arr, centroid_arr, atol=1e-6)))
    assert identical_ratio <= 0.95


if __name__ == "__main__":
    test_pressure_height_not_duplicate()
