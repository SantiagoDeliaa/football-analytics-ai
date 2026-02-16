"""
Tactical Metrics Calculator - Paso 3
Calcula métricas de comportamiento colectivo para análisis táctico de fútbol.

Métricas implementadas:
- Compactación del equipo (área ocupada)
- Altura de presión (posición promedio)
- Amplitud ofensiva (dispersión horizontal)
- Centroide del equipo
- Stretch Index (elongación)
- Métricas de scouting: block compactness + dual defensive line height
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial import ConvexHull
from collections import deque


class TacticalMetricsCalculator:
    """
    Calcula métricas tácticas para equipos de fútbol basándose en posiciones.
    """

    def __init__(self, field_width: float = 105.0, field_length: float = 68.0):
        """
        Args:
            field_width: Ancho del campo en metros (105m FIFA standard)
            field_length: Largo del campo en metros (68m FIFA standard)
        """
        self.field_width = field_width
        self.field_length = field_length

    def _clamp_positions(self, positions: np.ndarray) -> np.ndarray:
        if positions is None or len(positions) == 0:
            return np.array([], dtype=np.float32).reshape(0, 2)

        positions = np.asarray(positions, dtype=np.float32).copy()
        positions[:, 0] = np.clip(positions[:, 0], 0, 105)
        positions[:, 1] = np.clip(positions[:, 1], 0, 68)
        return positions

    def calculate_all_metrics(self, positions: np.ndarray) -> Dict:
        """
        Calcula todas las métricas para un conjunto de posiciones.

        Args:
            positions: Array (N, 2) con posiciones [x, y] en metros

        Returns:
            Dict con todas las métricas calculadas
        """
        positions = self._clamp_positions(positions)

        if len(positions) < 3:
            metrics = self._empty_metrics()
            block_compactness = self.calculate_block_compactness(positions)
            def_line = self.calculate_defensive_line_height_dual(positions)
            metrics.update({
                'block_depth_m': block_compactness['depth_m'],
                'block_width_m': block_compactness['width_m'],
                'block_area_m2': block_compactness['area_m2'],
                'def_line_left_m': def_line['def_line_left_m'],
                'def_line_right_m': def_line['def_line_right_m'],
                'num_players': len(positions),
            })
            return metrics

        block_compactness = self.calculate_block_compactness(positions)
        def_line = self.calculate_defensive_line_height_dual(positions)

        return {
            'compactness': self.calculate_compactness(positions),
            'pressure_height': self.calculate_pressure_height(positions),
            'offensive_width': self.calculate_offensive_width(positions),
            'centroid': self.calculate_centroid(positions),
            'stretch_index': self.calculate_stretch_index(positions),
            'defensive_depth': self.calculate_defensive_depth(positions),
            'block_depth_m': block_compactness['depth_m'],
            'block_width_m': block_compactness['width_m'],
            'block_area_m2': block_compactness['area_m2'],
            'def_line_left_m': def_line['def_line_left_m'],
            'def_line_right_m': def_line['def_line_right_m'],
            'num_players': len(positions)
        }

    def calculate_compactness(self, positions: np.ndarray) -> float:
        if len(positions) < 3:
            return 0.0

        try:
            hull = ConvexHull(positions)
            area = hull.volume
            return float(area)
        except Exception:
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            return float(x_range * y_range)

    def calculate_pressure_height(self, positions: np.ndarray) -> float:
        if len(positions) == 0:
            return 0.0
        return float(np.mean(positions[:, 0]))

    def calculate_offensive_width(self, positions: np.ndarray) -> float:
        if len(positions) == 0:
            return 0.0
        y_range = positions[:, 1].max() - positions[:, 1].min()
        return float(y_range)

    def calculate_centroid(self, positions: np.ndarray) -> Tuple[float, float]:
        if len(positions) == 0:
            return (0.0, 0.0)

        centroid_x = float(np.mean(positions[:, 0]))
        centroid_y = float(np.mean(positions[:, 1]))
        return (centroid_x, centroid_y)

    def calculate_stretch_index(self, positions: np.ndarray) -> float:
        if len(positions) < 2:
            return 1.0

        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])

        if std_y < 0.1:
            return 10.0 if std_x > 0.1 else 1.0

        return float(std_x / std_y)

    def calculate_defensive_depth(self, positions: np.ndarray) -> float:
        if len(positions) == 0:
            return 0.0

        x_range = positions[:, 0].max() - positions[:, 0].min()
        return float(x_range)

    def calculate_block_compactness(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Block compactness por bounding box del bloque del equipo.
        """
        if len(positions) < 2:
            return {'depth_m': 0.0, 'width_m': 0.0, 'area_m2': 0.0}

        positions = self._clamp_positions(positions)
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]

        depth_m = float(np.max(x_vals) - np.min(x_vals))
        width_m = float(np.max(y_vals) - np.min(y_vals))
        area_m2 = float(depth_m * width_m)

        return {
            'depth_m': depth_m,
            'width_m': width_m,
            'area_m2': area_m2,
        }

    def calculate_defensive_line_height_dual(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Calcula dos alturas de línea defensiva:
        - def_line_left_m: promedio de los N jugadores con menor X (defendiendo x=0)
        - def_line_right_m: promedio de los N jugadores con mayor X (defendiendo x=105)
        """
        if len(positions) == 0:
            return {'def_line_left_m': 0.0, 'def_line_right_m': 0.0}

        positions = self._clamp_positions(positions)
        x_vals = np.sort(positions[:, 0])
        n = min(4, len(x_vals))

        return {
            'def_line_left_m': float(np.mean(x_vals[:n])),
            'def_line_right_m': float(np.mean(x_vals[-n:])),
        }

    def _empty_metrics(self) -> Dict:
        return {
            'compactness': 0.0,
            'pressure_height': 0.0,
            'offensive_width': 0.0,
            'centroid': (0.0, 0.0),
            'stretch_index': 1.0,
            'defensive_depth': 0.0,
            'block_depth_m': 0.0,
            'block_width_m': 0.0,
            'block_area_m2': 0.0,
            'def_line_left_m': 0.0,
            'def_line_right_m': 0.0,
            'num_players': 0
        }


class TacticalMetricsTracker:
    """
    Rastrea métricas tácticas a lo largo del tiempo y calcula tendencias.
    """

    def __init__(self, history_size: int = 300):
        self.history_size = history_size
        self.valid_frames_count = 0
        self.metrics_history = {
            'compactness': deque(maxlen=history_size),
            'pressure_height': deque(maxlen=history_size),
            'offensive_width': deque(maxlen=history_size),
            'centroid_x': deque(maxlen=history_size),
            'centroid_y': deque(maxlen=history_size),
            'stretch_index': deque(maxlen=history_size),
            'defensive_depth': deque(maxlen=history_size),
            'block_depth_m': deque(maxlen=history_size),
            'block_width_m': deque(maxlen=history_size),
            'block_area_m2': deque(maxlen=history_size),
            'def_line_left_m': deque(maxlen=history_size),
            'def_line_right_m': deque(maxlen=history_size),
            'frame_number': deque(maxlen=history_size)
        }

    def update(self, metrics: Dict, frame_number: int):
        self.metrics_history['compactness'].append(metrics['compactness'])
        self.metrics_history['pressure_height'].append(metrics['pressure_height'])
        self.metrics_history['offensive_width'].append(metrics['offensive_width'])
        self.metrics_history['centroid_x'].append(metrics['centroid'][0])
        self.metrics_history['centroid_y'].append(metrics['centroid'][1])
        self.metrics_history['stretch_index'].append(metrics['stretch_index'])
        self.metrics_history['defensive_depth'].append(metrics['defensive_depth'])
        self.metrics_history['block_depth_m'].append(metrics.get('block_depth_m', 0.0))
        self.metrics_history['block_width_m'].append(metrics.get('block_width_m', 0.0))
        self.metrics_history['block_area_m2'].append(metrics.get('block_area_m2', 0.0))
        self.metrics_history['def_line_left_m'].append(metrics.get('def_line_left_m', 0.0))
        self.metrics_history['def_line_right_m'].append(metrics.get('def_line_right_m', 0.0))
        self.metrics_history['frame_number'].append(frame_number)

        if metrics.get('num_players', 0) >= 2:
            self.valid_frames_count += 1

    def get_statistics(self) -> Dict:
        stats = {}

        for metric_name, values in self.metrics_history.items():
            if metric_name == 'frame_number':
                continue

            if len(values) == 0:
                stats[metric_name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
                continue

            values_array = np.array(list(values))
            stats[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'current': float(values_array[-1]) if len(values_array) > 0 else 0.0
            }

        stats['valid_frames'] = self.valid_frames_count
        return stats

    def get_trend(self, metric_name: str, window: int = 30) -> str:
        if metric_name not in self.metrics_history:
            return "unknown"

        values = list(self.metrics_history[metric_name])

        if len(values) < window:
            return "insufficient_data"

        recent_values = values[-window:]
        first_half = np.mean(recent_values[:window//2])
        second_half = np.mean(recent_values[window//2:])

        change_pct = abs((second_half - first_half) / first_half) * 100 if first_half != 0 else 0

        if change_pct < 5:
            return "stable"
        elif second_half > first_half:
            return "increasing"
        else:
            return "decreasing"

    def export_to_dict(self) -> Dict:
        return {
            metric_name: list(values)
            for metric_name, values in self.metrics_history.items()
        }

    def export_to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            metric_name: np.array(list(values))
            for metric_name, values in self.metrics_history.items()
        }
