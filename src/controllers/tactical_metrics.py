"""
Tactical Metrics Calculator - Paso 3
Calcula métricas de comportamiento colectivo para análisis táctico de fútbol.

Métricas implementadas:
- Compactación del equipo (área ocupada)
- Altura de presión (posición promedio)
- Amplitud ofensiva (dispersión horizontal)
- Centroide del equipo
- Stretch Index (elongación)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import ConvexHull
from collections import deque


class TacticalMetricsCalculator:
    """
    Calcula métricas tácticas para equipos de fútbol basándose en posiciones.
    """

    def __init__(self, field_width: float = 105.0, field_length: float = 68.0, min_players_for_metrics: int = 6):
        """
        Args:
            field_width: Ancho del campo en metros (105m FIFA standard)
            field_length: Largo del campo en metros (68m FIFA standard)
        """
        self.field_width = field_width
        self.field_length = field_length
        self.min_players_for_metrics = max(1, int(min_players_for_metrics))

    def calculate_all_metrics(self, positions: np.ndarray) -> Optional[Dict]:
        """
        Calcula todas las métricas para un conjunto de posiciones.

        Args:
            positions: Array (N, 2) con posiciones [x, y] en metros

        Returns:
            Dict con todas las métricas calculadas
        """
        if len(positions) < self.min_players_for_metrics:
            return None

        block = self.calculate_block_compactness(positions)
        def_line = self.calculate_defensive_line_height_dual(positions)

        return {
            'compactness': self.calculate_compactness(positions),
            'pressure_height': self.calculate_pressure_height(positions),
            'offensive_width': self.calculate_offensive_width(positions),
            'centroid': self.calculate_centroid(positions),
            'stretch_index': self.calculate_stretch_index(positions),
            'defensive_depth': self.calculate_defensive_depth(positions),
            'block_depth_m': block['depth_m'],
            'block_width_m': block['width_m'],
            'block_area_m2': block['area_m2'],
            'def_line_left_m': def_line['def_line_left_m'],
            'def_line_right_m': def_line['def_line_right_m'],
            'num_players': len(positions)
        }

    def calculate_block_compactness(self, positions: np.ndarray) -> Dict:
        if len(positions) < 2:
            return {'depth_m': 0.0, 'width_m': 0.0, 'area_m2': 0.0}
        x = np.clip(positions[:, 0], 0, 105)
        y = np.clip(positions[:, 1], 0, 68)
        depth = float(np.max(x) - np.min(x))
        width = float(np.max(y) - np.min(y))
        area = float(depth * width)
        return {'depth_m': depth, 'width_m': width, 'area_m2': area}

    def calculate_defensive_line_height_dual(self, positions: np.ndarray) -> Dict:
        if len(positions) == 0:
            return {'def_line_left_m': 0.0, 'def_line_right_m': 0.0}
        x = np.clip(positions[:, 0], 0, 105)
        order = np.argsort(x)
        x_sorted = x[order]
        n = int(min(4, len(x_sorted)))
        left = float(np.mean(x_sorted[:n])) if n > 0 else 0.0
        right = float(np.mean(x_sorted[-n:])) if n > 0 else 0.0
        return {'def_line_left_m': left, 'def_line_right_m': right}

    def calculate_compactness(self, positions: np.ndarray) -> float:
        """
        Calcula la compactación del equipo (área ocupada).

        Compactación = Área del polígono convexo que contiene a los jugadores

        Menor área = Mayor compactación (equipo más junto)
        Mayor área = Menor compactación (equipo más disperso)

        Returns:
            Área en metros cuadrados
        """
        if len(positions) < 3:
            return 0.0

        try:
            hull = ConvexHull(positions)
            area = hull.volume  # En 2D, 'volume' es el área
            return float(area)
        except Exception:
            # Si los puntos son colineales, calcular área del rectángulo
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            return x_range * y_range

    def calculate_pressure_height(self, positions: np.ndarray) -> float:
        """
        Calcula la altura de presión del equipo.

        Returns:
            Posición X percentil 85 en metros (0-105)
        """
        if len(positions) == 0:
            return 0.0

        return float(np.percentile(positions[:, 0], 85))

    def calculate_offensive_width(self, positions: np.ndarray) -> float:
        """
        Calcula la amplitud ofensiva (dispersión horizontal).

        Amplitud = Rango en el eje Y (ancho del campo ocupado)

        Mayor valor = Mayor amplitud (equipo estirado horizontalmente)
        Menor valor = Menor amplitud (equipo concentrado al centro)

        Returns:
            Amplitud en metros (0-68)
        """
        if len(positions) == 0:
            return 0.0

        y_range = positions[:, 1].max() - positions[:, 1].min()
        return float(y_range)

    def calculate_centroid(self, positions: np.ndarray) -> Tuple[float, float]:
        """
        Calcula el centroide (centro geométrico) del equipo.

        Returns:
            Tupla (x, y) con el centroide en metros
        """
        if len(positions) == 0:
            return (0.0, 0.0)

        centroid_x = float(np.mean(positions[:, 0]))
        centroid_y = float(np.mean(positions[:, 1]))
        return (centroid_x, centroid_y)

    def calculate_stretch_index(self, positions: np.ndarray) -> float:
        """
        Calcula el índice de elongación del equipo.

        Stretch Index = Desviación estándar de posiciones X / Desviación estándar de posiciones Y

        > 1.0 = Equipo más estirado verticalmente (en profundidad)
        < 1.0 = Equipo más estirado horizontalmente (en amplitud)
        ~ 1.0 = Equipo equilibrado

        Returns:
            Ratio de elongación
        """
        if len(positions) < 2:
            return 1.0

        std_x = np.std(positions[:, 0])
        std_y = np.std(positions[:, 1])

        if std_y < 0.1:  # Evitar división por cero
            return 10.0 if std_x > 0.1 else 1.0

        return float(std_x / std_y)

    def calculate_defensive_depth(self, positions: np.ndarray) -> float:
        """
        Calcula la profundidad defensiva (distancia entre jugador más adelantado y más retrasado).

        Returns:
            Profundidad en metros
        """
        if len(positions) == 0:
            return 0.0

        x_range = positions[:, 0].max() - positions[:, 0].min()
        return float(x_range)

    def calculate_defensive_block_compactness(self, positions: np.ndarray) -> float:
        """
        Calcula la compactación del bloque defensivo (jugadores más retrasados).

        Returns:
            Área del 50% de jugadores más retrasados
        """
        if len(positions) < 4:
            return self.calculate_compactness(positions)

        # Tomar mitad de jugadores más retrasados
        sorted_by_x = positions[np.argsort(positions[:, 0])]
        defensive_half = sorted_by_x[:len(positions)//2]

        return self.calculate_compactness(defensive_half)

    def _empty_metrics(self) -> Dict:
        """Retorna métricas vacías cuando no hay suficientes jugadores."""
        return {
            'compactness': 0.0,
            'pressure_height': 0.0,
            'offensive_width': 0.0,
            'centroid': (0.0, 0.0),
            'stretch_index': 1.0,
            'defensive_depth': 0.0,
            'num_players': 0
        }


class TacticalMetricsTracker:
    """
    Rastrea métricas tácticas a lo largo del tiempo y calcula tendencias.
    """

    def __init__(self, history_size: int = 300):
        """
        Args:
            history_size: Número de frames a mantener en historia (default: 300 = ~10 seg a 30fps)
        """
        self.history_size = history_size
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
        self.valid_frames_count = 0

    def update(self, metrics: Optional[Dict], frame_number: int):
        """
        Actualiza el historial con nuevas métricas.

        Args:
            metrics: Dict con métricas del frame actual
            frame_number: Número del frame
        """
        if metrics is None:
            return
        self.metrics_history['compactness'].append(metrics.get('compactness'))
        self.metrics_history['pressure_height'].append(metrics.get('pressure_height'))
        self.metrics_history['offensive_width'].append(metrics.get('offensive_width'))
        centroid = metrics.get('centroid')
        if isinstance(centroid, (tuple, list)) and len(centroid) == 2:
            self.metrics_history['centroid_x'].append(centroid[0])
            self.metrics_history['centroid_y'].append(centroid[1])
        self.metrics_history['stretch_index'].append(metrics.get('stretch_index'))
        self.metrics_history['defensive_depth'].append(metrics.get('defensive_depth'))
        self.metrics_history['block_depth_m'].append(metrics.get('block_depth_m'))
        self.metrics_history['block_width_m'].append(metrics.get('block_width_m'))
        self.metrics_history['block_area_m2'].append(metrics.get('block_area_m2'))
        self.metrics_history['def_line_left_m'].append(metrics.get('def_line_left_m'))
        self.metrics_history['def_line_right_m'].append(metrics.get('def_line_right_m'))
        self.metrics_history['frame_number'].append(frame_number)
        self.valid_frames_count += 1

    def get_statistics(self) -> Dict:
        """
        Calcula estadísticas sobre las métricas históricas.

        Returns:
            Dict con media, std, min, max para cada métrica
        """
        stats = {}

        for metric_name, values in self.metrics_history.items():
            if metric_name == 'frame_number':
                continue

            if len(values) == 0:
                stats[metric_name] = {'mean': None, 'std': None, 'min': None, 'max': None, 'current': None}
                continue

            values_array = np.array(list(values))
            stats[metric_name] = {
                'mean': float(np.mean(values_array)) if len(values_array) > 0 else None,
                'std': float(np.std(values_array)) if len(values_array) > 0 else None,
                'min': float(np.min(values_array)) if len(values_array) > 0 else None,
                'max': float(np.max(values_array)) if len(values_array) > 0 else None,
                'current': float(values_array[-1]) if len(values_array) > 0 else None
            }

        stats['valid_frames'] = self.valid_frames_count
        return stats

    def get_trend(self, metric_name: str, window: int = 30) -> str:
        """
        Determina la tendencia de una métrica (creciente, decreciente, estable).

        Args:
            metric_name: Nombre de la métrica
            window: Ventana para calcular tendencia

        Returns:
            "increasing", "decreasing", "stable"
        """
        if metric_name not in self.metrics_history:
            return "unknown"

        values = list(self.metrics_history[metric_name])

        if len(values) < window:
            return "insufficient_data"

        recent_values = values[-window:]
        first_half = np.mean(recent_values[:window//2])
        second_half = np.mean(recent_values[window//2:])

        change_pct = abs((second_half - first_half) / first_half) * 100 if first_half != 0 else 0

        if change_pct < 5:  # Menos del 5% de cambio
            return "stable"
        elif second_half > first_half:
            return "increasing"
        else:
            return "decreasing"

    def export_to_dict(self) -> Dict:
        """
        Exporta todo el historial a un diccionario (para guardar en JSON/CSV).

        Returns:
            Dict con arrays de métricas por frame
        """
        return {
            metric_name: list(values)
            for metric_name, values in self.metrics_history.items()
        }

    def export_to_arrays(self) -> Dict[str, np.ndarray]:
        """
        Exporta historial como arrays de NumPy (para gráficos).

        Returns:
            Dict con arrays de NumPy por métrica
        """
        return {
            metric_name: np.array(list(values))
            for metric_name, values in self.metrics_history.items()
        }
