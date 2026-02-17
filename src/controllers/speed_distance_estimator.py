"""
Speed & Distance Estimator
Calcula velocidad instantánea (km/h) y distancia acumulada (m) por jugador
usando posiciones proyectadas al campo real vía homografía.

Adaptado de: https://huggingface.co/spaces/ixxan/football-vision-analytics
"""

import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Optional, Tuple


class SpeedDistanceEstimator:
    def __init__(self, fps: float = 30.0, window_frames: int = 5):
        """
        Args:
            fps: FPS del video (para convertir frames → segundos).
            window_frames: Ventana de suavizado para velocidad instantánea.
        """
        self.fps = max(1.0, fps)
        self.window_frames = window_frames

        # Estado por jugador
        self.prev_positions: Dict[int, np.ndarray] = {}
        self.total_distance: Dict[int, float] = {}
        self.speed_history: Dict[int, deque] = {}
        self.max_speed: Dict[int, float] = {}
        self.team_assignment: Dict[int, str] = {}

    def update(
        self,
        tracker_ids: List[int],
        field_positions: np.ndarray,
        team_labels: List[str],
    ) -> Dict[int, Dict]:
        """
        Actualiza velocidad y distancia para un conjunto de jugadores en un frame.

        Args:
            tracker_ids: IDs de tracking.
            field_positions: Array (N, 2) en metros (coordenadas de campo).
            team_labels: Lista de 'team1'|'team2' por jugador.

        Returns:
            Dict {tracker_id: {'speed_kmh': float, 'distance_m': float}}
        """
        result = {}

        if field_positions is None or len(field_positions) == 0:
            return result

        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            pos = field_positions[i]

            # Registrar equipo
            if tid not in self.team_assignment and i < len(team_labels):
                self.team_assignment[tid] = team_labels[i]

            if tid in self.prev_positions:
                delta = pos - self.prev_positions[tid]
                dist_m = float(np.linalg.norm(delta))

                # Filtrar saltos imposibles (artefactos de homografía)
                if dist_m > 10.0:
                    # Descartar este frame para este jugador
                    result[tid] = {
                        "speed_kmh": self._get_smoothed_speed(tid),
                        "distance_m": self.total_distance.get(tid, 0.0),
                    }
                    continue

                # Velocidad instantánea
                speed_mps = dist_m * self.fps  # metros por segundo
                speed_kmh = speed_mps * 3.6

                # Acumular
                self.total_distance[tid] = self.total_distance.get(tid, 0.0) + dist_m

                if tid not in self.speed_history:
                    self.speed_history[tid] = deque(maxlen=self.window_frames)
                self.speed_history[tid].append(speed_kmh)

                if tid not in self.max_speed:
                    self.max_speed[tid] = 0.0
                smoothed = self._get_smoothed_speed(tid)
                if smoothed > self.max_speed[tid]:
                    self.max_speed[tid] = smoothed

                result[tid] = {
                    "speed_kmh": smoothed,
                    "distance_m": self.total_distance[tid],
                }
            else:
                # Primer frame para este jugador
                self.total_distance[tid] = 0.0
                result[tid] = {"speed_kmh": 0.0, "distance_m": 0.0}

            self.prev_positions[tid] = pos.copy()

        return result

    def _get_smoothed_speed(self, tid: int) -> float:
        if tid not in self.speed_history or len(self.speed_history[tid]) == 0:
            return 0.0
        return float(np.mean(self.speed_history[tid]))

    def draw_player_speed(
        self,
        frame: np.ndarray,
        tracker_id: int,
        xyxy: np.ndarray,
        speed_kmh: float,
    ) -> np.ndarray:
        """Dibuja etiqueta de velocidad debajo del bounding box del jugador."""
        if speed_kmh < 2.0:
            return frame

        x_center = int((xyxy[0] + xyxy[2]) / 2)
        y_bottom = int(xyxy[3]) + 18

        text = f"{speed_kmh:.1f} km/h"

        # Fondo oscuro para legibilidad
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            frame,
            (x_center - tw // 2 - 2, y_bottom - th - 2),
            (x_center + tw // 2 + 2, y_bottom + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x_center - tw // 2, y_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return frame

    def get_summary_stats(self) -> Dict:
        """Retorna estadísticas agregadas para el JSON de salida."""
        per_player = {}
        team_stats: Dict[str, Dict] = {}

        for tid, dist in self.total_distance.items():
            team = self.team_assignment.get(tid, "unknown")
            per_player[str(tid)] = {
                "distance_m": round(dist, 1),
                "max_speed_kmh": round(self.max_speed.get(tid, 0.0), 1),
                "team": team,
            }

            if team not in team_stats:
                team_stats[team] = {
                    "total_distance_m": 0.0,
                    "player_count": 0,
                    "max_speed_kmh": 0.0,
                }
            team_stats[team]["total_distance_m"] += dist
            team_stats[team]["player_count"] += 1
            player_max = self.max_speed.get(tid, 0.0)
            if player_max > team_stats[team]["max_speed_kmh"]:
                team_stats[team]["max_speed_kmh"] = player_max

        # Promedios
        per_team = {}
        for team, data in team_stats.items():
            count = max(1, data["player_count"])
            per_team[team] = {
                "total_distance_m": round(data["total_distance_m"], 1),
                "avg_distance_m": round(data["total_distance_m"] / count, 1),
                "max_speed_kmh": round(data["max_speed_kmh"], 1),
                "player_count": data["player_count"],
            }

        return {
            "per_player": per_player,
            "per_team": per_team,
        }
