"""
Speed & Distance Estimator
Calcula velocidad instantánea (km/h) y distancia acumulada (m) por jugador
usando posiciones proyectadas al campo real vía homografía.

Incluye: detección de sprints y zonas de intensidad de carrera.
"""

import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Optional, Tuple

# Umbrales de intensidad de carrera (km/h) — estándar FIFA
INTENSITY_ZONES = {
    "walking": (0, 7),
    "jogging": (7, 14),
    "running": (14, 21),
    "high_intensity": (21, 25),
    "sprint": (25, 100),
}

SPRINT_THRESHOLD_KMH = 25.0
MAX_REALISTIC_DIST_PER_FRAME = 0.55  # ~60 km/h a 30fps → 0.55 m/frame


class SpeedDistanceEstimator:
    def __init__(self, fps: float = 30.0, window_frames: int = 5):
        self.fps = max(1.0, fps)
        self.window_frames = window_frames

        # Estado por jugador
        self.prev_positions: Dict[int, np.ndarray] = {}
        self.total_distance: Dict[int, float] = {}
        self.speed_history: Dict[int, deque] = {}
        self.max_speed: Dict[int, float] = {}
        self.team_assignment: Dict[int, str] = {}

        # Zonas de intensidad: tracker_id → {zone_name: distancia_m}
        self.zone_distance: Dict[int, Dict[str, float]] = {}

        # Sprints: tracker_id → list of sprint episodes
        self.sprints: Dict[int, List[Dict]] = {}
        self._in_sprint: Dict[int, bool] = {}
        self._sprint_start_frame: Dict[int, int] = {}
        self._sprint_distance: Dict[int, float] = {}
        self._current_frame: int = 0

    def update(
        self,
        tracker_ids: List[int],
        field_positions: np.ndarray,
        team_labels: List[str],
    ) -> Dict[int, Dict]:
        result = {}
        self._current_frame += 1

        if field_positions is None or len(field_positions) == 0:
            return result

        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            pos = field_positions[i]

            if tid not in self.team_assignment and i < len(team_labels):
                self.team_assignment[tid] = team_labels[i]

            if tid in self.prev_positions:
                delta = pos - self.prev_positions[tid]
                dist_m = float(np.linalg.norm(delta))

                # Fix B: umbral realista en vez de 10m
                if dist_m > MAX_REALISTIC_DIST_PER_FRAME:
                    result[tid] = {
                        "speed_kmh": self._get_smoothed_speed(tid),
                        "distance_m": self.total_distance.get(tid, 0.0),
                    }
                    continue

                speed_mps = dist_m * self.fps
                speed_kmh = speed_mps * 3.6

                self.total_distance[tid] = self.total_distance.get(tid, 0.0) + dist_m

                if tid not in self.speed_history:
                    self.speed_history[tid] = deque(maxlen=self.window_frames)
                self.speed_history[tid].append(speed_kmh)

                if tid not in self.max_speed:
                    self.max_speed[tid] = 0.0
                smoothed = self._get_smoothed_speed(tid)
                if smoothed > self.max_speed[tid]:
                    self.max_speed[tid] = smoothed

                # Mejora G: acumular distancia por zona de intensidad
                self._accumulate_intensity_zone(tid, dist_m, smoothed)

                # Mejora G: detección de sprints
                self._update_sprint_tracking(tid, smoothed, dist_m)

                result[tid] = {
                    "speed_kmh": smoothed,
                    "distance_m": self.total_distance[tid],
                }
            else:
                self.total_distance[tid] = 0.0
                result[tid] = {"speed_kmh": 0.0, "distance_m": 0.0}

            self.prev_positions[tid] = pos.copy()

        return result

    def _accumulate_intensity_zone(self, tid: int, dist_m: float, speed_kmh: float):
        if tid not in self.zone_distance:
            self.zone_distance[tid] = {z: 0.0 for z in INTENSITY_ZONES}
        for zone_name, (lo, hi) in INTENSITY_ZONES.items():
            if lo <= speed_kmh < hi:
                self.zone_distance[tid][zone_name] += dist_m
                break

    def _update_sprint_tracking(self, tid: int, speed_kmh: float, dist_m: float):
        is_sprinting = speed_kmh >= SPRINT_THRESHOLD_KMH
        was_sprinting = self._in_sprint.get(tid, False)

        if is_sprinting and not was_sprinting:
            # Inicio de sprint
            self._in_sprint[tid] = True
            self._sprint_start_frame[tid] = self._current_frame
            self._sprint_distance[tid] = dist_m
        elif is_sprinting and was_sprinting:
            # Continúa sprint
            self._sprint_distance[tid] = self._sprint_distance.get(tid, 0.0) + dist_m
        elif not is_sprinting and was_sprinting:
            # Fin de sprint
            self._in_sprint[tid] = False
            duration_frames = self._current_frame - self._sprint_start_frame.get(tid, self._current_frame)
            if duration_frames >= 3:  # Mínimo 3 frames para contar como sprint
                if tid not in self.sprints:
                    self.sprints[tid] = []
                self.sprints[tid].append({
                    "start_frame": self._sprint_start_frame[tid],
                    "duration_frames": duration_frames,
                    "duration_s": round(duration_frames / self.fps, 2),
                    "distance_m": round(self._sprint_distance.get(tid, 0.0), 1),
                })

    def _get_smoothed_speed(self, tid: int) -> float:
        if tid not in self.speed_history or len(self.speed_history[tid]) == 0:
            return 0.0
        return float(np.mean(self.speed_history[tid]))

    def draw_player_speed(
        self, frame: np.ndarray, tracker_id: int, xyxy: np.ndarray, speed_kmh: float,
    ) -> np.ndarray:
        if speed_kmh < 2.0:
            return frame

        x_center = int((xyxy[0] + xyxy[2]) / 2)
        y_bottom = int(xyxy[3]) + 18

        # Color por zona de intensidad
        if speed_kmh >= SPRINT_THRESHOLD_KMH:
            color = (0, 0, 255)  # Rojo sprint
        elif speed_kmh >= 21:
            color = (0, 140, 255)  # Naranja alta intensidad
        elif speed_kmh >= 14:
            color = (0, 255, 255)  # Amarillo running
        else:
            color = (0, 255, 200)  # Verde claro jogging

        text = f"{speed_kmh:.1f} km/h"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(
            frame, (x_center - tw // 2 - 2, y_bottom - th - 2),
            (x_center + tw // 2 + 2, y_bottom + 2), (0, 0, 0), -1,
        )
        cv2.putText(
            frame, text, (x_center - tw // 2, y_bottom),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
        return frame

    def get_summary_stats(self) -> Dict:
        per_player = {}
        team_stats: Dict[str, Dict] = {}

        for tid, dist in self.total_distance.items():
            team = self.team_assignment.get(tid, "unknown")
            zones = self.zone_distance.get(tid, {})
            sprint_list = self.sprints.get(tid, [])
            sprint_dist = sum(s["distance_m"] for s in sprint_list)

            per_player[str(tid)] = {
                "distance_m": round(dist, 1),
                "max_speed_kmh": round(self.max_speed.get(tid, 0.0), 1),
                "team": team,
                "sprint_count": len(sprint_list),
                "sprint_distance_m": round(sprint_dist, 1),
                "intensity_zones_m": {k: round(v, 1) for k, v in zones.items()},
            }

            if team not in team_stats:
                team_stats[team] = {
                    "total_distance_m": 0.0, "player_count": 0,
                    "max_speed_kmh": 0.0, "total_sprints": 0,
                    "total_sprint_distance_m": 0.0,
                }
            team_stats[team]["total_distance_m"] += dist
            team_stats[team]["player_count"] += 1
            team_stats[team]["total_sprints"] += len(sprint_list)
            team_stats[team]["total_sprint_distance_m"] += sprint_dist
            player_max = self.max_speed.get(tid, 0.0)
            if player_max > team_stats[team]["max_speed_kmh"]:
                team_stats[team]["max_speed_kmh"] = player_max

        per_team = {}
        for team, data in team_stats.items():
            count = max(1, data["player_count"])
            per_team[team] = {
                "total_distance_m": round(data["total_distance_m"], 1),
                "avg_distance_m": round(data["total_distance_m"] / count, 1),
                "max_speed_kmh": round(data["max_speed_kmh"], 1),
                "player_count": data["player_count"],
                "total_sprints": data["total_sprints"],
                "total_sprint_distance_m": round(data["total_sprint_distance_m"], 1),
            }

        return {"per_player": per_player, "per_team": per_team}
