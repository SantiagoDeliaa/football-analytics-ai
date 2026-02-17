"""
Ball Possession Tracker
Asigna posesión de pelota al jugador más cercano y acumula estadísticas
de posesión por equipo frame a frame.

Adaptado de: https://huggingface.co/spaces/ixxan/football-vision-analytics
"""

import numpy as np
import cv2
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple


class BallPossessionTracker:
    def __init__(
        self,
        max_player_ball_distance_px: float = 70.0,
        possession_inertia: int = 5,
    ):
        """
        Args:
            max_player_ball_distance_px: Distancia máxima en píxeles para
                asignar posesión (pie del jugador → centro de la pelota).
            possession_inertia: Frames que se mantiene la última posesión
                cuando no hay asignación clara (evita parpadeo).
        """
        self.max_distance = max_player_ball_distance_px
        self.possession_inertia = possession_inertia

        # Contadores globales
        self.team1_frames: int = 0
        self.team2_frames: int = 0
        self.contested_frames: int = 0
        self.total_frames: int = 0

        # Estado actual
        self.current_team: Optional[str] = None  # 'team1' | 'team2' | None
        self.current_player_id: Optional[int] = None
        self._no_assign_streak: int = 0

        # Timeline para gráficos
        self.timeline: List[Optional[str]] = []

        # Posesión por jugador  tracker_id → frame count
        self.per_player: Dict[int, int] = {}
        self.player_teams: Dict[int, str] = {}

    def assign_possession(
        self,
        tracked_persons,
        team1_mask: List[bool],
        team2_mask: List[bool],
        tracked_ball,
    ) -> Optional[Tuple[str, int]]:
        """
        Determina qué jugador tiene la pelota en este frame.

        Args:
            tracked_persons: sv.Detections con tracker_id
            team1_mask: máscara booleana de team1
            team2_mask: máscara booleana de team2
            tracked_ball: sv.Detections de la pelota (0 o 1 detección)

        Returns:
            (team_label, tracker_id) del poseedor, o None.
        """
        self.total_frames += 1

        # Sin pelota detectada
        if tracked_ball is None or len(tracked_ball) == 0:
            return self._handle_no_assignment()

        # Centro de la pelota
        ball_xyxy = tracked_ball.xyxy[0]
        ball_cx = (ball_xyxy[0] + ball_xyxy[2]) / 2
        ball_cy = (ball_xyxy[1] + ball_xyxy[3]) / 2

        if len(tracked_persons) == 0:
            return self._handle_no_assignment()

        # Buscar jugador más cercano (solo team1 y team2)
        min_dist = float("inf")
        best_idx = -1

        for i in range(len(tracked_persons)):
            if not (team1_mask[i] or team2_mask[i]):
                continue
            bbox = tracked_persons.xyxy[i]
            # Pie del jugador (centro-inferior del bbox)
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]
            dist = np.sqrt((foot_x - ball_cx) ** 2 + (foot_y - ball_cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        if best_idx == -1 or min_dist > self.max_distance:
            return self._handle_no_assignment()

        # Asignar posesión
        if team1_mask[best_idx]:
            team = "team1"
        else:
            team = "team2"

        tracker_id = int(tracked_persons.tracker_id[best_idx]) if tracked_persons.tracker_id is not None else -1

        self._no_assign_streak = 0
        self.current_team = team
        self.current_player_id = tracker_id

        if team == "team1":
            self.team1_frames += 1
        else:
            self.team2_frames += 1

        self.per_player[tracker_id] = self.per_player.get(tracker_id, 0) + 1
        self.player_teams[tracker_id] = team
        self.timeline.append(team)

        return (team, tracker_id)

    def _handle_no_assignment(self) -> Optional[Tuple[str, int]]:
        """Mantiene inercia de la última posesión por unos frames."""
        self._no_assign_streak += 1

        if self._no_assign_streak <= self.possession_inertia and self.current_team is not None:
            # Mantener la última posesión (inercia)
            if self.current_team == "team1":
                self.team1_frames += 1
            else:
                self.team2_frames += 1
            self.timeline.append(self.current_team)
            return (self.current_team, self.current_player_id) if self.current_player_id else None

        # Sin posesión clara
        self.contested_frames += 1
        self.timeline.append(None)
        return None

    def get_possession_percentages(self) -> Dict[str, float]:
        assigned = self.team1_frames + self.team2_frames
        if assigned == 0:
            return {"team1": 0.0, "team2": 0.0}
        return {
            "team1": (self.team1_frames / assigned) * 100,
            "team2": (self.team2_frames / assigned) * 100,
        }

    def draw_possession_bar(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja barra de posesión semi-transparente en la parte inferior del frame."""
        h, w = frame.shape[:2]
        pct = self.get_possession_percentages()
        t1 = pct["team1"]
        t2 = pct["team2"]

        # Dimensiones de la barra
        bar_w = min(400, int(w * 0.35))
        bar_h = 50
        margin = 15
        x0 = (w - bar_w) // 2
        y0 = h - bar_h - margin

        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 30), (x0 + bar_w + 10, y0 + bar_h + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Título
        cv2.putText(
            frame, "POSESION",
            (x0 + bar_w // 2 - 45, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Barra bicolor
        split_x = x0 + int(bar_w * t1 / 100) if (t1 + t2) > 0 else x0 + bar_w // 2
        # Team 1 (verde)
        cv2.rectangle(frame, (x0, y0), (split_x, y0 + bar_h), (0, 200, 0), -1)
        # Team 2 (azul)
        cv2.rectangle(frame, (split_x, y0), (x0 + bar_w, y0 + bar_h), (200, 120, 0), -1)

        # Borde
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)

        # Textos de porcentaje
        cv2.putText(
            frame, f"T1 {t1:.0f}%",
            (x0 + 5, y0 + bar_h // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"{t2:.0f}% T2",
            (x0 + bar_w - 95, y0 + bar_h // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Indicador de poseedor actual
        if self.current_team and self.current_player_id is not None:
            indicator_color = (0, 255, 0) if self.current_team == "team1" else (255, 180, 0)
            cv2.circle(frame, (x0 + bar_w + 25, y0 + bar_h // 2), 8, indicator_color, -1)

        return frame

    def get_stats_dict(self) -> Dict:
        pct = self.get_possession_percentages()
        # Top 10 poseedores
        sorted_players = sorted(self.per_player.items(), key=lambda x: x[1], reverse=True)[:10]
        top_possessors = [
            {
                "tracker_id": int(tid),
                "frames": frames,
                "team": self.player_teams.get(tid, "unknown"),
            }
            for tid, frames in sorted_players
        ]
        return {
            "team1_possession_pct": round(pct["team1"], 2),
            "team2_possession_pct": round(pct["team2"], 2),
            "total_possession_frames": self.team1_frames + self.team2_frames,
            "contested_frames": self.contested_frames,
            "total_frames_analyzed": self.total_frames,
            "top_possessors": top_possessors,
        }
