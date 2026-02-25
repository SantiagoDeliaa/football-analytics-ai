import numpy as np
import cv2
from typing import Optional, Tuple
from src.utils.quality_config import (
    REPROJ_OK_MAX,
    REPROJ_WARN_MAX,
    REPROJ_INVALID,
    REPROJ_WARMUP_MAX,
    DELTA_H_WARN,
    DELTA_H_CUT,
    IMPROVE_MARGIN,
    WARMUP_FRAMES,
    REACQUIRE_FRAMES,
)
from src.utils.view_transformer import ViewTransformer

class HomographyManager:
    def __init__(
        self,
        spread_threshold: float = 0.30,
        max_inertia_frames: int = 30,
        alpha: float = 0.15,
        min_points: int = 4,
        max_reproj_error: float = 3.0,
        delta_matrix_thresh: float = 0.35,
        switch_after_frames: int = 10,
        debug: bool = False
    ):
        self.current_H: Optional[np.ndarray] = None
        self.frames_since_valid: int = 0
        self.spread_threshold = spread_threshold
        self.max_inertia_frames = max_inertia_frames
        self.alpha = alpha
        self.min_points = min_points
        self.max_reproj_error = max_reproj_error
        self.delta_matrix_thresh = delta_matrix_thresh
        self.switch_after_frames = switch_after_frames
        self.debug = debug
        self.last_state: str = ""
        self.last_reproj_error: Optional[float] = None
        self.last_delta: Optional[float] = None
        self.last_inlier_ratio: Optional[float] = None
        self.mode: str = "ACQUIRE"
        self.homography_state: str = "WARMUP"
        self.warmup_frames = WARMUP_FRAMES
        self.reacquire_frames = REACQUIRE_FRAMES
        self.warmup_counter = 0
        self.reacquire_counter = 0
        self.anchor_H: Optional[np.ndarray] = None
        self.anchor_score: Optional[float] = None
        self.anchor_reproj: Optional[float] = None
        self.anchor_delta: Optional[float] = None
        self.cut_detected = False

    def _sync_state(self):
        if self.mode == "ACQUIRE":
            self.homography_state = "WARMUP"
        elif self.mode == "TRACK":
            self.homography_state = "STABLE"
        elif self.mode == "INERTIA":
            self.homography_state = "DEGRADED"
        elif self.mode == "REACQUIRE":
            self.homography_state = "REACQUIRE"
        else:
            self.homography_state = "FALLBACK"

    def _normalize(self, H: np.ndarray) -> np.ndarray:
        d = H[2, 2]
        if d == 0:
            return H
        return H / d

    def _mean_reproj_error(
        self,
        H: np.ndarray,
        src: np.ndarray,
        tgt: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        sp = src.reshape(-1, 1, 2).astype(np.float32)
        pred = cv2.perspectiveTransform(sp, H).reshape(-1, 2)
        diff = pred - tgt
        d = np.linalg.norm(diff, axis=1)
        if weights is not None and len(weights) == len(d):
            w = np.array(weights, dtype=np.float32)
            w = np.clip(w, 1e-6, None)
            return float(np.sum(d * w) / np.sum(w))
        return float(np.mean(d))

    def update(
        self,
        source_points: Optional[np.ndarray],
        target_points: Optional[np.ndarray],
        confidences: Optional[np.ndarray] = None,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        self.last_reproj_error = None
        self.last_delta = None
        self.last_inlier_ratio = None
        self.cut_detected = False
        if source_points is None or target_points is None:
            self.frames_since_valid += 1
            if self.current_H is not None and self.mode != "ACQUIRE":
                self.mode = "INERTIA"
            elif self.current_H is None:
                self.mode = "FALLBACK"
            self._sync_state()
            return False
        if len(source_points) < self.min_points or len(target_points) < self.min_points:
            self.frames_since_valid += 1
            if self.current_H is not None and self.mode != "ACQUIRE":
                self.mode = "INERTIA"
            elif self.current_H is None:
                self.mode = "FALLBACK"
            self._sync_state()
            return False

        w = None
        h = None
        if frame_size is not None:
            w, h = frame_size

        if w is not None and h is not None:
            x_range = float(np.max(source_points[:, 0]) - np.min(source_points[:, 0]))
            y_range = float(np.max(source_points[:, 1]) - np.min(source_points[:, 1]))
            spread_ok = (x_range / max(w, 1e-6)) >= self.spread_threshold and (y_range / max(h, 1e-6)) >= self.spread_threshold
            if not spread_ok:
                self.frames_since_valid += 1
                self.last_state = "INVALID_SPREAD"
                if self.current_H is not None and self.mode != "ACQUIRE":
                    self.mode = "INERTIA"
                elif self.current_H is None:
                    self.mode = "FALLBACK"
                self._sync_state()
                return False

        sp = source_points.astype(np.float32)
        tp = target_points.astype(np.float32)
        H_new, mask = cv2.findHomography(sp, tp, cv2.RANSAC, 5.0)
        if H_new is None:
            self.frames_since_valid += 1
            self.last_state = "INVALID_RANSAC"
            if self.current_H is not None and self.mode != "ACQUIRE":
                self.mode = "INERTIA"
            elif self.current_H is None:
                self.mode = "FALLBACK"
            self._sync_state()
            return False

        if not np.isfinite(H_new).all():
            self.frames_since_valid += 1
            self.last_state = "INVALID_RANSAC"
            if self.current_H is not None and self.mode != "ACQUIRE":
                self.mode = "INERTIA"
            elif self.current_H is None:
                self.mode = "FALLBACK"
            self._sync_state()
            return False
        try:
            det2 = float(np.linalg.det(H_new[0:2, 0:2]))
        except Exception:
            det2 = 0.0
        if abs(det2) < 1e-6:
            self.frames_since_valid += 1
            self.last_state = "INVALID_RANSAC"
            if self.current_H is not None and self.mode != "ACQUIRE":
                self.mode = "INERTIA"
            elif self.current_H is None:
                self.mode = "FALLBACK"
            self._sync_state()
            return False

        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
        else:
            inlier_mask = np.ones(len(sp), dtype=bool)
        if len(inlier_mask) > 0:
            self.last_inlier_ratio = float(np.sum(inlier_mask) / len(inlier_mask))
        sp_in = sp[inlier_mask]
        tp_in = tp[inlier_mask]

        weights = None
        if confidences is not None:
            conf = np.array(confidences).reshape(-1)
            conf_in = conf[inlier_mask] if len(conf) == len(inlier_mask) else conf
            weights = conf_in

        err = self._mean_reproj_error(H_new, sp_in, tp_in, weights)
        self.last_reproj_error = float(err)
        if err > self.max_reproj_error:
            self.frames_since_valid += 1
            self.last_state = "INVALID_REPROJ"
            if self.current_H is not None and self.mode != "ACQUIRE":
                self.mode = "INERTIA"
            elif self.current_H is None:
                self.mode = "FALLBACK"
            self._sync_state()
            return False

        Hn_new = self._normalize(H_new)
        if self.current_H is None and self.mode != "ACQUIRE":
            self.current_H = Hn_new
            self.frames_since_valid = 0
            self.last_state = "UPDATED_EMA"
            if self.debug:
                print("H updated")
            self._sync_state()
            return True

        Hn_cur = self.current_H
        delta = None
        if Hn_cur is not None:
            denom = np.linalg.norm(Hn_cur) + 1e-8
            delta = float(np.linalg.norm(Hn_new - Hn_cur) / denom)
            self.last_delta = float(delta)

        if self.mode == "ACQUIRE":
            warmup_score = float(err + 0.5 * (delta if delta is not None else 0.0))
            if err <= REPROJ_WARMUP_MAX:
                if self.anchor_score is None or warmup_score < self.anchor_score:
                    self.anchor_score = warmup_score
                    self.anchor_H = Hn_new.copy()
                    self.anchor_reproj = float(err)
                    self.anchor_delta = float(delta) if delta is not None else None
            self.warmup_counter += 1
            if self.warmup_counter >= self.warmup_frames:
                if self.anchor_H is not None:
                    self.current_H = self.anchor_H
                    self.frames_since_valid = 0
                    self.mode = "TRACK"
                    self.last_state = "WARMUP_ANCHOR"
                    self._sync_state()
                    return True
                self.mode = "FALLBACK"
                self.last_state = "WARMUP_FAIL"
                self._sync_state()
                return False
            self.last_state = "WARMUP"
            self._sync_state()
            return False

        if Hn_cur is None:
            self.current_H = Hn_new
            self.frames_since_valid = 0
            self.last_state = "UPDATED_EMA"
            self.mode = "TRACK"
            self._sync_state()
            return True

        current_err = self._mean_reproj_error(Hn_cur, sp_in, tp_in, weights)
        accept = False
        if current_err is None:
            accept = True
        else:
            if err <= current_err - IMPROVE_MARGIN:
                accept = True
            elif err <= REPROJ_OK_MAX and (delta is not None and delta <= DELTA_H_WARN):
                accept = True

        if accept:
            if delta is not None and delta < self.delta_matrix_thresh:
                Hn_smooth = self.alpha * Hn_new + (1.0 - self.alpha) * Hn_cur
                self.current_H = self._normalize(Hn_smooth)
                self.last_state = "UPDATED_EMA"
            else:
                self.current_H = self._normalize(Hn_new)
                self.last_state = "UPDATED_SWITCH"
            self.frames_since_valid = 0
            self.mode = "TRACK"
            self._sync_state()
            return True

        self.frames_since_valid += 1
        self.last_state = "GUARDRAIL_HOLD"
        if self.current_H is not None and self.mode != "ACQUIRE":
            self.mode = "REACQUIRE" if self.mode == "REACQUIRE" else "INERTIA"
        elif self.current_H is None:
            self.mode = "FALLBACK"
        self._sync_state()
        return False

    def get_transformer(self) -> Optional[ViewTransformer]:
        if self.current_H is None:
            return None
        if self.frames_since_valid < self.max_inertia_frames:
            if self.frames_since_valid > 0:
                self.last_state = "REUSED_INERTIA"
                if self.debug:
                    print("H reused (inertia)")
            self._sync_state()
            return ViewTransformer(m=self.current_H)
        self.mode = "FALLBACK"
        self._sync_state()
        return None

    def start_reacquire(self):
        self.mode = "REACQUIRE"
        self.reacquire_counter = 0
        self._sync_state()

    def tick_reacquire(self):
        if self.mode == "REACQUIRE":
            self.reacquire_counter += 1
            if self.reacquire_counter >= self.reacquire_frames:
                self.mode = "INERTIA" if self.current_H is not None else "FALLBACK"
                self._sync_state()
