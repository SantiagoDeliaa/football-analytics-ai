import numpy as np
import cv2
from typing import Optional, Tuple
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
        if source_points is None or target_points is None:
            self.frames_since_valid += 1
            return False
        if len(source_points) < self.min_points or len(target_points) < self.min_points:
            self.frames_since_valid += 1
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
                return False

        sp = source_points.astype(np.float32)
        tp = target_points.astype(np.float32)
        H_new, mask = cv2.findHomography(sp, tp, cv2.RANSAC, 5.0)
        if H_new is None:
            self.frames_since_valid += 1
            self.last_state = "INVALID_RANSAC"
            return False

        if not np.isfinite(H_new).all():
            self.frames_since_valid += 1
            self.last_state = "INVALID_RANSAC"
            return False
        try:
            det2 = float(np.linalg.det(H_new[0:2, 0:2]))
        except Exception:
            det2 = 0.0
        if abs(det2) < 1e-6:
            self.frames_since_valid += 1
            self.last_state = "INVALID_RANSAC"
            return False

        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
        else:
            inlier_mask = np.ones(len(sp), dtype=bool)
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
            return False

        Hn_new = self._normalize(H_new)
        if self.current_H is None:
            self.current_H = Hn_new
            self.frames_since_valid = 0
            self.last_state = "UPDATED_EMA"
            if self.debug:
                print("H updated")
            return True

        Hn_cur = self.current_H
        denom = np.linalg.norm(Hn_cur) + 1e-8
        delta = float(np.linalg.norm(Hn_new - Hn_cur) / denom)
        self.last_delta = float(delta)
        if delta < self.delta_matrix_thresh:
            Hn_smooth = self.alpha * Hn_new + (1.0 - self.alpha) * Hn_cur
            self.current_H = self._normalize(Hn_smooth)
            self.frames_since_valid = 0
            self.last_state = "UPDATED_EMA"
            if self.debug:
                print("H updated")
            return True
        else:
            if self.frames_since_valid >= self.switch_after_frames:
                self.current_H = self._normalize(Hn_new)
                self.frames_since_valid = 0
                self.last_state = "UPDATED_SWITCH"
                if self.debug:
                    print("H updated")
                return True
            else:
                self.frames_since_valid += 1
                self.last_state = "INVALID_DELTA"
                return False

    def get_transformer(self) -> Optional[ViewTransformer]:
        if self.current_H is None:
            return None
        if self.frames_since_valid < self.max_inertia_frames:
            if self.frames_since_valid > 0:
                self.last_state = "REUSED_INERTIA"
                if self.debug:
                    print("H reused (inertia)")
            return ViewTransformer(m=self.current_H)
        return None
