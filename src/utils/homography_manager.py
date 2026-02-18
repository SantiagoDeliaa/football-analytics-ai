import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
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
        # Public accessors (from main branch)
        self.last_reproj_error: Optional[float] = None
        self.last_delta: Optional[float] = None

        # M1: Telemetry
        self.telemetry: List[Dict] = []
        self.frame_index: int = 0
        # Store last computed values for telemetry recording
        self._last_reproj_error: Optional[float] = None
        self._last_delta_H: Optional[float] = None
        self._last_spread_ok: bool = True
        self._last_det_H: Optional[float] = None
        self._last_cond_H: Optional[float] = None
        self._last_inlier_ratio: Optional[float] = None
        self._last_n_keypoints: int = 0

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

    def _validate_H(self, H: np.ndarray, frame_size: Optional[Tuple[int, int]] = None) -> Tuple[bool, Dict]:
        """M4: Validate homography matrix with cascading sanity checks."""
        result = {"det": None, "cond": None, "corners_ok": True, "inbounds_pct": 1.0}

        # 1. Determinant check
        try:
            det = float(np.linalg.det(H[:2, :2]))
        except Exception:
            det = 0.0
        result["det"] = det
        if abs(det) < 0.01 or abs(det) > 100:
            return False, result

        # 2. Condition number (SVD)
        try:
            svs = np.linalg.svd(H, compute_uv=False)
            cond = float(svs[0] / max(svs[-1], 1e-12))
        except Exception:
            cond = float("inf")
        result["cond"] = cond
        if cond > 100000:
            return False, result

        # 3 & 4. Corner and in-bounds mapping (only if frame_size known)
        if frame_size is not None:
            w, h = frame_size
            # 8 distributed points across the frame
            test_pts = np.array([
                [0, 0], [w, 0], [w, h], [0, h],
                [w * 0.25, h * 0.25], [w * 0.75, h * 0.25],
                [w * 0.75, h * 0.75], [w * 0.25, h * 0.75]
            ], dtype=np.float32)

            try:
                transformed = cv2.perspectiveTransform(
                    test_pts.reshape(-1, 1, 2), H
                ).reshape(-1, 2)
            except Exception:
                result["corners_ok"] = False
                result["inbounds_pct"] = 0.0
                return False, result

            # Check corners (first 4 points) within field + 10m margin
            margin = 10.0
            corners = transformed[:4]
            corners_ok = np.all(
                (corners[:, 0] >= -margin) & (corners[:, 0] <= 105 + margin) &
                (corners[:, 1] >= -margin) & (corners[:, 1] <= 68 + margin)
            )
            result["corners_ok"] = bool(corners_ok)

            # In-bounds ratio for all 8 points (within field, no margin)
            in_bounds = (
                (transformed[:, 0] >= 0) & (transformed[:, 0] <= 105) &
                (transformed[:, 1] >= 0) & (transformed[:, 1] <= 68)
            )
            inbounds_pct = float(np.mean(in_bounds))
            result["inbounds_pct"] = inbounds_pct
            if inbounds_pct < 0.5:
                return False, result

        return True, result

    def record_telemetry(self):
        """M1: Record telemetry for the current frame."""
        record = {
            "frame": self.frame_index,
            "mode": self.last_state,
            "reproj_error": self._last_reproj_error,
            "delta_H": self._last_delta_H,
            "spread_ok": self._last_spread_ok,
            "det_H": self._last_det_H,
            "cond_H": self._last_cond_H,
            "inlier_ratio": self._last_inlier_ratio,
            "n_keypoints": self._last_n_keypoints,
        }
        self.telemetry.append(record)
        self.frame_index += 1

    def get_telemetry_summary(self) -> Dict:
        """M1: Return aggregated telemetry summary."""
        total = len(self.telemetry)
        if total == 0:
            return {
                "homography_valid_ratio": 0.0,
                "inertia_ratio": 0.0,
                "fallback_ratio": 0.0,
                "invalid_ratio": 0.0,
                "avg_reproj_error": 0.0,
                "avg_det_H": 0.0,
                "mode_histogram": {},
            }

        modes = [t["mode"] for t in self.telemetry]
        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        updated = sum(1 for m in modes if m.startswith("UPDATED"))
        inertia = sum(1 for m in modes if m == "REUSED_INERTIA")
        fullscreen = sum(1 for m in modes if m == "FULLSCREEN")
        invalid = sum(1 for m in modes if m.startswith("INVALID"))

        reproj_errors = [t["reproj_error"] for t in self.telemetry if t["reproj_error"] is not None]
        det_values = [t["det_H"] for t in self.telemetry if t["det_H"] is not None]

        return {
            "homography_valid_ratio": updated / total,
            "inertia_ratio": inertia / total,
            "fallback_ratio": fullscreen / total,
            "invalid_ratio": invalid / total,
            "avg_reproj_error": float(np.mean(reproj_errors)) if reproj_errors else 0.0,
            "avg_det_H": float(np.mean(det_values)) if det_values else 0.0,
            "mode_histogram": mode_counts,
        }

    def update(
        self,
        source_points: Optional[np.ndarray],
        target_points: Optional[np.ndarray],
        confidences: Optional[np.ndarray] = None,
        frame_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        # Reset public accessors
        self.last_reproj_error = None
        self.last_delta = None
        # Reset per-frame telemetry values
        self._last_reproj_error = None
        self._last_delta_H = None
        self._last_spread_ok = True
        self._last_det_H = None
        self._last_cond_H = None
        self._last_inlier_ratio = None
        self._last_n_keypoints = 0

        if source_points is None or target_points is None:
            self.frames_since_valid += 1
            return False
        if len(source_points) < self.min_points or len(target_points) < self.min_points:
            self._last_n_keypoints = len(source_points) if source_points is not None else 0
            self.frames_since_valid += 1
            return False

        self._last_n_keypoints = len(source_points)

        w = None
        h = None
        if frame_size is not None:
            w, h = frame_size

        if w is not None and h is not None:
            x_range = float(np.max(source_points[:, 0]) - np.min(source_points[:, 0]))
            y_range = float(np.max(source_points[:, 1]) - np.min(source_points[:, 1]))
            spread_ok = (x_range / max(w, 1e-6)) >= self.spread_threshold and (y_range / max(h, 1e-6)) >= self.spread_threshold
            self._last_spread_ok = spread_ok
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

        # M4: Validate H with cascading checks
        is_valid, val_info = self._validate_H(H_new, frame_size)
        self._last_det_H = val_info["det"]
        self._last_cond_H = val_info["cond"]
        if not is_valid:
            self.frames_since_valid += 1
            self.last_state = "INVALID_SANITY"
            return False

        if mask is not None:
            inlier_mask = mask.ravel().astype(bool)
        else:
            inlier_mask = np.ones(len(sp), dtype=bool)

        self._last_inlier_ratio = float(np.sum(inlier_mask)) / max(len(sp), 1)

        sp_in = sp[inlier_mask]
        tp_in = tp[inlier_mask]

        weights = None
        if confidences is not None:
            conf = np.array(confidences).reshape(-1)
            conf_in = conf[inlier_mask] if len(conf) == len(inlier_mask) else conf
            weights = conf_in

        err = self._mean_reproj_error(H_new, sp_in, tp_in, weights)
        self.last_reproj_error = float(err)
        self._last_reproj_error = err
        if err > self.max_reproj_error:
            self.frames_since_valid += 1
            self.last_state = "INVALID_REPROJ"
            return False

        Hn_new = self._normalize(H_new)
        if self.current_H is None:
            self.current_H = Hn_new
            self.frames_since_valid = 0
            self.last_state = "UPDATED_EMA"
            self._last_delta_H = None
            if self.debug:
                print("H updated")
            return True

        Hn_cur = self.current_H
        denom = np.linalg.norm(Hn_cur) + 1e-8
        delta = float(np.linalg.norm(Hn_new - Hn_cur) / denom)
        self.last_delta = float(delta)
        self._last_delta_H = delta
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
