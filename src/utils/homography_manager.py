from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.utils.view_transformer import ViewTransformer


@dataclass
class HomographyManager:
    spread_threshold: float = 0.30
    max_inertia_frames: int = 30
    alpha: float = 0.15
    min_points: int = 4
    ransac_reproj_threshold: float = 5.0
    reprojection_error_threshold: float = 8.0
    max_matrix_delta: float = 0.35
    debug: bool = False

    def __post_init__(self) -> None:
        self.current_H: Optional[np.ndarray] = None
        self.frames_since_valid: int = 0

    @staticmethod
    def _normalize_h(H: np.ndarray) -> Optional[np.ndarray]:
        H = np.asarray(H, dtype=np.float32)
        if H.shape != (3, 3):
            return None
        if abs(H[2, 2]) < 1e-8:
            return None
        return H / H[2, 2]

    @staticmethod
    def _matrix_delta(H1: np.ndarray, H2: np.ndarray) -> float:
        return float(np.linalg.norm(H1 - H2, ord="fro") / np.linalg.norm(H1, ord="fro"))

    def _log(self, frame_count: Optional[int], msg: str) -> None:
        if not self.debug:
            return
        if frame_count is None:
            print(msg)
        else:
            print(f"Frame {frame_count}: {msg}")

    def update(
        self,
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        pitch_config,
        frame_width: int,
        conf_threshold: float,
        frame_count: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Update internal homography using temporal inertia and robust validation."""
        valid_update = False

        if keypoints_xy is not None and keypoints_conf is not None and len(keypoints_xy) > 0:
            valid_kp_mask = keypoints_conf > conf_threshold
            valid_keypoints = keypoints_xy[valid_kp_mask]
            valid_indices = np.where(valid_kp_mask)[0]

            mapped_indices_mask = np.isin(valid_indices, list(pitch_config.keypoints_map.keys()))
            valid_indices = valid_indices[mapped_indices_mask]
            valid_keypoints = valid_keypoints[mapped_indices_mask]

            if len(valid_keypoints) >= self.min_points:
                x_range = float(valid_keypoints[:, 0].max() - valid_keypoints[:, 0].min())
                y_range = float(valid_keypoints[:, 1].max() - valid_keypoints[:, 1].min())
                min_spread = frame_width * self.spread_threshold
                well_distributed = x_range > min_spread or y_range > min_spread

                if well_distributed:
                    target_points = pitch_config.get_keypoints_from_ids(valid_indices).astype(np.float32)
                    source_points = valid_keypoints.astype(np.float32)
                    H_new, inlier_mask = cv2.findHomography(
                        source_points,
                        target_points,
                        cv2.RANSAC,
                        self.ransac_reproj_threshold,
                    )

                    H_new_norm = self._normalize_h(H_new) if H_new is not None else None
                    if H_new_norm is not None:
                        reproj_pts = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), H_new_norm).reshape(-1, 2)
                        errors = np.linalg.norm(reproj_pts - target_points, axis=1)
                        if inlier_mask is not None:
                            inlier_mask = inlier_mask.ravel().astype(bool)
                            if np.any(inlier_mask):
                                errors = errors[inlier_mask]
                        mean_error = float(np.mean(errors)) if errors.size > 0 else float("inf")

                        if mean_error <= self.reprojection_error_threshold:
                            if self.current_H is None:
                                self.current_H = H_new_norm
                                self._log(frame_count, f"H updated (init) with {len(valid_keypoints)} keypoints")
                                valid_update = True
                            else:
                                current_norm = self._normalize_h(self.current_H)
                                delta = self._matrix_delta(current_norm, H_new_norm)
                                if delta < self.max_matrix_delta:
                                    smoothed = self.alpha * H_new_norm + (1.0 - self.alpha) * current_norm
                                    self.current_H = self._normalize_h(smoothed)
                                    self._log(frame_count, f"H updated (EMA), delta={delta:.3f}, err={mean_error:.2f}")
                                    valid_update = True
                                else:
                                    self._log(frame_count, f"H rejected (delta={delta:.3f} > {self.max_matrix_delta})")
                        else:
                            self._log(
                                frame_count,
                                f"H rejected (reprojection error {mean_error:.2f} > {self.reprojection_error_threshold})",
                            )
                else:
                    self._log(frame_count, f"H rejected (spread low: {x_range:.1f}x{y_range:.1f})")
            else:
                self._log(frame_count, f"H rejected (insufficient keypoints: {len(valid_keypoints)})")

        if valid_update:
            self.frames_since_valid = 0
            return self.current_H

        self.frames_since_valid += 1

        if self.current_H is not None and self.frames_since_valid < self.max_inertia_frames:
            self._log(frame_count, f"H reused (inertia {self.frames_since_valid}/{self.max_inertia_frames})")
            return self.current_H

        if self.frames_since_valid >= self.max_inertia_frames:
            self.current_H = None
            self._log(frame_count, "fallback full-screen (inertia exhausted)")

        return None

    def get_transformer(self, flip_x: bool = False) -> Optional[ViewTransformer]:
        """
        Return the current transformer if an active homography exists.

        Note: `flip_x` is kept for API compatibility at call sites.
        """
        _ = flip_x
        if self.current_H is None:
            return None
        return ViewTransformer(self.current_H)
