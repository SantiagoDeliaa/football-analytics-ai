import numpy as np
import cv2


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray = None):
        """
        Initialize ViewTransformer either with:
        1) source + target points (computes homography with RANSAC), or
        2) a precomputed 3x3 homography matrix in `source`.

        Args:
            source: Array of shape (N, 2) with source points OR homography matrix (3, 3).
            target: Optional array of shape (N, 2) with target points.
        """
        source = np.asarray(source, dtype=np.float32)

        if target is None:
            self.m = source if source.shape == (3, 3) else None
        else:
            target = np.asarray(target, dtype=np.float32)
            # Usar RANSAC para robustez si hay suficientes puntos
            if len(source) >= 4:
                self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            else:
                self.m = None

    def transform_points(self, points: np.ndarray, flip_x: bool = False) -> np.ndarray:
        """
        Transform points from source perspective to target perspective.

        Args:
            points: Array of shape (N, 2) containing points to transform.
            flip_x: If True, inverts X coordinates (105 - x) to match broadcast view.

        Returns:
            Transformed points of shape (N, 2).
        """
        if self.m is None or points is None or points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        transformed_points = transformed_points.reshape(-1, 2)

        # Invertir eje X para que coincida con la vista de broadcast
        # En broadcast: equipo de la izquierda ataca hacia la derecha
        # Sin inversión: el radar muestra el campo al revés
        if flip_x:
            transformed_points[:, 0] = 105.0 - transformed_points[:, 0]

        return transformed_points
