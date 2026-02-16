import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, source: np.ndarray = None, target: np.ndarray = None, m: np.ndarray = None):
        if m is not None:
            self.m = m.astype(np.float32)
        else:
            source = source.astype(np.float32) if source is not None else None
            target = target.astype(np.float32) if target is not None else None
            if source is not None and target is not None and len(source) >= 4:
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
