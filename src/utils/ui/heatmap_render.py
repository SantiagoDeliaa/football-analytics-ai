import cv2
import numpy as np


def draw_pitch_base(width_px: int, height_px: int) -> np.ndarray:
    pitch = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    pitch[:, :] = (20, 70, 20)
    line_color = (255, 255, 255)
    thickness = max(1, int(min(width_px, height_px) * 0.004))
    cv2.rectangle(pitch, (0, 0), (width_px - 1, height_px - 1), line_color, thickness)
    mid_x = width_px // 2
    cv2.line(pitch, (mid_x, 0), (mid_x, height_px - 1), line_color, thickness)
    circle_radius = int(height_px * 0.12)
    cv2.circle(pitch, (mid_x, height_px // 2), circle_radius, line_color, thickness)
    box_w = int(width_px * 0.17)
    box_h = int(height_px * 0.36)
    box_y1 = (height_px - box_h) // 2
    box_y2 = box_y1 + box_h
    cv2.rectangle(pitch, (0, box_y1), (box_w, box_y2), line_color, thickness)
    cv2.rectangle(pitch, (width_px - box_w, box_y1), (width_px - 1, box_y2), line_color, thickness)
    return pitch


def render_heatmap_overlay(heatmap_small: np.ndarray, out_w: int, out_h: int, flip_vertical: bool, use_log: bool) -> np.ndarray:
    heatmap = heatmap_small.astype(np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap[heatmap < 0] = 0.0
    if flip_vertical:
        heatmap = np.flipud(heatmap)
    if use_log:
        heatmap = np.log1p(heatmap)
    positive = heatmap[heatmap > 0]
    if positive.size > 0:
        p2 = float(np.percentile(positive, 2))
        p98 = float(np.percentile(positive, 98))
        if p98 <= p2:
            p98 = p2 + 1e-6
        heatmap = np.clip(heatmap, p2, p98)
        heatmap = (heatmap - p2) / (p98 - p2)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    else:
        heatmap_uint8 = np.zeros_like(heatmap, dtype=np.uint8)
    heatmap_up = cv2.resize(heatmap_uint8, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_HOT
    heatmap_color = cv2.applyColorMap(heatmap_up, cmap)
    alpha = (heatmap_up.astype(np.float32) / 255.0) * 0.85
    mask = heatmap_up > 0
    alpha[mask] = np.maximum(alpha[mask], 0.12)
    alpha_3 = np.dstack([alpha, alpha, alpha])
    pitch = draw_pitch_base(out_w, out_h).astype(np.float32)
    overlay = heatmap_color.astype(np.float32)
    blended = pitch * (1.0 - alpha_3) + overlay * alpha_3
    return blended.astype(np.uint8)


def draw_heatmap_legend(height_px: int, width_px: int = 90) -> np.ndarray:
    grad = np.linspace(1, 0, height_px, dtype=np.float32)
    grad_img = (grad[:, None] * 255).astype(np.uint8)
    grad_img = np.repeat(grad_img, width_px, axis=1)
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_HOT
    legend = cv2.applyColorMap(grad_img, cmap)
    text_color = (255, 255, 255)
    cv2.putText(legend, "Presencia", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    cv2.putText(legend, "(clip)", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    return legend


def build_centroid_heatmap(telemetry: dict, team_key: str, bins: tuple[int, int] = (26, 17)) -> np.ndarray | None:
    if not telemetry:
        return None
    xs = telemetry.get(f"{team_key}_centroid_x", [])
    ys = telemetry.get(f"{team_key}_centroid_y", [])
    if not xs or not ys:
        return None
    heatmap = np.zeros((bins[0], bins[1]), dtype=np.float32)
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        if not (0 <= x <= 105 and 0 <= y <= 68):
            continue
        xi = int(x * (bins[0] - 1) / 105)
        yi = int(y * (bins[1] - 1) / 68)
        heatmap[xi, yi] += 1.0
    if np.sum(heatmap) <= 0:
        return None
    return heatmap
