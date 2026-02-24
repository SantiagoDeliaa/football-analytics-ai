import cv2
import numpy as np
import supervision as sv
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
from collections import deque, Counter
import json
from pathlib import Path
from src.utils.view_transformer import ViewTransformer
from src.utils.homography_manager import HomographyManager
from src.utils.radar import SoccerPitchConfiguration, draw_radar_view, draw_radar_with_metrics
from src.controllers.formation_detector import FormationDetector
from src.controllers.tactical_metrics import TacticalMetricsCalculator, TacticalMetricsTracker
from src.utils.quality_config import (
    REPROJ_OK_MAX,
    REPROJ_WARN_MAX,
    REPROJ_INVALID,
    DELTA_H_WARN,
    DELTA_H_CUT,
    MIN_TRACKS_ACTIVE,
    MIN_DETECTIONS,
    SHORT_TRACK_AGE,
    SPEED_MAX_MPS,
    JUMP_MAX_M,
    CHURN_WARN,
)
from ultralytics import YOLO


def convert_to_native_types(obj):
    """
    Convierte tipos de NumPy a tipos nativos de Python para serialización JSON.

    Args:
        obj: Objeto a convertir (puede ser dict, list, ndarray, float32, etc.)

    Returns:
        Objeto con tipos nativos de Python
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    else:
        return obj


def extract_color_features(frame: np.ndarray, bbox: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extrae características de color de una persona, separando camiseta y pantalón.

    Returns:
        Dict con:
        - 'shirt': [H_mean, S_mean, V_mean] para camiseta
        - 'pants': [H_mean, S_mean, V_mean] para pantalón
        - 'combined': [H_shirt, S_shirt, V_shirt, H_pants, S_pants, V_pants]
        - 'color_variance': diferencia entre color de camiseta y pantalón
    """
    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    width = x2 - x1

    # ROI para camiseta (torso superior)
    shirt_center_x = x1 + width // 2
    shirt_center_y = y1 + int(height * 0.30)
    shirt_roi_width = int(width * 0.5)
    shirt_roi_height = int(height * 0.25)

    shirt_x1 = max(0, shirt_center_x - shirt_roi_width // 2)
    shirt_y1 = max(0, shirt_center_y - shirt_roi_height // 2)
    shirt_x2 = min(frame.shape[1], shirt_center_x + shirt_roi_width // 2)
    shirt_y2 = min(frame.shape[0], shirt_center_y + shirt_roi_height // 2)

    # ROI para pantalón (piernas)
    pants_center_x = x1 + width // 2
    pants_center_y = y1 + int(height * 0.70)
    pants_roi_width = int(width * 0.5)
    pants_roi_height = int(height * 0.20)

    pants_x1 = max(0, pants_center_x - pants_roi_width // 2)
    pants_y1 = max(0, pants_center_y - pants_roi_height // 2)
    pants_x2 = min(frame.shape[1], pants_center_x + pants_roi_width // 2)
    pants_y2 = min(frame.shape[0], pants_center_y + pants_roi_height // 2)

    # Extraer colores de camiseta
    shirt_roi = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    if shirt_roi.size == 0:
        shirt_color = np.array([0, 0, 0])
    else:
        shirt_hsv = cv2.cvtColor(shirt_roi, cv2.COLOR_BGR2HSV)
        shirt_color = np.array([
            np.mean(shirt_hsv[:, :, 0]),
            np.mean(shirt_hsv[:, :, 1]),
            np.mean(shirt_hsv[:, :, 2])
        ])

    # Extraer colores de pantalón
    pants_roi = frame[pants_y1:pants_y2, pants_x1:pants_x2]
    if pants_roi.size == 0:
        pants_color = np.array([0, 0, 0])
    else:
        pants_hsv = cv2.cvtColor(pants_roi, cv2.COLOR_BGR2HSV)
        pants_color = np.array([
            np.mean(pants_hsv[:, :, 0]),
            np.mean(pants_hsv[:, :, 1]),
            np.mean(pants_hsv[:, :, 2])
        ])

    # Calcular varianza de color (diferencia entre camiseta y pantalón)
    color_variance = np.linalg.norm(shirt_color - pants_color)

    return {
        'shirt': shirt_color,
        'pants': pants_color,
        'combined': np.concatenate([shirt_color, pants_color]),
        'color_variance': color_variance
    }


def is_in_playing_field(bbox: np.ndarray, frame_width: int, frame_height: int) -> bool:
    """
    Determina si una persona está dentro del área de juego visible.
    Filtra personas en las bandas (banquillos, línea de banda) y áreas no jugables.

    Args:
        bbox: [x1, y1, x2, y2] - Coordenadas del bounding box
        frame_width: Ancho del frame
        frame_height: Alto del frame

    Returns:
        True si la persona está en el área de juego
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    bbox_height = y2 - y1

    # 1. Filtrar zona superior (marcadores, estadísticas, público alto)
    # Solo considerar personas en la mitad/tercio inferior de la imagen
    if center_y < frame_height * 0.15:  # Top 15% del frame (ajustado de 20%)
        return False

    # 2. Filtrar zona muy inferior (publicidad, borde inferior)
    if center_y > frame_height * 0.95:  # Bottom 5% del frame
        return False

    # 3. Filtrar personas muy pequeñas (probablemente en el fondo o público)
    min_height = frame_height * 0.05  # Al menos 5% de la altura (ajustado de 6%)
    if bbox_height < min_height:
        return False

    # 4. Filtrar bandas laterales (banquillos, línea de banda, entrenadores)
    # Área de juego principal: 8% - 92% horizontal (ajustado de 5-95 para excluir DTs)
    if center_x < frame_width * 0.08 or center_x > frame_width * 0.92:
        # Si está en los extremos laterales Y no está muy abajo (córner)
        # Los entrenadores suelen estar en los costados y no en la línea de fondo
        if center_y < frame_height * 0.85:
            return False

    return True


def is_in_goal_area(bbox: np.ndarray, frame_width: int, frame_height: int) -> bool:
    """
    Determina si una persona está en el área de gol (mejorado).
    Área más estricta: 10% de los extremos con verificación de posición vertical.

    Args:
        bbox: [x1, y1, x2, y2] - Coordenadas del bounding box
        frame_width: Ancho del frame
        frame_height: Alto del frame

    Returns:
        True si la persona está en área de gol
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    bbox_height = y2 - y1

    # Área de gol más estricta: 5% de cada lado del campo
    left_goal_area = frame_width * 0.05
    right_goal_area = frame_width * 0.95

    # Verificaciones mejoradas:
    # 1. Debe estar en la zona del campo (no en marcadores o cielo)
    in_field_area = center_y > frame_height * 0.25 and center_y < frame_height * 0.95

    # 2. Debe tener tamaño razonable (no ser una detección muy pequeña en el fondo)
    min_height = frame_height * 0.08  # Al menos 8% de la altura del frame
    is_reasonable_size = bbox_height > min_height

    # 3. En los extremos del campo
    is_in_goal_zone = center_x < left_goal_area or center_x > right_goal_area

    return is_in_goal_zone and in_field_area and is_reasonable_size


def cluster_teams(frame: np.ndarray, detections: sv.Detections, frame_width: int, frame_height: int) -> Tuple[Dict, Dict, List[int], List[int], List[int]]:
    """
    Agrupa jugadores en 2 equipos y detecta árbitros usando K-means clustering mejorado.

    Estrategia mejorada (v3.1):
    1. Filtrar personas FUERA del área de juego (banquillos, director técnico)
    2. Excluir porteros (por posición en área de gol)
    3. Detectar árbitros por score (color típico + varianza) - EXCLUYE ROJO
    4. K-means con 2 clusters solo sobre jugadores (no árbitros ni porteros)
    5. Usar características de camiseta para clustering

    Returns:
        team1_colors, team2_colors, team1_indices, team2_indices, referee_indices
        donde team_colors es un Dict con 'shirt' y 'pants'
    """
    if len(detections) < 4:
        mid = len(detections) // 2
        team1_indices = list(range(mid))
        team2_indices = list(range(mid, len(detections)))
        default_team1 = {'shirt': np.array([90, 100, 100]), 'pants': np.array([0, 50, 50])}
        default_team2 = {'shirt': np.array([0, 100, 100]), 'pants': np.array([0, 50, 50])}
        return default_team1, default_team2, team1_indices, team2_indices, []

    # Paso 1: Filtrar personas FUERA del área de juego (banquillos, bancas)
    in_field_indices = []
    out_of_field_indices = []
    for idx, bbox in enumerate(detections.xyxy):
        if is_in_playing_field(bbox, frame_width, frame_height):
            in_field_indices.append(idx)
        else:
            out_of_field_indices.append(idx)

    # Paso 2: Identificar porteros por posición (solo de los que están en campo)
    goalkeeper_indices_set = set()
    for idx in in_field_indices:
        bbox = detections.xyxy[idx]
        if is_in_goal_area(bbox, frame_width, frame_height):
            x_center = (bbox[0] + bbox[2]) / 2
            if x_center < frame_width * 0.05 or x_center > frame_width * 0.95:
                goalkeeper_indices_set.add(idx)

    # Paso 3: Extraer características de color de personas en campo (NO-porteros)
    color_data = []
    non_gk_indices = []

    for idx in in_field_indices:
        if idx not in goalkeeper_indices_set:
            bbox = detections.xyxy[idx]
            features = extract_color_features(frame, bbox)
            color_data.append({
                'idx': idx,
                'features': features,
                'shirt': features['shirt'],
                'pants': features['pants'],
                'variance': features['color_variance']
            })
            non_gk_indices.append(idx)

    if len(color_data) < 3:
        mid = len(non_gk_indices) // 2
        team1_indices = non_gk_indices[:mid]
        team2_indices = non_gk_indices[mid:]
        default_team1 = {'shirt': np.array([90, 100, 100]), 'pants': np.array([0, 50, 50])}
        default_team2 = {'shirt': np.array([0, 100, 100]), 'pants': np.array([0, 50, 50])}
        return default_team1, default_team2, team1_indices, team2_indices, []

    # Paso 3: Detectar árbitros por alta varianza de color (algoritmo mejorado v3.2)
    # Los árbitros suelen tener colores muy diferentes entre camiseta y pantalón
    variances = np.array([d['variance'] for d in color_data])

    # Usar percentil más estricto para evitar falsos positivos
    if len(variances) > 5:
        variance_threshold = np.percentile(variances, 80)  # Top 20% de varianza (antes 70%)
    else:
        variance_threshold = np.median(variances) + np.std(variances) * 0.7

    # También considerar si el color es muy diferente del promedio general
    all_shirt_colors = np.array([d['shirt'] for d in color_data])
    mean_shirt_color = np.mean(all_shirt_colors, axis=0)
    std_shirt_color = np.std(all_shirt_colors, axis=0)

    # Análisis HSV específico para árbitros
    # Los árbitros suelen usar negro, amarillo neón, o verde brillante
    referee_candidates = []
    player_candidates = []

    for data in color_data:
        shirt_hsv = data['shirt']
        pants_hsv = data['pants']
        is_high_variance = data['variance'] > max(variance_threshold, 45)  # Aumentado de 35 a 45
        shirt_distance_from_mean = np.linalg.norm(data['shirt'] - mean_shirt_color)

        # Detectar colores típicos de árbitros en HSV (MEJORADO v3.2)
        h, s, v = shirt_hsv
        h_pants, s_pants, v_pants = pants_hsv

        # Colores de camiseta típicos de árbitros
        is_black_shirt = v < 70 and s < 70  # Negro/gris oscuro
        is_yellow_shirt = 20 < h < 35 and s > 100 and v > 130  # Amarillo brillante
        is_bright_green_shirt = 40 < h < 75 and s > 110 and v > 110  # Verde lima

        # EXCLUIR BLANCO - Los equipos usan blanco, árbitros rara vez
        is_white_shirt = v > 180 and s < 50  # Blanco (V alto, S bajo)

        # EXCLUIR ROJO - Los equipos pueden usar rojo
        is_red_shirt = (h < 10 or h > 170) and s > 80 and v > 80

        # Solo considerar colores de árbitro si NO es rojo NI blanco
        is_ref_color = (is_black_shirt or is_yellow_shirt or is_bright_green_shirt) and not is_red_shirt and not is_white_shirt

        # Detectar pantalón negro (típico de árbitros)
        is_black_pants = v_pants < 80 and s_pants < 80

        # Calcular score de árbitro (MÁS ESTRICTO)
        referee_score = 0

        # Varianza alta es fuerte indicador (camiseta diferente de pantalón)
        if is_high_variance:
            referee_score += 3  # Aumentado de 2 a 3

        # Color típico de árbitro (amarillo/verde/negro) en camiseta
        if is_ref_color:
            referee_score += 3  # Aumentado de 2 a 3

        # Pantalón negro + camiseta de color → típico árbitro
        if is_black_pants and (is_yellow_shirt or is_bright_green_shirt):
            referee_score += 2  # Bonus por combinación típica

        # Muy diferente del promedio del grupo
        if shirt_distance_from_mean > 75:  # Aumentado de 65 a 75
            referee_score += 1
        if shirt_distance_from_mean > 95:  # Aumentado de 85 a 95
            referee_score += 1

        # Penalización por colores de jugador
        if is_white_shirt:
            referee_score -= 3  # Penalizar blanco fuertemente

        # Score mínimo más alto para clasificar como árbitro
        # Antes: >= 3, Ahora: >= 6
        if referee_score >= 6:
            referee_candidates.append(data)
        else:
            player_candidates.append(data)

    # Limitar árbitros a máximo 4 personas (usualmente 1-3 en cámara)
    if len(referee_candidates) > 4:
        # Ordenar por score combinado (varianza + distancia)
        referee_candidates.sort(
            key=lambda x: x['variance'] + np.linalg.norm(x['shirt'] - mean_shirt_color),
            reverse=True
        )
        extra_players = referee_candidates[4:]
        referee_candidates = referee_candidates[:4]
        player_candidates.extend(extra_players)

    # Si no hay suficientes jugadores, reclasificar
    if len(player_candidates) < 4:
        player_candidates.extend(referee_candidates)
        referee_candidates = []

    referee_indices = [d['idx'] for d in referee_candidates]

    # Paso 4: Clustering solo sobre jugadores usando color de camiseta
    if len(player_candidates) < 2:
        default_team1 = {'shirt': np.array([90, 100, 100]), 'pants': np.array([0, 50, 50])}
        default_team2 = {'shirt': np.array([0, 100, 100]), 'pants': np.array([0, 50, 50])}
        return default_team1, default_team2, [], [], referee_indices

    player_shirt_colors = np.array([d['shirt'] for d in player_candidates])

    # K-means con 2 clusters (los 2 equipos)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(player_shirt_colors)

    # Separar jugadores por equipo
    team1_data = [player_candidates[i] for i in range(len(player_candidates)) if labels[i] == 0]
    team2_data = [player_candidates[i] for i in range(len(player_candidates)) if labels[i] == 1]

    team1_indices = [d['idx'] for d in team1_data]
    team2_indices = [d['idx'] for d in team2_data]

    # Calcular colores representativos de cada equipo (camiseta y pantalón por separado)
    team1_shirt_color = kmeans.cluster_centers_[0]
    team2_shirt_color = kmeans.cluster_centers_[1]

    # Calcular color promedio de pantalón para cada equipo
    team1_pants_colors = np.array([d['pants'] for d in team1_data]) if team1_data else np.array([[0, 50, 50]])
    team2_pants_colors = np.array([d['pants'] for d in team2_data]) if team2_data else np.array([[0, 50, 50]])

    team1_pants_color = np.mean(team1_pants_colors, axis=0)
    team2_pants_color = np.mean(team2_pants_colors, axis=0)

    team1_colors = {'shirt': team1_shirt_color, 'pants': team1_pants_color}
    team2_colors = {'shirt': team2_shirt_color, 'pants': team2_pants_color}

    return team1_colors, team2_colors, team1_indices, team2_indices, referee_indices


def classify_person_smart(
    frame: np.ndarray,
    bbox: np.ndarray,
    team1_colors: Dict,
    team2_colors: Dict,
    frame_width: int,
    frame_height: int
) -> Tuple[str, int]:
    """
    Clasifica una persona como jugador de equipo 1, equipo 2, árbitro o portero.

    Mejoras v3.1:
    - Excluye ROJO de colores típicos de árbitros
    - Score mejorado para clasificación de árbitros
    - Mejor distinción entre equipos con camisetas coloridas

    Args:
        team1_colors: Dict con 'shirt' y 'pants' del equipo 1
        team2_colors: Dict con 'shirt' y 'pants' del equipo 2

    Returns:
        ('team1'|'team2'|'referee'|'goalkeeper', team_number)
    """
    features = extract_color_features(frame, bbox)
    person_shirt = features['shirt']
    person_pants = features['pants']
    color_variance = features['color_variance']

    in_goal_area = is_in_goal_area(bbox, frame_width, frame_height)

    # Calcular distancias a colores de equipos (MOVIDO ARRIBA para verificación de portero)
    dist_team1_shirt = np.linalg.norm(person_shirt - team1_colors['shirt'])
    dist_team2_shirt = np.linalg.norm(person_shirt - team2_colors['shirt'])

    # Si está en área de gol cerca de los extremos, probablemente es portero
    # Ampliado a 5% para cubrir mejor el área chica (ajustado de 12%)
    if in_goal_area:
        x_center = (bbox[0] + bbox[2]) / 2
        is_left_gk = x_center < frame_width * 0.05
        is_right_gk = x_center > frame_width * 0.95
        
        if is_left_gk or is_right_gk:
             # VERIFICACIÓN DE COLOR: El portero debe tener color distinto a los equipos
             # Si el color es muy similar a alguno de los equipos (< 45), es probable que sea un jugador en la banda
             min_dist_to_teams = min(dist_team1_shirt, dist_team2_shirt)
             
             # Solo clasificamos como portero si el color es suficientemente distinto
             if min_dist_to_teams > 45:
                 if is_left_gk:
                     return ('goalkeeper', 1)
                 else:
                     return ('goalkeeper', 2)
             # Si es muy similar, continuamos a la clasificación normal

    # Detectar árbitros con criterio mejorado v3.3:
    # 1. Alta varianza de color (camiseta diferente de pantalón)
    # 2. Color muy diferente de ambos equipos
    # 3. Patrones de color típicos de árbitros
    # 4. EXCLUIR blanco y rojo explícitamente (con umbrales relajados para rojo)
    min_team_dist = min(dist_team1_shirt, dist_team2_shirt)

    # Umbrales más estrictos
    high_variance_threshold = 50  # Aumentado de 45 a 50
    outlier_threshold = 75  # Aumentado de 70 a 75

    # Detectar colores típicos de árbitros (MEJORADO v3.2)
    h, s, v = person_shirt
    h_pants, s_pants, v_pants = person_pants

    # Colores de camiseta
    is_black_shirt = v < 70 and s < 70
    is_yellow_shirt = 20 < h < 35 and s > 100 and v > 130
    is_bright_green_shirt = 40 < h < 75 and s > 110 and v > 110

    # EXCLUIR BLANCO - Equipos usan blanco frecuentemente
    is_white_shirt = v > 180 and s < 50

    # EXCLUIR ROJO - Los equipos pueden usar rojo
    # Rango de Hue para rojo: 0-10 y 160-180
    # Relajamos S y V a > 50 para capturar rojos oscuros o desaturados
    is_red_shirt = (h < 15 or h > 165) and s > 50 and v > 50

    # Solo considerar colores de árbitro si NO es rojo NI blanco
    is_ref_color = (is_black_shirt or is_yellow_shirt or is_bright_green_shirt) and not is_red_shirt and not is_white_shirt

    # Detectar pantalón negro (típico de árbitros)
    is_black_pants = v_pants < 80 and s_pants < 80

    # Calcular score de árbitro (MÁS ESTRICTO)
    referee_score = 0

    # Varianza alta
    if color_variance > high_variance_threshold:
        referee_score += 3  # Aumentado de 2 a 3

    # Color típico de árbitro
    if is_ref_color:
        referee_score += 3  # Aumentado de 2 a 3

    # Combinación típica: camiseta de color + pantalón negro
    if is_black_pants and (is_yellow_shirt or is_bright_green_shirt):
        referee_score += 2

    # Muy diferente de ambos equipos
    if min_team_dist > outlier_threshold:
        referee_score += 2
    if min_team_dist > 95:  # Aumentado de 85 a 95
        referee_score += 1

    # Penalización por colores de jugador
    if is_white_shirt:
        referee_score -= 3  # Penalizar blanco fuertemente

    # Score mínimo más alto: >= 6 (antes era >= 4)
    is_likely_referee = referee_score >= 6

    if is_likely_referee:
        # Doble verificación: si es rojo, NO es árbitro (prioridad a equipo rojo)
        if is_red_shirt:
            # Si parece árbitro pero es rojo, forzar clasificación por distancia
            if dist_team1_shirt < dist_team2_shirt:
                return ('team1', 1)
            else:
                return ('team2', 2)
        return ('referee', 0)

    # Clasificar en equipos basándose en color de camiseta
    if dist_team1_shirt < dist_team2_shirt:
        return ('team1', 1)
    else:
        return ('team2', 2)


def process_video(
    source_path: str,
    target_path: str,
    player_model,
    ball_model=None,
    pitch_model=None,
    conf: float = 0.3,
    detection_mode: str = "players_and_ball",
    img_size: int = 640,
    full_field_approx: bool = False,
    progress_callback=None,
    enable_possession: bool = False,
    disable_inertia: bool = False
):
    """
    Procesa video completo con detección y tracking mejorado usando múltiples modelos.

    Args:
        source_path: Ruta al video de entrada
        target_path: Ruta al video de salida
        player_model: Modelo YOLO para detección de jugadores
        ball_model: Modelo YOLO para detección de pelota (opcional)
        pitch_model: Modelo YOLO para detección de campo (opcional)
        conf: Umbral de confianza para detecciones de personas
        detection_mode: Modo de detección ('players_only', 'ball_only', 'players_and_ball')
        img_size: Tamaño de imagen para inferencia
        full_field_approx: Si True, asume que la imagen completa es el campo (experimental)
    """
    source_path = str(source_path)
    target_path = str(target_path)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video fuente: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Trackers
    person_tracker = sv.ByteTrack()
    ball_tracker = sv.ByteTrack()

    # Configurar modelo de pitch si se proporciona O si se usa aproximación
    pitch_config = None
    pitch_model_type = 'default'  # Default fallback

    if pitch_model or full_field_approx:
        # Detectar automáticamente el tipo de modelo basado en número de keypoints
        if pitch_model:
            try:
                # Hacer inferencia en un frame vacío para detectar estructura del modelo
                test_results = pitch_model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                if test_results and test_results[0].keypoints is not None:
                    num_keypoints = test_results[0].keypoints.data.shape[1] if len(test_results[0].keypoints.data.shape) > 1 else 0

                    # Determinar tipo de modelo por número de keypoints
                    if num_keypoints == 29:
                        pitch_model_type = 'soccana'
                        print(f"✅ Modelo Soccana detectado: {num_keypoints} keypoints")
                    elif num_keypoints == 32:
                        pitch_model_type = 'roboflow'  # homography.pt tiene 32 kps como roboflow
                        print(f"✅ Modelo Homography/Roboflow detectado: {num_keypoints} keypoints")
                    else:
                        print(f"⚠️ Modelo desconocido con {num_keypoints} keypoints, usando default")
            except Exception as e:
                print(f"⚠️ No se pudo detectar tipo de modelo: {e}, usando default")

        # Crear configuración con el tipo correcto
        pitch_config = SoccerPitchConfiguration(model_type=pitch_model_type)
        homography_manager = HomographyManager()
        if disable_inertia:
            homography_manager.max_inertia_frames = 0

    # Mejora J: Anotadores estilo broadcast (elipse + trace)
    team1_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=2)
    team1_trace_annotator = sv.TraceAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=1, trace_length=30)
    team1_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.BLACK, text_padding=3)

    team2_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex("#00BFFF"), thickness=2)
    team2_trace_annotator = sv.TraceAnnotator(color=sv.Color.from_hex("#00BFFF"), thickness=1, trace_length=30)
    team2_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.WHITE, text_padding=3)

    goalkeeper_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex("#9B59B6"), thickness=2)
    goalkeeper_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.WHITE, text_padding=3)

    referee_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex("#FFD700"), thickness=2)
    referee_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_color=sv.Color.BLACK, text_padding=3)

    ball_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF0000"), thickness=3)
    ball_label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, text_color=sv.Color.WHITE, text_padding=4)

    # Variables de estado
    team1_colors = None
    team2_colors = None
    frame_count = 0
    reference_team1_color = None
    reference_team2_color = None
    clustering_initialized = False
    track_history = {}
    HISTORY_LEN = 30

    # Suavizado temporal para posiciones en radar
    radar_positions_history = {}  # tracker_id -> deque de posiciones (x, y)
    RADAR_SMOOTH_WINDOW = 5       # Ventana de suavizado (5 frames)

    # Mejora H: historial de posiciones para trails en radar
    TRAIL_LENGTH = 15
    ball_trail = deque(maxlen=TRAIL_LENGTH)  # deque de (x, y)
    debug_homography_overlay: bool = True
    heatmap_bins_team1 = np.zeros((53, 34), dtype=np.float32)
    heatmap_bins_team2 = np.zeros((53, 34), dtype=np.float32)
    heatmap_sample_every = 5
    heatmap_samples_count = 0
    debug_scouting = False
    _heatmap_valid_frame_counter = 0
    qc_sample_every = 10
    frame_qc_samples = []
    homography_telemetry = {
        'frame_number': [],
        'homography_mode': [],
        'reproj_error': [],
        'delta_H': [],
        'inlier_ratio': [],
        'team1_centroid_x': [],
        'team1_centroid_y': [],
        'team2_centroid_x': [],
        'team2_centroid_y': [],
        'team1_percent_out_of_bounds': [],
        'team2_percent_out_of_bounds': []
    }
    health_timeline = []
    track_age = {}
    prev_track_ids = set()
    ball_track_age_map = {}
    prev_field_positions = {}
    max_speed_mps_list = []
    speed_violation_flags = []
    max_jump_m_list = []
    jump_violation_flags = []
    churn_ratio_list = []
    frame_valid_demo_flags = []
    frame_valid_strict_flags = []
    last_good_H = None
    last_good_age_frames = None

    # Inicializar módulos tácticos
    formation_detector = FormationDetector()
    metrics_calculator = TacticalMetricsCalculator()
    team1_tracker = TacticalMetricsTracker(history_size=5000)
    team2_tracker = TacticalMetricsTracker(history_size=5000)
    formations_timeline = {'team1': [], 'team2': []}

    # --- INICIALIZAR MÓDULOS DE POSESIÓN (si está habilitado) ---
    possession_tracker = None
    speed_estimator = None
    if enable_possession:
        from src.controllers.ball_possession_tracker import BallPossessionTracker
        from src.controllers.speed_distance_estimator import SpeedDistanceEstimator
        possession_tracker = BallPossessionTracker(
            max_player_ball_distance_px=70.0,
            possession_inertia=5,
        )
        speed_estimator = SpeedDistanceEstimator(fps=fps, window_frames=5)

    # --- IDENTIFICAR CLASES DEL MODELO DE JUGADORES ---
    player_model_names = player_model.names
    player_class_ids = []
    # Fallback ball class IDs from player model (if ball model is missing)
    player_model_ball_class_ids = []

    # Detectar si es modelo fine-tuneado con clases específicas de fútbol
    # (player, goalkeeper, referee, ball) vs COCO genérico (person, sports ball)
    _names_lower = {k: str(v).lower() for k, v in player_model_names.items()}
    _all_names = set(_names_lower.values())
    is_football_model = ('player' in _all_names or 'referee' in _all_names)

    # IDs específicos para modelo fine-tuneado
    referee_class_ids = []
    goalkeeper_class_ids = []

    for id, name in player_model_names.items():
        name_lower = str(name).lower()
        # Clases para personas (jugadores, árbitros, porteros, personas genéricas)
        if any(x in name_lower for x in ['person', 'player', 'goalkeeper', 'referee']):
            player_class_ids.append(id)
        # Clases para pelota en el modelo de jugadores
        if any(x in name_lower for x in ['ball', 'sports ball']):
            player_model_ball_class_ids.append(id)
        # Mapear clases específicas del modelo fine-tuneado
        if is_football_model:
            if name_lower == 'referee':
                referee_class_ids.append(id)
            if name_lower == 'goalkeeper':
                goalkeeper_class_ids.append(id)

    if is_football_model:
        print(f"✅ Modelo fine-tuneado de fútbol detectado: {dict(player_model_names)}")

    # Fallbacks por defecto si no se detectan nombres conocidos
    if not player_class_ids:
        # Asumir clase 0 si es un modelo custom desconocido o COCO standard
        player_class_ids = [0]

    if not player_model_ball_class_ids:
        # Solo si parece ser COCO (muchas clases), usamos 32
        if len(player_model_names) > 30:
            player_model_ball_class_ids = [32]

    # --- ESTADO PARA INTERPOLACIÓN DE PELOTA (Mejora A + F: Kalman) ---
    last_known_ball_bbox = None      # Última bbox válida de pelota
    ball_missing_frames = 0          # Frames consecutivos sin pelota
    BALL_INTERP_MAX_FRAMES = 10      # Máximo de frames a interpolar

    # Mejora F: Kalman filter para pelota (estado = [cx, cy, vx, vy])
    ball_kalman = cv2.KalmanFilter(4, 2)
    ball_kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    ball_kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], np.float32)
    ball_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    ball_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    ball_kalman_initialized = False

    # --- CACHÉ DE EQUIPO POR TRACKER_ID (Mejora C) ---
    team_cache = {}                  # tracker_id → {'team': str, 'confidence': int}
    TEAM_CACHE_LOCK_THRESHOLD = 15   # Después de N votos consistentes, bloquear asignación
            
    # --- IDENTIFICAR CLASES DEL MODELO DE PELOTA (si existe) ---
    ball_model_class_ids = []
    if ball_model:
        for id, name in ball_model.names.items():
            name_lower = str(name).lower()
            if any(x in name_lower for x in ['ball', 'sports ball']):
                ball_model_class_ids.append(id)
        
        # Fallback si no se encuentra nombre explícito
        if not ball_model_class_ids:
            # Si es modelo de 1 sola clase, asumimos que es la pelota
            if len(ball_model.names) == 1:
                ball_model_class_ids = [0]
            # Si parece ser COCO
            elif len(ball_model.names) > 30:
                ball_model_class_ids = [32]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if progress_callback and total_frames > 0 and frame_count % 10 == 0:
                try:
                    progress_callback(frame_count / max(1, total_frames), frame_count, total_frames)
                except Exception:
                    pass
            annotated_frame = frame.copy()

            # --- DETECCIÓN DE JUGADORES ---
            player_results = player_model.predict(
                frame,
                conf=conf,
                iou=0.3,
                imgsz=img_size,
                max_det=100,
                verbose=False
            )
            player_detections = sv.Detections.from_ultralytics(player_results[0])
            
            # Filtrar clases dinámicamente usando los IDs identificados
            if player_detections.class_id is not None:
                mask = np.isin(player_detections.class_id, player_class_ids)
                player_detections = player_detections[mask]
            
            # --- DETECCIÓN DE PELOTA MEJORADA ---
            ball_detections = sv.Detections.empty()
            if detection_mode in ["ball_only", "players_and_ball"]:
                if ball_model:
                    # Usar modelo específico de pelota con parámetros optimizados
                    ball_results = ball_model.predict(
                        frame,
                        conf=max(0.1, conf * 0.3),  # Umbral aún más bajo (10% mínimo)
                        iou=0.4,                     # IoU más permisivo para pelota
                        imgsz=img_size,
                        verbose=False
                    )
                    ball_detections = sv.Detections.from_ultralytics(ball_results[0])
                    
                    # Filtrar clases válidas del modelo de pelota
                    if ball_detections.class_id is not None and ball_model_class_ids:
                         mask = np.isin(ball_detections.class_id, ball_model_class_ids)
                         ball_detections = ball_detections[mask]

                    # Post-procesamiento: Filtrar por tamaño (pelotas muy grandes = falsos positivos)
                    if len(ball_detections) > 0:
                        areas = (ball_detections.xyxy[:, 2] - ball_detections.xyxy[:, 0]) * \
                                (ball_detections.xyxy[:, 3] - ball_detections.xyxy[:, 1])
                        max_ball_area = (width * 0.05) * (height * 0.05)  # Máximo 5% del frame
                        min_ball_area = (width * 0.005) * (height * 0.005)  # Mínimo 0.5% del frame
                        valid_size = (areas < max_ball_area) & (areas > min_ball_area)
                        ball_detections = ball_detections[valid_size]

                        # Si hay múltiples detecciones, tomar la de mayor confianza
                        if len(ball_detections) > 1:
                            best_idx = np.argmax(ball_detections.confidence)
                            ball_detections = ball_detections[best_idx:best_idx+1]
                else:
                    # Fallback: Usar modelo de jugadores si tiene clase de pelota
                    raw_detections = sv.Detections.from_ultralytics(player_results[0])
                    if raw_detections.class_id is not None and player_model_ball_class_ids:
                         mask = np.isin(raw_detections.class_id, player_model_ball_class_ids)
                         ball_detections = raw_detections[mask]
                         if len(ball_detections) > 0:
                             ball_conf_threshold = max(0.1, conf * 0.3)
                             ball_detections = ball_detections[ball_detections.confidence >= ball_conf_threshold]

                             # Mismo post-procesamiento
                             areas = (ball_detections.xyxy[:, 2] - ball_detections.xyxy[:, 0]) * \
                                     (ball_detections.xyxy[:, 3] - ball_detections.xyxy[:, 1])
                             max_ball_area = (width * 0.05) * (height * 0.05)
                             min_ball_area = (width * 0.005) * (height * 0.005)
                             valid_size = (areas < max_ball_area) & (areas > min_ball_area)
                             ball_detections = ball_detections[valid_size]

                             if len(ball_detections) > 1:
                                 best_idx = np.argmax(ball_detections.confidence)
                                 ball_detections = ball_detections[best_idx:best_idx+1]

            # --- TRACKING DE JUGADORES ---
            tracked_persons = person_tracker.update_with_detections(player_detections)
            
            # Actualizar modelos de color (Clustering)
            if (frame_count % 45 == 1 or team1_colors is None) and len(player_detections) > 0:
                team1_colors_new, team2_colors_new, _, _, _ = cluster_teams(
                    frame, player_detections, width, height
                )

                if not clustering_initialized:
                    reference_team1_color = team1_colors_new['shirt'].copy()
                    reference_team2_color = team2_colors_new['shirt'].copy()
                    team1_colors = team1_colors_new
                    team2_colors = team2_colors_new
                    clustering_initialized = True
                else:
                    # Verificar intercambio
                    dist_t1_ref1 = np.linalg.norm(team1_colors_new['shirt'] - reference_team1_color)
                    dist_t1_ref2 = np.linalg.norm(team1_colors_new['shirt'] - reference_team2_color)
                    dist_t2_ref1 = np.linalg.norm(team2_colors_new['shirt'] - reference_team1_color)
                    dist_t2_ref2 = np.linalg.norm(team2_colors_new['shirt'] - reference_team2_color)

                    score_keep = dist_t1_ref1 + dist_t2_ref2
                    score_swap = dist_t1_ref2 + dist_t2_ref1

                    if score_swap < score_keep:
                        team1_colors = team2_colors_new
                        team2_colors = team1_colors_new
                    else:
                        team1_colors = team1_colors_new
                        team2_colors = team2_colors_new
                    
                    alpha = 0.1
                    reference_team1_color = (1 - alpha) * reference_team1_color + alpha * team1_colors['shirt']
                    reference_team2_color = (1 - alpha) * reference_team2_color + alpha * team2_colors['shirt']

            # Clasificar cada persona rastreada
            team1_mask = []
            team2_mask = []
            referee_mask = []
            goalkeeper_mask = []

            # --- MEJORA B: Mapear clases nativas del modelo fine-tuneado ---
            # Si el modelo tiene clases referee/goalkeeper/ball nativas,
            # construimos un dict tracker_id → class_name desde las detecciones originales
            native_class_map = {}  # índice en tracked_persons → clase nativa
            if is_football_model and tracked_persons.class_id is not None:
                for i in range(len(tracked_persons)):
                    cls_id = tracked_persons.class_id[i]
                    cls_name = _names_lower.get(int(cls_id), '')
                    if cls_name in ('referee', 'goalkeeper'):
                        native_class_map[i] = cls_name

            if len(tracked_persons) > 0 and team1_colors is not None:
                for i, (xyxy, _, _, _, tracker_id, _) in enumerate(tracked_persons):
                    if not is_in_playing_field(xyxy, width, height):
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)
                        continue

                    # --- MEJORA B: Usar clase nativa si disponible ---
                    if i in native_class_map:
                        current_vote = native_class_map[i]
                    else:
                        in_goal = is_in_goal_area(xyxy, width, height)
                        x_center = (xyxy[0] + xyxy[2]) / 2

                        if in_goal and (x_center < width * 0.05 or x_center > width * 0.95):
                            current_vote = 'goalkeeper'
                        else:
                            person_type, _ = classify_person_smart(
                                frame, xyxy, team1_colors, team2_colors, width, height
                            )
                            current_vote = person_type

                    if tracker_id not in track_history:
                        track_history[tracker_id] = deque(maxlen=HISTORY_LEN)

                    track_history[tracker_id].append(current_vote)
                    vote_counts = Counter(track_history[tracker_id])
                    most_common = vote_counts.most_common(1)[0][0]

                    if current_vote == 'goalkeeper':
                        final_class = 'goalkeeper'
                    else:
                        # --- MEJORA C: Caché de equipo por tracker_id ---
                        # Si el tracker ya tiene suficientes votos consistentes, bloquear
                        if tracker_id in team_cache:
                            cached = team_cache[tracker_id]
                            if cached['confidence'] >= TEAM_CACHE_LOCK_THRESHOLD:
                                final_class = cached['team']
                            else:
                                final_class = most_common
                                if final_class == cached['team']:
                                    cached['confidence'] += 1
                                else:
                                    cached['confidence'] = max(0, cached['confidence'] - 1)
                                    if cached['confidence'] == 0:
                                        cached['team'] = final_class
                                        cached['confidence'] = 1
                        else:
                            final_class = most_common
                            team_cache[tracker_id] = {'team': final_class, 'confidence': 1}

                    if final_class == 'goalkeeper':
                        goalkeeper_mask.append(True)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)
                    elif final_class == 'referee':
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(True)
                    elif final_class == 'team1':
                        goalkeeper_mask.append(False)
                        team1_mask.append(True)
                        team2_mask.append(False)
                        referee_mask.append(False)
                    elif final_class == 'team2':
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(True)
                        referee_mask.append(False)
                    else:
                        goalkeeper_mask.append(False)
                        team1_mask.append(False)
                        team2_mask.append(False)
                        referee_mask.append(False)

            refs_detected_count = int(np.sum(referee_mask))
            players_detected_count = int(np.sum(team1_mask) + np.sum(team2_mask) + np.sum(goalkeeper_mask))
            refs_filtered_out_count = refs_detected_count
            
            # Anotar
            if any(team1_mask):
                t1_dets = tracked_persons[np.array(team1_mask)]
                annotated_frame = team1_annotator.annotate(scene=annotated_frame, detections=t1_dets)
                annotated_frame = team1_trace_annotator.annotate(scene=annotated_frame, detections=t1_dets)
                if t1_dets.tracker_id is not None:
                    labels = [f"Team 1 #{tid}" for tid in t1_dets.tracker_id]
                    annotated_frame = team1_label_annotator.annotate(scene=annotated_frame, detections=t1_dets, labels=labels)

            if any(team2_mask):
                t2_dets = tracked_persons[np.array(team2_mask)]
                annotated_frame = team2_annotator.annotate(scene=annotated_frame, detections=t2_dets)
                annotated_frame = team2_trace_annotator.annotate(scene=annotated_frame, detections=t2_dets)
                if t2_dets.tracker_id is not None:
                    labels = [f"Team 2 #{tid}" for tid in t2_dets.tracker_id]
                    annotated_frame = team2_label_annotator.annotate(scene=annotated_frame, detections=t2_dets, labels=labels)

            if any(goalkeeper_mask):
                gk_dets = tracked_persons[np.array(goalkeeper_mask)]
                annotated_frame = goalkeeper_annotator.annotate(scene=annotated_frame, detections=gk_dets)
                if gk_dets.tracker_id is not None:
                    labels = [f"GK #{tid}" for tid in gk_dets.tracker_id]
                    annotated_frame = goalkeeper_label_annotator.annotate(scene=annotated_frame, detections=gk_dets, labels=labels)

            if any(referee_mask):
                ref_dets = tracked_persons[np.array(referee_mask)]
                annotated_frame = referee_annotator.annotate(scene=annotated_frame, detections=ref_dets)
                if ref_dets.tracker_id is not None:
                    labels = [f"Referee #{tid}" for tid in ref_dets.tracker_id]
                    annotated_frame = referee_label_annotator.annotate(scene=annotated_frame, detections=ref_dets, labels=labels)

            # --- TRACKING DE PELOTA + KALMAN (Mejora A + F) ---
            tracked_ball = None
            if len(ball_detections) > 0:
                tracked_ball = ball_tracker.update_with_detections(ball_detections)
                if len(tracked_ball) > 0:
                    annotated_frame = ball_annotator.annotate(scene=annotated_frame, detections=tracked_ball)
                    ball_labels = ["BALL"] * len(tracked_ball)
                    annotated_frame = ball_label_annotator.annotate(scene=annotated_frame, detections=tracked_ball, labels=ball_labels)
                    last_known_ball_bbox = tracked_ball.xyxy[0].copy()
                    ball_missing_frames = 0
                    # Mejora F: actualizar Kalman con medición real
                    bb = tracked_ball.xyxy[0]
                    cx = (bb[0] + bb[2]) / 2
                    cy = (bb[1] + bb[3]) / 2
                    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                    if not ball_kalman_initialized:
                        ball_kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                        ball_kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                        ball_kalman_initialized = True
                    ball_kalman.correct(measurement)
                    ball_kalman.predict()
                else:
                    tracked_ball = None
                    ball_missing_frames += 1
            elif ball_kalman_initialized and ball_missing_frames < BALL_INTERP_MAX_FRAMES:
                # Mejora F: Kalman predict en vez de frozen-position
                ball_missing_frames += 1
                prediction = ball_kalman.predict()
                pred_cx = float(prediction[0])
                pred_cy = float(prediction[1])
                # Reconstruir bbox del mismo tamaño que la última conocida
                if last_known_ball_bbox is not None:
                    half_w = (last_known_ball_bbox[2] - last_known_ball_bbox[0]) / 2
                    half_h = (last_known_ball_bbox[3] - last_known_ball_bbox[1]) / 2
                else:
                    half_w, half_h = 10.0, 10.0
                synthetic_xyxy = np.array([[
                    pred_cx - half_w, pred_cy - half_h,
                    pred_cx + half_w, pred_cy + half_h
                ]], dtype=np.float32)
                synthetic_conf = np.array([max(0.1, 0.5 - ball_missing_frames * 0.04)])
                ball_detections = sv.Detections(
                    xyxy=synthetic_xyxy,
                    confidence=synthetic_conf,
                )
                tracked_ball = ball_tracker.update_with_detections(ball_detections)
            else:
                ball_missing_frames += 1

            ball_detected = bool(tracked_ball is not None and len(tracked_ball) > 0)
            ball_conf = None
            if ball_detected and getattr(tracked_ball, "confidence", None) is not None and len(tracked_ball.confidence) > 0:
                ball_conf = float(tracked_ball.confidence[0])
            ball_track_age = None
            if ball_detected and tracked_ball.tracker_id is not None and len(tracked_ball.tracker_id) > 0:
                ball_tid = int(tracked_ball.tracker_id[0])
                for tid in list(ball_track_age_map.keys()):
                    if tid != ball_tid:
                        del ball_track_age_map[tid]
                ball_track_age_map[ball_tid] = ball_track_age_map.get(ball_tid, 0) + 1
                ball_track_age = int(ball_track_age_map[ball_tid])

            # --- ANÁLISIS DE POSESIÓN (fallback píxeles, se mejora abajo con métrico) ---
            possession_result = None
            _possession_assigned_metric = False
            if possession_tracker is not None and not pitch_config:
                # Sin homografía: usar método original en píxeles
                possession_result = possession_tracker.assign_possession(
                    tracked_persons, team1_mask, team2_mask, tracked_ball
                )
                annotated_frame = possession_tracker.draw_possession_bar(annotated_frame)

            # --- RADAR ---
            max_speed_mps = None
            speed_violation = False
            max_jump_m = None
            homography_mode = "fallback"
            using_last_good = False
            spread_ok = None
            metrics1 = None
            metrics2 = None
            team1_centroid = None
            team2_centroid = None
            team1_oob = None
            team2_oob = None
            if pitch_config:  # Si hay configuración de pitch (ya sea por modelo o approx)
                transformer = None
                if last_good_H is not None:
                    last_good_age_frames = 0 if last_good_age_frames is None else last_good_age_frames + 1
                
                # Caso A: Modelo de Pitch disponible
                if pitch_model:
                    try:
                        pitch_results = pitch_model(frame, verbose=False, conf=0.01)[0]
                        if pitch_results.keypoints is not None and len(pitch_results.keypoints) > 0:
                            keypoints_xy = pitch_results.keypoints.xy.cpu().numpy()[0]
                            keypoints_conf = pitch_results.keypoints.conf.cpu().numpy()[0]
                            conf_threshold = 0.05 if pitch_model_type == 'soccana' else 0.5
                            valid_kp_mask = keypoints_conf > conf_threshold
                            valid_keypoints = keypoints_xy[valid_kp_mask]
                            valid_indices = np.where(valid_kp_mask)[0]
                            mapped_indices_mask = np.isin(valid_indices, list(pitch_config.keypoints_map.keys()))
                            valid_indices = valid_indices[mapped_indices_mask]
                            valid_keypoints = valid_keypoints[mapped_indices_mask]
                            if len(valid_keypoints) >= 4:
                                target_points = pitch_config.get_keypoints_from_ids(valid_indices)
                                valid_confidences = keypoints_conf[valid_kp_mask][mapped_indices_mask]
                                homography_manager.update(valid_keypoints, target_points, valid_confidences, (width, height))
                                if homography_manager.cut_detected:
                                    homography_manager.start_reacquire()
                                transformer = homography_manager.get_transformer()
                    except Exception as e:
                        if frame_count % 100 == 0:
                            print(f"Error en inferencia de pitch (frame {frame_count}): {e}")

                if transformer is None and (full_field_approx or pitch_model):
                    margin_top = height * 0.15     # 15% superior es cielo/público
                    margin_bottom = height * 0.05   # 5% inferior es fuera del campo
                    margin_left = width * 0.08      # 8% lateral (bancas)
                    margin_right = width * 0.08     # 8% lateral

                    source_points = np.array([
                        [margin_left, margin_top],                          # Top-Left (ajustado)
                        [width - margin_right, margin_top],                 # Top-Right (ajustado)
                        [width - margin_right, height - margin_bottom],     # Bottom-Right (ajustado)
                        [margin_left, height - margin_bottom]               # Bottom-Left (ajustado)
                    ], dtype=np.float32)

                    # Mapear a las esquinas del campo usando el método correcto
                    # Esto funciona tanto con Roboflow Sports (IDs: 0, 5, 29, 24)
                    # como con configuración default (IDs: 0, 1, 2, 3)
                    corner_ids = pitch_config.get_corner_keypoint_ids()
                    target_points = np.array([
                        pitch_config.keypoints_map[corner_ids[0]],  # Top-Left corner
                        pitch_config.keypoints_map[corner_ids[1]],  # Top-Right corner
                        pitch_config.keypoints_map[corner_ids[2]],  # Bottom-Right corner
                        pitch_config.keypoints_map[corner_ids[3]]   # Bottom-Left corner
                    ], dtype=np.float32)

                    transformer = ViewTransformer(source_points, target_points)
                    homography_manager.last_state = "FULLSCREEN"
                    homography_manager.mode = "FALLBACK"
                    if frame_count % 100 == 0:
                        print("fallback full-screen")

                # Si tenemos un transformer válido, proyectamos y dibujamos
                homography_state = getattr(homography_manager, 'last_state', "")
                hm_mode = getattr(homography_manager, 'mode', "fallback")
                if hm_mode == "ACQUIRE":
                    homography_mode = "acquire"
                elif hm_mode == "TRACK":
                    homography_mode = "tracking"
                elif hm_mode == "INERTIA":
                    homography_mode = "inertia"
                elif hm_mode == "REACQUIRE":
                    homography_mode = "reacquire"
                else:
                    homography_mode = "fallback"
                if homography_state:
                    spread_ok = homography_state != "INVALID_SPREAD"
                reproj_error_m_pre = getattr(homography_manager, 'last_reproj_error', None)
                inlier_ratio_pre = getattr(homography_manager, 'last_inlier_ratio', None)
                reproj_nan_pre = False
                if reproj_error_m_pre is not None:
                    try:
                        reproj_nan_pre = bool(np.isnan(reproj_error_m_pre))
                    except Exception:
                        reproj_nan_pre = False
                invalid_state_pre = bool(homography_state.startswith("INVALID"))
                fallback_usable = (
                    homography_mode == "fallback"
                    and last_good_H is not None
                    and reproj_error_m_pre is not None
                    and not reproj_nan_pre
                    and reproj_error_m_pre <= REPROJ_WARN_MAX
                    and (inlier_ratio_pre is None or inlier_ratio_pre >= 0.6)
                    and not invalid_state_pre
                )
                if transformer is None and fallback_usable:
                    transformer = ViewTransformer(m=last_good_H)
                    homography_mode = "inertia"
                    using_last_good = True
                if transformer:
                    try:
                        points_to_transform = {}

                        def get_bottom_center(dets):
                            return np.column_stack([
                                (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2,
                                dets.xyxy[:, 3]
                            ])

                        def smooth_positions(tracker_ids, raw_positions):
                            """Suaviza posiciones usando promedio móvil."""
                            smoothed = []
                            for tid, pos in zip(tracker_ids, raw_positions):
                                if tid not in radar_positions_history:
                                    radar_positions_history[tid] = deque(maxlen=RADAR_SMOOTH_WINDOW)

                                radar_positions_history[tid].append(pos)
                                # Promedio de las últimas N posiciones
                                avg_pos = np.mean(list(radar_positions_history[tid]), axis=0)
                                smoothed.append(avg_pos)
                            return np.array(smoothed)

                        # Transformar y suavizar team1 (con flip_x para corregir inversión)
                        if any(team1_mask):
                            t1_dets = tracked_persons[np.array(team1_mask)]
                            raw_pos = transformer.transform_points(get_bottom_center(t1_dets), flip_x=True)
                            if t1_dets.tracker_id is not None:
                                points_to_transform['team1'] = smooth_positions(t1_dets.tracker_id, raw_pos)
                            else:
                                points_to_transform['team1'] = raw_pos

                        # Transformar y suavizar team2 (con flip_x para corregir inversión)
                        if any(team2_mask):
                            t2_dets = tracked_persons[np.array(team2_mask)]
                            raw_pos = transformer.transform_points(get_bottom_center(t2_dets), flip_x=True)
                            if t2_dets.tracker_id is not None:
                                points_to_transform['team2'] = smooth_positions(t2_dets.tracker_id, raw_pos)
                            else:
                                points_to_transform['team2'] = raw_pos
                        
                        if 'team1' in points_to_transform and len(points_to_transform['team1']) > 0:
                            t1_arr = points_to_transform['team1']
                            team1_centroid = (float(np.mean(t1_arr[:, 0])), float(np.mean(t1_arr[:, 1])))
                            oob_mask = (t1_arr[:, 0] < 0) | (t1_arr[:, 0] > 105) | (t1_arr[:, 1] < 0) | (t1_arr[:, 1] > 68)
                            team1_oob = float(np.sum(oob_mask) / max(len(t1_arr), 1) * 100.0)
                        if 'team2' in points_to_transform and len(points_to_transform['team2']) > 0:
                            t2_arr = points_to_transform['team2']
                            team2_centroid = (float(np.mean(t2_arr[:, 0])), float(np.mean(t2_arr[:, 1])))
                            oob_mask = (t2_arr[:, 0] < 0) | (t2_arr[:, 0] > 105) | (t2_arr[:, 1] < 0) | (t2_arr[:, 1] > 68)
                            team2_oob = float(np.sum(oob_mask) / max(len(t2_arr), 1) * 100.0)

                        # Transformar árbitros (sin suavizar mucho ya que se mueven menos)
                        if any(referee_mask):
                            ref_dets = tracked_persons[np.array(referee_mask)]
                            points_to_transform['referee'] = transformer.transform_points(
                                get_bottom_center(ref_dets), flip_x=True
                            )

                        # Transformar porteros
                        if any(goalkeeper_mask):
                            gk_dets = tracked_persons[np.array(goalkeeper_mask)]
                            points_to_transform['goalkeeper'] = transformer.transform_points(
                                get_bottom_center(gk_dets), flip_x=True
                            )

                        # Transformar pelota (suavizado agresivo por su movimiento rápido)
                        if tracked_ball is not None and len(tracked_ball) > 0:
                            raw_ball_pos = transformer.transform_points(get_bottom_center(tracked_ball), flip_x=True)
                            if tracked_ball.tracker_id is not None and len(tracked_ball.tracker_id) > 0:
                                ball_tid = tracked_ball.tracker_id[0]
                                if ball_tid not in radar_positions_history:
                                    radar_positions_history[ball_tid] = deque(maxlen=3)  # Ventana más corta para pelota
                                radar_positions_history[ball_tid].append(raw_ball_pos[0])
                                smooth_ball = np.mean(list(radar_positions_history[ball_tid]), axis=0)
                                points_to_transform['ball'] = np.array([smooth_ball])
                            else:
                                points_to_transform['ball'] = raw_ball_pos
                        
                        # --- VELOCIDAD Y DISTANCIA (posesión) ---
                        speeds_kmh_frame = []
                        max_jump_m = None
                        if speed_estimator is not None:
                            for team_key in ('team1', 'team2'):
                                mask_arr = np.array(team1_mask if team_key == 'team1' else team2_mask)
                                if team_key in points_to_transform and any(mask_arr):
                                    team_dets = tracked_persons[mask_arr]
                                    if team_dets.tracker_id is not None:
                                        speed_data = speed_estimator.update(
                                            team_dets.tracker_id.tolist(),
                                            points_to_transform[team_key],
                                            [team_key] * len(team_dets),
                                        )
                                        for tid, data in speed_data.items():
                                            speed_kmh = data.get('speed_kmh')
                                            if speed_kmh is not None:
                                                speeds_kmh_frame.append(float(speed_kmh))
                                        for j, tid in enumerate(team_dets.tracker_id):
                                            if tid in speed_data and speed_data[tid]['speed_kmh'] > 2.0:
                                                annotated_frame = speed_estimator.draw_player_speed(
                                                    annotated_frame, tid, team_dets.xyxy[j],
                                                    speed_data[tid]['speed_kmh'],
                                                )
                        for team_key in ('team1', 'team2'):
                            mask_arr = np.array(team1_mask if team_key == 'team1' else team2_mask)
                            if team_key in points_to_transform and any(mask_arr):
                                team_dets = tracked_persons[mask_arr]
                                if team_dets.tracker_id is not None:
                                    for j, tid in enumerate(team_dets.tracker_id):
                                        tid = int(tid)
                                        pos = points_to_transform[team_key][j]
                                        if tid in prev_field_positions:
                                            dist = float(np.linalg.norm(pos - prev_field_positions[tid]))
                                            if max_jump_m is None or dist > max_jump_m:
                                                max_jump_m = dist
                                        prev_field_positions[tid] = pos.copy()
                        if speeds_kmh_frame:
                            max_speed_mps = float(max(speeds_kmh_frame) / 3.6)
                            speed_violation = max_speed_mps > SPEED_MAX_MPS

                        # --- Fix C: POSESIÓN EN ESPACIO MÉTRICO ---
                        if possession_tracker is not None:
                            ball_field = None
                            if 'ball' in points_to_transform and len(points_to_transform['ball']) > 0:
                                ball_field = points_to_transform['ball'][0]
                            # Combinar tracker_ids y team_labels de team1+team2
                            all_tids = []
                            all_field_pos = []
                            all_team_labels = []
                            for tk in ('team1', 'team2'):
                                mask_arr = np.array(team1_mask if tk == 'team1' else team2_mask)
                                if tk in points_to_transform and any(mask_arr):
                                    team_dets = tracked_persons[mask_arr]
                                    if team_dets.tracker_id is not None:
                                        all_tids.extend(team_dets.tracker_id.tolist())
                                        all_field_pos.extend(points_to_transform[tk].tolist())
                                        all_team_labels.extend([tk] * len(team_dets))
                            if all_tids and ball_field is not None:
                                possession_result = possession_tracker.assign_possession_metric(
                                    all_tids,
                                    np.array(all_field_pos),
                                    all_team_labels,
                                    ball_field,
                                )
                                _possession_assigned_metric = True
                            annotated_frame = possession_tracker.draw_possession_bar(annotated_frame)

                        # --- ACTUALIZACIÓN DE MÉTRICAS TÁCTICAS ---
                        # Fix D: auto-detectar dirección de ataque por centroide
                        t1_dir = "right"
                        t2_dir = "right"
                        if 'team1' in points_to_transform and 'team2' in points_to_transform:
                            if len(points_to_transform['team1']) > 0 and len(points_to_transform['team2']) > 0:
                                cx1 = float(np.mean(points_to_transform['team1'][:, 0]))
                                cx2 = float(np.mean(points_to_transform['team2'][:, 0]))
                                if cx1 < cx2:
                                    t1_dir = "right"
                                    t2_dir = "left"
                                else:
                                    t1_dir = "left"
                                    t2_dir = "right"

                        if 'team1' in points_to_transform and len(points_to_transform['team1']) > 0:
                            formation1 = formation_detector.detect_formation(points_to_transform['team1'], team_attacking_direction=t1_dir)
                            formations_timeline['team1'].append(formation1)

                            metrics1 = metrics_calculator.calculate_all_metrics(points_to_transform['team1'])
                            team1_tracker.update(metrics1, frame_count)

                        if 'team2' in points_to_transform and len(points_to_transform['team2']) > 0:
                            formation2 = formation_detector.detect_formation(points_to_transform['team2'], team_attacking_direction=t2_dir)
                            formations_timeline['team2'].append(formation2)
                            
                            metrics2 = metrics_calculator.calculate_all_metrics(points_to_transform['team2'])
                            team2_tracker.update(metrics2, frame_count)
                        
                        _heatmap_valid_frame_counter += 1
                        last_reproj = getattr(homography_manager, 'last_reproj_error', None)
                        allow_heatmap = homography_mode == "tracking"
                        if not allow_heatmap and homography_mode == "inertia":
                            allow_heatmap = last_reproj is not None and last_reproj < homography_manager.max_reproj_error
                        if allow_heatmap and _heatmap_valid_frame_counter % heatmap_sample_every == 0:
                            sampled = False
                            if 'team1' in points_to_transform and len(points_to_transform['team1']) >= 6:
                                sampled = True
                                for x, y in points_to_transform['team1']:
                                    x = float(np.clip(x, 0, 105))
                                    y = float(np.clip(y, 0, 68))
                                    xi = int(x * 53 / 105)
                                    yi = int(y * 34 / 68)
                                    xi = max(0, min(xi, 52))
                                    yi = max(0, min(yi, 33))
                                    heatmap_bins_team1[xi, yi] += 1.0
                            if 'team2' in points_to_transform and len(points_to_transform['team2']) >= 6:
                                sampled = True
                                for x, y in points_to_transform['team2']:
                                    x = float(np.clip(x, 0, 105))
                                    y = float(np.clip(y, 0, 68))
                                    xi = int(x * 53 / 105)
                                    yi = int(y * 34 / 68)
                                    xi = max(0, min(xi, 52))
                                    yi = max(0, min(yi, 33))
                                    heatmap_bins_team2[xi, yi] += 1.0
                            if sampled:
                                heatmap_samples_count += 1
                            if debug_scouting and frame_count % 100 == 0:
                                print(f"Frame {frame_count}: Heatmap samples={heatmap_samples_count}")
                            
                        homography_telemetry['frame_number'].append(frame_count)
                        homography_telemetry['homography_mode'].append(homography_mode)
                        homography_telemetry['reproj_error'].append(getattr(homography_manager, 'last_reproj_error', None))
                        homography_telemetry['delta_H'].append(getattr(homography_manager, 'last_delta', None))
                        homography_telemetry['inlier_ratio'].append(getattr(homography_manager, 'last_inlier_ratio', None))
                        homography_telemetry['team1_centroid_x'].append(team1_centroid[0] if team1_centroid else None)
                        homography_telemetry['team1_centroid_y'].append(team1_centroid[1] if team1_centroid else None)
                        homography_telemetry['team2_centroid_x'].append(team2_centroid[0] if team2_centroid else None)
                        homography_telemetry['team2_centroid_y'].append(team2_centroid[1] if team2_centroid else None)
                        homography_telemetry['team1_percent_out_of_bounds'].append(team1_oob)
                        homography_telemetry['team2_percent_out_of_bounds'].append(team2_oob)

                        # Fix A: usar radar con métricas tácticas
                        current_formations = {}
                        current_metrics = {}
                        if formations_timeline['team1']:
                            current_formations['team1'] = formations_timeline['team1'][-1].get('formation', 'N/A')
                        if formations_timeline['team2']:
                            current_formations['team2'] = formations_timeline['team2'][-1].get('formation', 'N/A')
                        if team1_tracker.metrics_history['pressure_height']:
                            current_metrics['team1'] = {
                                'pressure_height': team1_tracker.metrics_history['pressure_height'][-1],
                                'offensive_width': team1_tracker.metrics_history['offensive_width'][-1],
                                'compactness': team1_tracker.metrics_history['compactness'][-1],
                            }
                        if team2_tracker.metrics_history['pressure_height']:
                            current_metrics['team2'] = {
                                'pressure_height': team2_tracker.metrics_history['pressure_height'][-1],
                                'offensive_width': team2_tracker.metrics_history['offensive_width'][-1],
                                'compactness': team2_tracker.metrics_history['compactness'][-1],
                            }

                        # Mejora H: acumular trail de pelota
                        radar_trails = {}
                        if 'ball' in points_to_transform and len(points_to_transform['ball']) > 0:
                            ball_trail.append(points_to_transform['ball'][0].copy())
                            if len(ball_trail) >= 2:
                                radar_trails['ball'] = [np.array(list(ball_trail))]

                        # Mejora I: calcular línea de offside (penúltimo defensor)
                        radar_offside = {}
                        if 'team1' in points_to_transform and len(points_to_transform['team1']) >= 2:
                            t1_x = np.sort(points_to_transform['team1'][:, 0])
                            if t1_dir == "right":
                                radar_offside['team1'] = float(t1_x[1])  # 2do más bajo X
                            else:
                                radar_offside['team1'] = float(t1_x[-2])  # 2do más alto X
                        if 'team2' in points_to_transform and len(points_to_transform['team2']) >= 2:
                            t2_x = np.sort(points_to_transform['team2'][:, 0])
                            if t2_dir == "right":
                                radar_offside['team2'] = float(t2_x[1])
                            else:
                                radar_offside['team2'] = float(t2_x[-2])

                        radar_view = draw_radar_with_metrics(
                            pitch_config, points_to_transform,
                            formations=current_formations if current_formations else None,
                            metrics=current_metrics if current_metrics else None,
                            scale=8,
                            trails=radar_trails if radar_trails else None,
                            offside_x=radar_offside if radar_offside else None,
                        )
                        radar_scale = 8
                        ox = int(pitch_config.margins * radar_scale)
                        oy = int(pitch_config.margins * radar_scale)
                        def radar_px(x_m, y_m):
                            x = int(x_m * radar_scale) + ox
                            y = int(y_m * radar_scale) + oy
                            x = max(0, min(x, radar_view.shape[1] - 1))
                            y = max(0, min(y, radar_view.shape[0] - 1))
                            return x, y
                        if team1_centroid is not None:
                            cx, cy = radar_px(team1_centroid[0], team1_centroid[1])
                            cv2.circle(radar_view, (cx, cy), 10, (0, 255, 0), -1)
                            cv2.putText(radar_view, f"T1 ({team1_centroid[0]:.1f},{team1_centroid[1]:.1f})", (cx + 8, cy - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        if team2_centroid is not None:
                            cx, cy = radar_px(team2_centroid[0], team2_centroid[1])
                            cv2.circle(radar_view, (cx, cy), 10, (255, 191, 0), -1)
                            cv2.putText(radar_view, f"T2 ({team2_centroid[0]:.1f},{team2_centroid[1]:.1f})", (cx + 8, cy - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                        # Mantener horizontal (sin rotación)
                        # Tamaño: 22% del ancho (reducido para no molestar)
                        scale_factor = 0.22
                        new_w = int(width * scale_factor)
                        aspect_ratio = radar_view.shape[0] / radar_view.shape[1]
                        new_h = int(new_w * aspect_ratio)

                        radar_resized = cv2.resize(radar_view, (new_w, new_h))

                        # Posición: esquina inferior derecha
                        margin_bottom = 20
                        margin_right = 20
                        offset_x = width - new_w - margin_right  # Esquina derecha con margen
                        offset_y = height - new_h - margin_bottom  # Abajo con margen

                        # Verificar que cabe en el frame
                        if offset_y >= 0 and offset_x >= 0:
                            # Aplicar transparencia (alpha blending)
                            alpha = 0.65  # 65% radar, 35% video
                            roi = annotated_frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w]
                            blended = cv2.addWeighted(roi, 1-alpha, radar_resized, alpha, 0)
                            annotated_frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = blended
                    except Exception as e:
                        print(f"Error dibujando radar: {e}")
                else:
                    homography_telemetry['frame_number'].append(frame_count)
                    homography_telemetry['homography_mode'].append(homography_mode)
                    homography_telemetry['reproj_error'].append(getattr(homography_manager, 'last_reproj_error', None))
                    homography_telemetry['delta_H'].append(getattr(homography_manager, 'last_delta', None))
                    homography_telemetry['inlier_ratio'].append(getattr(homography_manager, 'last_inlier_ratio', None))
                    homography_telemetry['team1_centroid_x'].append(None)
                    homography_telemetry['team1_centroid_y'].append(None)
                    homography_telemetry['team2_centroid_x'].append(None)
                    homography_telemetry['team2_centroid_y'].append(None)
                    homography_telemetry['team1_percent_out_of_bounds'].append(None)
                    homography_telemetry['team2_percent_out_of_bounds'].append(None)

                if getattr(homography_manager, 'mode', "") == "REACQUIRE":
                    homography_manager.tick_reacquire()

            current_ids = set()
            if len(tracked_persons) > 0 and tracked_persons.tracker_id is not None:
                for tid in tracked_persons.tracker_id.tolist():
                    current_ids.add(int(tid))
            for tid in list(track_age.keys()):
                if tid not in current_ids:
                    del track_age[tid]
            for tid in current_ids:
                track_age[tid] = track_age.get(tid, 0) + 1
            for tid in list(prev_field_positions.keys()):
                if tid not in current_ids:
                    del prev_field_positions[tid]

            detections_count = int(len(player_detections)) if player_detections is not None else 0
            tracks_active = int(len(track_age))
            new_tracks_count = int(len(current_ids - prev_track_ids))
            ended_tracks_count = int(len(prev_track_ids - current_ids))
            churn_ratio = float((new_tracks_count + ended_tracks_count) / max(tracks_active, 1))
            churn_ratio_list.append(churn_ratio)
            prev_track_ids = set(current_ids)
            avg_track_age = float(np.mean(list(track_age.values()))) if tracks_active > 0 else 0.0
            short_tracks = int(np.sum([1 for a in track_age.values() if a < SHORT_TRACK_AGE]))
            short_tracks_ratio = float(short_tracks / tracks_active) if tracks_active > 0 else 0.0

            reproj_error_m = getattr(homography_manager, 'last_reproj_error', None)
            delta_H = getattr(homography_manager, 'last_delta', None)
            hm = homography_mode
            hs = "ok"
            h_accept_reason = "invalid"
            reproj_is_nan = False
            if reproj_error_m is not None:
                try:
                    reproj_is_nan = bool(np.isnan(reproj_error_m))
                except Exception:
                    reproj_is_nan = False
            homography_state = getattr(homography_manager, 'last_state', "")
            invalid_state = bool(homography_state.startswith("INVALID"))
            inlier_ratio = getattr(homography_manager, 'last_inlier_ratio', None)
            if reproj_error_m is None or reproj_is_nan or (reproj_error_m is not None and reproj_error_m > REPROJ_INVALID) or invalid_state:
                hs = "invalid"
                h_accept_reason = "invalid"
            else:
                if reproj_error_m <= REPROJ_OK_MAX and (delta_H is None or delta_H <= DELTA_H_WARN):
                    hs = "ok"
                    h_accept_reason = "accepted_ok"
                else:
                    hs = "warn"
                    if reproj_error_m > REPROJ_WARN_MAX:
                        h_accept_reason = "rejected_high_reproj"
                    elif inlier_ratio is not None and inlier_ratio < 0.6:
                        h_accept_reason = "rejected_low_inliers"
                    elif using_last_good:
                        h_accept_reason = "kept_last_good"
                    else:
                        h_accept_reason = "accepted_warn"
            cut_detected = bool(getattr(homography_manager, 'cut_detected', False))

            formation_label = None
            formation_valid = True
            formation_invalid_reason = None
            if formations_timeline['team1'] or formations_timeline['team2']:
                t1_form = formations_timeline['team1'][-1] if formations_timeline['team1'] else None
                t2_form = formations_timeline['team2'][-1] if formations_timeline['team2'] else None
                t1_label = t1_form.get('formation') if isinstance(t1_form, dict) else None
                t2_label = t2_form.get('formation') if isinstance(t2_form, dict) else None
                formation_label = f"{t1_label or 'N/A'} | {t2_label or 'N/A'}"
                reasons = []
                flags = []
                for form in (t1_form, t2_form):
                    if isinstance(form, dict):
                        ppl = form.get('players_per_line')
                        total = form.get('total_players')
                        if isinstance(ppl, list) and total is not None:
                            s = sum(ppl)
                            d = ppl[0] if len(ppl) > 0 else 0
                            a = ppl[2] if len(ppl) > 2 else 0
                            valid = (s == 10) and (d >= 2) and (a <= 4)
                            flags.append(valid)
                            if not valid:
                                if s != 10:
                                    reasons.append("sum_not_10")
                                if d < 2:
                                    reasons.append("no_defenders")
                                if a > 4:
                                    reasons.append("too_many_attackers")
                if flags:
                    formation_valid = all(flags)
                if reasons:
                    formation_invalid_reason = ",".join(sorted(set(reasons)))

            frame_valid_for_metrics_strict = bool(hs == "ok" and tracks_active >= MIN_TRACKS_ACTIVE and detections_count >= MIN_DETECTIONS)
            frame_valid_for_metrics_demo = bool(hs in {"ok", "warn"} and tracks_active >= MIN_TRACKS_ACTIVE and detections_count >= MIN_DETECTIONS)
            if hs in {"ok", "warn"} and reproj_error_m is not None and reproj_error_m <= REPROJ_WARN_MAX:
                if inlier_ratio is None or inlier_ratio >= 0.6:
                    if getattr(homography_manager, 'current_H', None) is not None and not using_last_good:
                        last_good_H = homography_manager.current_H.copy()
                        last_good_age_frames = 0
                elif h_accept_reason == "rejected_low_inliers":
                    pass
            frame_valid_demo_flags.append(frame_valid_for_metrics_demo)
            frame_valid_strict_flags.append(frame_valid_for_metrics_strict)

            if max_speed_mps is not None:
                max_speed_mps_list.append(max_speed_mps)
                speed_violation_flags.append(bool(speed_violation))
            jump_violation = False
            if max_jump_m is not None:
                jump_violation = bool(max_jump_m > JUMP_MAX_M and not cut_detected)
                max_jump_m_list.append(max_jump_m)
                jump_violation_flags.append(jump_violation)

            possession_state = "unknown"
            contested_reason = None
            if possession_tracker is not None:
                if possession_result is None:
                    possession_state = "contested"
                    ball_far = False
                    if ball_detected and len(tracked_persons) > 0 and (any(team1_mask) or any(team2_mask)):
                        player_mask = np.array(team1_mask) | np.array(team2_mask)
                        player_dets = tracked_persons[player_mask]
                        if len(player_dets) > 0:
                            ball_bb = tracked_ball.xyxy[0]
                            ball_center = np.array([(ball_bb[0] + ball_bb[2]) / 2, (ball_bb[1] + ball_bb[3]) / 2])
                            player_centers = get_bottom_center(player_dets)
                            dists = np.linalg.norm(player_centers - ball_center, axis=1)
                            if len(dists) > 0:
                                min_dist = float(np.min(dists))
                                max_dist = getattr(possession_tracker, "max_distance", 70.0)
                                ball_far = min_dist > max_dist
                    if not ball_detected:
                        contested_reason = "no_ball"
                    elif ball_conf is not None and ball_conf < 0.2:
                        contested_reason = "low_ball_conf"
                    elif ball_far:
                        contested_reason = "ball_far_from_players"
                    else:
                        contested_reason = "other"
                else:
                    possession_state = possession_result[0] if isinstance(possession_result, tuple) else possession_result

            health_timeline.append({
                'frame_idx': int(frame_count),
                'homography_mode': str(hm),
                'reproj_error_m': None if reproj_error_m is None else float(reproj_error_m),
                'delta_H': None if delta_H is None else float(delta_H),
                'homography_status': hs,
                'cut_detected': cut_detected,
                'detections_count': detections_count,
                'tracks_active': tracks_active,
                'new_tracks_count': new_tracks_count,
                'ended_tracks_count': ended_tracks_count,
                'churn_ratio': churn_ratio,
                'refs_detected_count': refs_detected_count,
                'refs_filtered_out_count': refs_filtered_out_count,
                'players_detected_count': players_detected_count,
                'ball_detected': ball_detected,
                'ball_conf': None if ball_conf is None else float(ball_conf),
                'ball_track_age': ball_track_age,
                'possession_state': possession_state,
                'contested_reason': contested_reason,
                'max_player_speed_mps': None if max_speed_mps is None else float(max_speed_mps),
                'speed_violation': bool(speed_violation) if max_speed_mps is not None else None,
                'max_player_jump_m': None if max_jump_m is None else float(max_jump_m),
                'jump_violation': bool(jump_violation) if max_jump_m is not None else None,
                'avg_track_age': avg_track_age,
                'short_tracks_ratio': short_tracks_ratio,
                'frame_valid_for_metrics': frame_valid_for_metrics_demo,
                'frame_valid_for_metrics_demo': frame_valid_for_metrics_demo,
                'frame_valid_for_metrics_strict': frame_valid_for_metrics_strict,
                'ransac_inlier_ratio': getattr(homography_manager, 'last_inlier_ratio', None),
                'H_accept_reason': h_accept_reason,
                'last_good_age_frames': last_good_age_frames,
                'formation_label': formation_label,
                'formation_valid': bool(formation_valid),
                'formation_invalid_reason': formation_invalid_reason
            })
            if qc_sample_every > 0 and frame_count % qc_sample_every == 0:
                frame_qc_samples.append({
                    'frame': frame_count,
                    't': float(frame_count / fps) if fps > 0 else 0.0,
                    'homography_mode': homography_mode,
                    'reproj_error': getattr(homography_manager, 'last_reproj_error', None),
                    'delta_H': getattr(homography_manager, 'last_delta', None),
                    'spread_ok': spread_ok,
                    'n_team1': int(np.sum(team1_mask)) if len(team1_mask) > 0 else 0,
                    'n_team2': int(np.sum(team2_mask)) if len(team2_mask) > 0 else 0,
                    'centroid_team1': team1_centroid,
                    'centroid_team2': team2_centroid,
                    'block_depth_team1': metrics1['block_depth_m'] if metrics1 else None,
                    'block_width_team1': metrics1['block_width_m'] if metrics1 else None,
                    'def_line_left_team1': metrics1['def_line_left_m'] if metrics1 else None,
                    'def_line_right_team1': metrics1['def_line_right_m'] if metrics1 else None,
                    'block_depth_team2': metrics2['block_depth_m'] if metrics2 else None,
                    'block_width_team2': metrics2['block_width_m'] if metrics2 else None,
                    'def_line_left_team2': metrics2['def_line_left_m'] if metrics2 else None,
                    'def_line_right_team2': metrics2['def_line_right_m'] if metrics2 else None,
                    'out_of_bounds_team1_ratio': team1_oob,
                    'out_of_bounds_team2_ratio': team2_oob
                })

            # Fallback: posesión píxeles si no se pudo hacer métrico
            if possession_tracker is not None and pitch_config and not _possession_assigned_metric:
                possession_result = possession_tracker.assign_possession(
                    tracked_persons, team1_mask, team2_mask, tracked_ball
                )
                annotated_frame = possession_tracker.draw_possession_bar(annotated_frame)

            if debug_homography_overlay:
                state = getattr(homography_manager, 'last_state', "")
                text = None
                color = (0, 255, 0)
                if state.startswith("UPDATED"):
                    if state == "UPDATED_EMA":
                        text = "Homografia: UPDATED (EMA)"
                    elif state == "UPDATED_SWITCH":
                        text = "Homografia: UPDATED (SWITCH)"
                    else:
                        text = "Homografia: UPDATED"
                    color = (0, 255, 0)
                elif state == "REUSED_INERTIA":
                    text = f"Homografia: INERCIA ({homography_manager.frames_since_valid}/{homography_manager.max_inertia_frames})"
                    color = (0, 255, 255)
                elif state == "FULLSCREEN":
                    text = "Homografia: FULLSCREEN"
                    color = (0, 0, 255)
                else:
                    if pitch_config and transformer is None:
                        text = "Homografia: NO_H"
                        color = (0, 0, 255)
                if text is not None:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA
                    )

            out.write(annotated_frame)

        # --- GENERAR ESTADÍSTICAS FINALES ---
        if progress_callback and total_frames > 0:
            try:
                progress_callback(1.0, frame_count, total_frames)
            except Exception:
                pass
        print("Generando estadísticas tácticas...")

        def get_dominant_formation(formations_list):
            if not formations_list: return "Unknown"
            formation_names = [f.get('formation', 'Unknown') for f in formations_list]
            if not formation_names: return "Unknown"
            return Counter(formation_names).most_common(1)[0][0]

        # Calcular estadísticas agregadas de métricas
        team1_stats = team1_tracker.get_statistics()
        team2_stats = team2_tracker.get_statistics()
        hm1_smooth = cv2.GaussianBlur(heatmap_bins_team1, (5, 5), 0)
        hm2_smooth = cv2.GaussianBlur(heatmap_bins_team2, (5, 5), 0)
        if float(hm1_smooth.max()) > 0:
            hm1_norm = (hm1_smooth / hm1_smooth.max() * 255).astype(np.uint8)
        else:
            hm1_norm = np.zeros_like(hm1_smooth, dtype=np.uint8)
        if float(hm2_smooth.max()) > 0:
            hm2_norm = (hm2_smooth / hm2_smooth.max() * 255).astype(np.uint8)
        else:
            hm2_norm = np.zeros_like(hm2_smooth, dtype=np.uint8)
        hm1_color = cv2.applyColorMap(hm1_norm, cv2.COLORMAP_JET)
        hm2_color = cv2.applyColorMap(hm2_norm, cv2.COLORMAP_JET)
        if debug_scouting:
            cv2.imwrite('heatmap_team1.png', hm1_color)
            cv2.imwrite('heatmap_team2.png', hm2_color)
        h1_down = hm1_norm[:52, :].reshape(26, 2, 17, 2).max(axis=(1, 3))
        h2_down = hm2_norm[:52, :].reshape(26, 2, 17, 2).max(axis=(1, 3))
        heatmap_list_team1 = h1_down.tolist()
        heatmap_list_team2 = h2_down.tolist()
        heatmap_total_samples_team1 = float(np.sum(heatmap_bins_team1))
        heatmap_total_samples_team2 = float(np.sum(heatmap_bins_team2))
        if heatmap_total_samples_team1 > 0:
            x_coords = (np.arange(heatmap_bins_team1.shape[0]) + 0.5) * (105.0 / 53.0)
            y_coords = (np.arange(heatmap_bins_team1.shape[1]) + 0.5) * (68.0 / 34.0)
            heatmap_center_team1 = (
                float(np.sum(heatmap_bins_team1 * x_coords[:, None]) / heatmap_total_samples_team1),
                float(np.sum(heatmap_bins_team1 * y_coords[None, :]) / heatmap_total_samples_team1)
            )
        else:
            heatmap_center_team1 = None
        if heatmap_total_samples_team2 > 0:
            x_coords = (np.arange(heatmap_bins_team2.shape[0]) + 0.5) * (105.0 / 53.0)
            y_coords = (np.arange(heatmap_bins_team2.shape[1]) + 0.5) * (68.0 / 34.0)
            heatmap_center_team2 = (
                float(np.sum(heatmap_bins_team2 * x_coords[:, None]) / heatmap_total_samples_team2),
                float(np.sum(heatmap_bins_team2 * y_coords[None, :]) / heatmap_total_samples_team2)
            )
        else:
            heatmap_center_team2 = None

        def _safe_mean(values):
            vals = [v for v in values if v is not None]
            if len(vals) == 0:
                return None
            return float(np.mean(vals))

        def _mean_tuple(xs, ys):
            mx = _safe_mean(xs)
            my = _safe_mean(ys)
            if mx is None or my is None:
                return None
            return (mx, my)

        def _abs_deltas(series):
            cleaned = [v for v in series if v is not None]
            if len(cleaned) < 2:
                return []
            arr = np.array(cleaned, dtype=np.float32)
            return np.abs(np.diff(arr)).tolist()

        def _avg_and_p95(deltas):
            if len(deltas) == 0:
                return None, None
            arr = np.array(deltas, dtype=np.float32)
            return float(np.mean(arr)), float(np.percentile(arr, 95))

        hm_modes = homography_telemetry['homography_mode']
        homography_tracking_frames = int(np.sum([m == 'tracking' for m in hm_modes]))
        homography_inertia_frames = int(np.sum([m == 'inertia' for m in hm_modes]))
        homography_fallback_frames = int(np.sum([m == 'fallback' for m in hm_modes]))
        homography_valid_frames = homography_tracking_frames + homography_inertia_frames
        homography_valid_ratio = float(homography_valid_frames / frame_count) if frame_count > 0 else 0.0
        avg_reproj_error = _safe_mean(homography_telemetry['reproj_error'])
        avg_delta_H = _safe_mean(homography_telemetry['delta_H'])
        team1_out_of_bounds_ratio = _safe_mean(homography_telemetry['team1_percent_out_of_bounds'])
        team2_out_of_bounds_ratio = _safe_mean(homography_telemetry['team2_percent_out_of_bounds'])
        avg_centroid_team1 = _mean_tuple(homography_telemetry['team1_centroid_x'], homography_telemetry['team1_centroid_y'])
        avg_centroid_team2 = _mean_tuple(homography_telemetry['team2_centroid_x'], homography_telemetry['team2_centroid_y'])

        t1_hist = team1_tracker.metrics_history
        t2_hist = team2_tracker.metrics_history
        t1_bd_avg, t1_bd_p95 = _avg_and_p95(_abs_deltas(list(t1_hist['block_depth_m'])))
        t2_bd_avg, t2_bd_p95 = _avg_and_p95(_abs_deltas(list(t2_hist['block_depth_m'])))
        t1_bw_avg, t1_bw_p95 = _avg_and_p95(_abs_deltas(list(t1_hist['block_width_m'])))
        t2_bw_avg, t2_bw_p95 = _avg_and_p95(_abs_deltas(list(t2_hist['block_width_m'])))
        t1_dl_avg, t1_dl_p95 = _avg_and_p95(_abs_deltas(list(t1_hist['def_line_left_m'])))
        t2_dl_avg, t2_dl_p95 = _avg_and_p95(_abs_deltas(list(t2_hist['def_line_left_m'])))
        t1_dr_avg, t1_dr_p95 = _avg_and_p95(_abs_deltas(list(t1_hist['def_line_right_m'])))
        t2_dr_avg, t2_dr_p95 = _avg_and_p95(_abs_deltas(list(t2_hist['def_line_right_m'])))

        def _grade(valid_ratio, homography_ratio):
            score = min(valid_ratio, homography_ratio)
            if score >= 0.7:
                return "High"
            if score >= 0.4:
                return "Medium"
            return "Low"

        valid_frames_demo = int(np.sum([1 for v in frame_valid_demo_flags if v]))
        valid_frames_strict = int(np.sum([1 for v in frame_valid_strict_flags if v]))
        team1_valid_ratio = float(valid_frames_demo / frame_count) if frame_count > 0 else 0.0
        team2_valid_ratio = float(valid_frames_demo / frame_count) if frame_count > 0 else 0.0
        confidence_grade_team1 = _grade(team1_valid_ratio, homography_valid_ratio)
        confidence_grade_team2 = _grade(team2_valid_ratio, homography_valid_ratio)

        warnings = []
        fallback_ratio = float(homography_fallback_frames / frame_count) if frame_count > 0 else 0.0
        if fallback_ratio > 0.4:
            warnings.append("High fallback ratio")
        if avg_centroid_team1 is None or not (0 <= avg_centroid_team1[0] <= 105 and 0 <= avg_centroid_team1[1] <= 68):
            warnings.append("Centroid out of expected range")
        if avg_centroid_team2 is None or not (0 <= avg_centroid_team2[0] <= 105 and 0 <= avg_centroid_team2[1] <= 68):
            warnings.append("Centroid out of expected range")
        jitter_flags = [
            (t1_bd_p95 or 0) > 8, (t2_bd_p95 or 0) > 8,
            (t1_bw_p95 or 0) > 6, (t2_bw_p95 or 0) > 6,
            (t1_dl_p95 or 0) > 6, (t2_dl_p95 or 0) > 6,
            (t1_dr_p95 or 0) > 6, (t2_dr_p95 or 0) > 6,
        ]
        if any(jitter_flags):
            warnings.append("Excessive jitter")

        def _safe_p95(values):
            vals = [v for v in values if v is not None]
            if len(vals) == 0:
                return None
            arr = np.array(vals, dtype=np.float32)
            return float(np.percentile(arr, 95))

        invalid_frames = int(np.sum([1 for h in health_timeline if h.get('homography_status') == 'invalid']))
        warn_frames = int(np.sum([1 for h in health_timeline if h.get('homography_status') == 'warn']))
        fallback_frames = int(np.sum([1 for h in health_timeline if h.get('homography_status') == 'fallback']))
        invalid_formation_frames = int(np.sum([1 for h in health_timeline if not h.get('formation_valid')]))
        avg_tracks_active = _safe_mean([h.get('tracks_active') for h in health_timeline])
        avg_short_tracks_ratio = _safe_mean([h.get('short_tracks_ratio') for h in health_timeline])
        avg_churn_ratio = _safe_mean(churn_ratio_list)
        p95_churn_ratio = _safe_p95(churn_ratio_list)
        churn_warn_ratio = None
        if len(churn_ratio_list) > 0:
            churn_warn_ratio = float(np.sum([c > CHURN_WARN for c in churn_ratio_list]) / len(churn_ratio_list))
        avg_max_speed_mps = _safe_mean(max_speed_mps_list)
        p95_max_speed_mps = _safe_p95(max_speed_mps_list)
        speed_violation_ratio = None
        if len(speed_violation_flags) > 0:
            speed_violation_ratio = float(np.sum(speed_violation_flags) / len(speed_violation_flags))
        avg_max_jump_m = _safe_mean(max_jump_m_list)
        p95_max_jump_m = _safe_p95(max_jump_m_list)
        jump_violation_ratio = None
        if len(jump_violation_flags) > 0:
            jump_violation_ratio = float(np.sum(jump_violation_flags) / len(jump_violation_flags))
        p95_reproj_error_m = _safe_p95(homography_telemetry['reproj_error'])
        p95_delta_H = _safe_p95(homography_telemetry['delta_H'])
        invalid_ratio = float(1.0 - (valid_frames_demo / frame_count)) if frame_count > 0 else 0.0
        strict_invalid_ratio = float(1.0 - (valid_frames_strict / frame_count)) if frame_count > 0 else 0.0
        warn_ratio = float(warn_frames / frame_count) if frame_count > 0 else 0.0
        fallback_ratio_health = float(fallback_frames / frame_count) if frame_count > 0 else 0.0
        invalid_formation_ratio = float(invalid_formation_frames / frame_count) if frame_count > 0 else 0.0
        team1_stats['valid_frames'] = valid_frames_demo
        team2_stats['valid_frames'] = valid_frames_demo
        stats_data = {
            'total_frames': frame_count,
            'duration_seconds': frame_count / fps if fps > 0 else 0,
            'fps': fps,
            'formations': {
                'team1': {
                    'most_common': get_dominant_formation(formations_timeline['team1']),
                    'timeline': [f.get('formation', 'Unknown') for f in formations_timeline['team1']]
                },
                'team2': {
                    'most_common': get_dominant_formation(formations_timeline['team2']),
                    'timeline': [f.get('formation', 'Unknown') for f in formations_timeline['team2']]
                }
            },
            'metrics': {
                'team1': team1_stats,
                'team2': team2_stats
            },
            'timeline': {
                'team1': team1_tracker.export_to_dict(),
                'team2': team2_tracker.export_to_dict(),
                'health': health_timeline
            },
            'scouting_heatmaps': {
                'team1': heatmap_list_team1,
                'team2': heatmap_list_team2,
                'bins_shape': [26, 17],
                'field_dims_m': [105, 68],
                'bin_size_m': [105 / 26.0, 68 / 17.0],
                'total_samples': heatmap_samples_count,
                'sample_rate': heatmap_sample_every
            },
            'homography_telemetry': homography_telemetry,
            'homography_settings': {
                'max_inertia_frames': homography_manager.max_inertia_frames,
                'disable_inertia': bool(disable_inertia)
            },
            'quality_control': {
                'total_frames': frame_count,
                'duration_seconds': frame_count / fps if fps > 0 else 0,
                'fps': fps,
                'homography_valid_frames': homography_valid_frames,
                'homography_inertia_frames': homography_inertia_frames,
                'homography_fallback_frames': homography_fallback_frames,
                'homography_valid_ratio': homography_valid_ratio,
                'avg_reproj_error': avg_reproj_error,
                'avg_delta_H': avg_delta_H,
                'team1_out_of_bounds_ratio': team1_out_of_bounds_ratio,
                'team2_out_of_bounds_ratio': team2_out_of_bounds_ratio,
                'avg_centroid_team1': avg_centroid_team1,
                'avg_centroid_team2': avg_centroid_team2,
                'avg_abs_delta_block_depth_team1': t1_bd_avg,
                'avg_abs_delta_block_depth_team2': t2_bd_avg,
                'p95_abs_delta_block_depth_team1': t1_bd_p95,
                'p95_abs_delta_block_depth_team2': t2_bd_p95,
                'avg_abs_delta_block_width_team1': t1_bw_avg,
                'avg_abs_delta_block_width_team2': t2_bw_avg,
                'p95_abs_delta_block_width_team1': t1_bw_p95,
                'p95_abs_delta_block_width_team2': t2_bw_p95,
                'avg_abs_delta_def_line_left_team1': t1_dl_avg,
                'avg_abs_delta_def_line_left_team2': t2_dl_avg,
                'p95_abs_delta_def_line_left_team1': t1_dl_p95,
                'p95_abs_delta_def_line_left_team2': t2_dl_p95,
                'avg_abs_delta_def_line_right_team1': t1_dr_avg,
                'avg_abs_delta_def_line_right_team2': t2_dr_avg,
                'p95_abs_delta_def_line_right_team1': t1_dr_p95,
                'p95_abs_delta_def_line_right_team2': t2_dr_p95,
                'heatmap_center_of_mass_team1': heatmap_center_team1,
                'heatmap_center_of_mass_team2': heatmap_center_team2,
                'heatmap_total_samples_team1': heatmap_total_samples_team1,
                'heatmap_total_samples_team2': heatmap_total_samples_team2,
                'confidence_grade_team1': confidence_grade_team1,
                'confidence_grade_team2': confidence_grade_team2,
                'warnings': warnings
            },
            'health_summary': {
                'total_frames': frame_count,
                'valid_frames': valid_frames_demo,
                'valid_frames_strict': valid_frames_strict,
                'invalid_ratio': invalid_ratio,
                'strict_invalid_ratio': strict_invalid_ratio,
                'fallback_ratio': fallback_ratio_health,
                'warn_ratio': warn_ratio,
                'avg_reproj_error_m': avg_reproj_error,
                'p95_reproj_error_m': p95_reproj_error_m,
                'avg_delta_H': avg_delta_H,
                'p95_delta_H': p95_delta_H,
                'avg_tracks_active': avg_tracks_active,
                'avg_short_tracks_ratio': avg_short_tracks_ratio,
                'invalid_formation_ratio': invalid_formation_ratio,
                'avg_churn_ratio': avg_churn_ratio,
                'p95_churn_ratio': p95_churn_ratio,
                'churn_warn_ratio': churn_warn_ratio,
                'avg_max_speed_mps': avg_max_speed_mps,
                'p95_max_speed_mps': p95_max_speed_mps,
                'speed_violation_ratio': speed_violation_ratio,
                'avg_max_jump_m': avg_max_jump_m,
                'p95_max_jump_m': p95_max_jump_m,
                'jump_violation_ratio': jump_violation_ratio
            },
            'frame_qc_samples': frame_qc_samples
        }

        # --- ESTADÍSTICAS DE POSESIÓN ---
        if possession_tracker is not None:
            stats_data['possession'] = possession_tracker.get_stats_dict()
        if speed_estimator is not None:
            stats_data['speed_distance'] = speed_estimator.get_summary_stats()

        stats_path = Path(target_path).parent / f"{Path(target_path).stem}_stats.json"
        with open(stats_path, 'w') as f:
            # Convertir tipos de NumPy a tipos nativos de Python
            stats_data_converted = convert_to_native_types(stats_data)
            json.dump(stats_data_converted, f, indent=2)
        print(f"Estadísticas guardadas en: {stats_path}")

    finally:
        cap.release()
        out.release()
