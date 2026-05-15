export type VideoSourceMode = 'upload' | 'soccernet'

export interface ComputerVisionConfig {
  model_name: 'yolov8n.pt' | 'yolov8s.pt' | 'yolov8m.pt' | 'yolov8l.pt' | 'yolov8x.pt'
  player_model_source: 'builtin' | 'custom'
  ball_model_source: 'heuristic' | 'custom'
  pitch_source: 'homography' | 'soccana' | 'full_field_approx'
  confidence: number
  image_size: 640 | 720 | 960
  only_person: boolean
  segment_mode: boolean
  start_seconds: number
  duration_seconds: number
  full_field_approx: boolean
  enable_radar: boolean
  enable_analytics: boolean
  enable_possession: boolean
  disable_inertia: boolean
  export_profile: 'summary' | 'debug_sampled' | 'full'
  sample_stride: number
  topk_frames: number
  enable_compression: boolean
}

export interface ComputerVisionModelAssets {
  playerModelFile?: File
  ballModelFile?: File
}

export interface VideoSourceInput {
  source_mode: VideoSourceMode
  soccernet_path?: string
  file?: File
}

export interface TeamMetricRange {
  mean: number | null
  min?: number | null
  max?: number | null
}

export interface TeamFormationSummary {
  most_common: string
}

export interface TeamTacticalMetrics {
  pressure_height?: TeamMetricRange
  offensive_width?: TeamMetricRange
  compactness?: TeamMetricRange
  block_depth_m?: TeamMetricRange
  block_width_m?: TeamMetricRange
  def_line_left_m?: TeamMetricRange
  def_line_right_m?: TeamMetricRange
  valid_frames?: number
}

export interface PipelineHealthSummary {
  demo_mode?: 'stable' | 'degraded'
  total_frames?: number
  valid_frames?: number
  valid_frames_strict?: number
  invalid_ratio?: number
  strict_invalid_ratio?: number
  fallback_ratio?: number
  warn_ratio?: number
  avg_reproj_error_m?: number
  p95_reproj_error_m?: number
  avg_delta_H?: number
  p95_delta_H?: number
  avg_churn_ratio?: number
  p95_churn_ratio?: number
  churn_warn_ratio?: number
  avg_max_speed_mps?: number
  p95_max_speed_mps?: number
  avg_max_speed_mps_strict?: number
  p95_max_speed_mps_strict?: number
  speed_violation_ratio?: number
  speed_violation_ratio_strict?: number
  ball_detected_ratio?: number
  ball_track_age_p95?: number
}

export interface TimelineSeries {
  frames: number[]
  team1: number[]
  team2: number[]
}

export interface ComputerVisionTimeline {
  pressure_height?: TimelineSeries
  compactness?: TimelineSeries
  offensive_width?: TimelineSeries
}

export interface ScoutingSummary {
  confidence: {
    team1: { label: string; score: number }
    team2: { label: string; score: number }
  }
  bullets: {
    team1: string[]
    team2: string[]
  }
}

export interface PossessionSummary {
  team1_pct: number
  team2_pct: number
  contested_pct: number
  unknown_pct?: number
  total_frames_analyzed?: number
  top_possessors?: Array<{
    tracker_id: number
    frames: number
    team: string
  }>
  passes?: {
    team1_passes?: number
    team2_passes?: number
    turnovers?: number
    total?: number
  }
  timeline?: {
    frames?: number[]
    state?: string[]
  }
  reason?: string | null
}

export interface SpeedDistanceSummary {
  per_player?: Record<
    string,
    {
      distance_m: number
      max_speed_kmh: number
      team: string
      sprint_count: number
      sprint_distance_m: number
      intensity_zones_m?: Record<string, number>
    }
  >
  per_team?: Record<
    string,
    {
      total_distance_m: number
      avg_distance_m: number
      max_speed_kmh: number
      player_count: number
      total_sprints: number
      total_sprint_distance_m: number
    }
  >
}

export interface ScoutingHeatmapSummary {
  total_samples?: number
  downsampled_shape?: number[]
}

export interface ComputerVisionArtifacts {
  video_url?: string | null
  stats_json_url?: string | null
  pdf_url?: string | null
}

export interface ComputerVisionResult {
  source: 'api' | 'mock'
  status_message?: string
  video_name: string
  duration_seconds: number
  total_frames: number
  fps: number
  health_summary: PipelineHealthSummary
  formations: {
    team1: TeamFormationSummary
    team2: TeamFormationSummary
  }
  metrics: {
    team1: TeamTacticalMetrics
    team2: TeamTacticalMetrics
  }
  timeline: ComputerVisionTimeline
  possession?: PossessionSummary
  scouting: ScoutingSummary
  exports: {
    json: boolean
    csv: boolean
    pdf: boolean
  }
  quality_control?: Record<string, unknown>
  speed_distance?: SpeedDistanceSummary
  scouting_heatmaps?: {
    team1?: ScoutingHeatmapSummary
    team2?: ScoutingHeatmapSummary
    bins_shape?: number[]
  }
  homography_telemetry?: Record<string, unknown>
  artifacts?: ComputerVisionArtifacts
  warnings: string[]
  interpretation: string[]
}

export interface ComputerVisionJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  video_name: string
  processing_id?: string | null
  result?: ComputerVisionResult | null
  error?: string | null
}

export interface ComputerVisionHistoryItem {
  processing_id: string
  job_id?: string | null
  source_mode: VideoSourceMode
  source_label: string
  video_name: string
  status: string
  created_at: string
  updated_at: string
  video_url?: string | null
  stats_json_url?: string | null
}

export interface ComputerVisionHistoryResponse {
  items: ComputerVisionHistoryItem[]
}

export interface ComputerVisionHistoryDetail {
  metadata: ComputerVisionHistoryItem
  result: ComputerVisionResult
}

export interface DeleteComputerVisionHistoryResponse {
  ok: boolean
  message: string
}
