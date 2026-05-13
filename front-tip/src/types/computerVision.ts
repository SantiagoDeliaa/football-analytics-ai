export type VideoSourceMode = 'upload' | 'soccernet'

export interface ComputerVisionConfig {
  model_name: 'yolov8n.pt' | 'yolov8s.pt'
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
  warnings: string[]
  interpretation: string[]
}

export interface ComputerVisionJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  video_name: string
  result?: ComputerVisionResult | null
  error?: string | null
}
