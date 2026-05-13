import { API_BASE_URL, ApiError } from './apiClient'
import type {
  ComputerVisionConfig,
  ComputerVisionJob,
  ComputerVisionResult,
  VideoSourceInput,
} from '../types/computerVision'

function buildMockResult(source: VideoSourceInput, config: ComputerVisionConfig): ComputerVisionResult {
  const videoName =
    source.source_mode === 'upload'
      ? source.file?.name ?? 'clip-demo.mp4'
      : source.soccernet_path?.split(/[\\/]/).pop() ?? 'soccernet-demo.mp4'

  return {
    source: 'mock',
    status_message: 'No se pudo conectar con el backend de Computer Vision; se muestra una salida demo estable.',
    video_name: videoName,
    duration_seconds: config.segment_mode ? config.duration_seconds : 18.4,
    total_frames: config.segment_mode ? Math.round(config.duration_seconds * 25) : 460,
    fps: 25,
    health_summary: {
      demo_mode: 'degraded',
      fallback_ratio: 0.18,
      p95_reproj_error_m: 1.4,
      p95_churn_ratio: 0.32,
      p95_max_speed_mps: 8.4,
      ball_detected_ratio: 0.42,
    },
    formations: {
      team1: { most_common: '4-3-3' },
      team2: { most_common: '4-4-2' },
    },
    metrics: {
      team1: {
        pressure_height: { mean: 42.1, min: 35.2, max: 51.8 },
        offensive_width: { mean: 34.6, min: 28.4, max: 42.2 },
        compactness: { mean: 812, min: 744, max: 930 },
        block_depth_m: { mean: 37.4, min: 30.1, max: 44.8 },
        block_width_m: { mean: 33.8, min: 27.3, max: 40.4 },
        def_line_left_m: { mean: 63.2, min: 54.7, max: 70.4 },
        def_line_right_m: { mean: 65.7, min: 58.9, max: 72.3 },
        valid_frames: 332,
      },
      team2: {
        pressure_height: { mean: 35.8, min: 29.4, max: 44.1 },
        offensive_width: { mean: 29.1, min: 22.5, max: 36.8 },
        compactness: { mean: 748, min: 681, max: 840 },
        block_depth_m: { mean: 31.8, min: 27.6, max: 38.9 },
        block_width_m: { mean: 29.4, min: 24.3, max: 35.8 },
        def_line_left_m: { mean: 54.8, min: 49.2, max: 61.7 },
        def_line_right_m: { mean: 57.4, min: 51.3, max: 62.8 },
        valid_frames: 320,
      },
    },
    timeline: {
      pressure_height: {
        frames: [0, 60, 120, 180, 240, 300, 360, 420],
        team1: [38, 40, 43, 45, 41, 44, 46, 42],
        team2: [30, 31, 34, 36, 35, 37, 39, 36],
      },
      compactness: {
        frames: [0, 60, 120, 180, 240, 300, 360, 420],
        team1: [790, 810, 820, 835, 800, 815, 830, 808],
        team2: [710, 725, 742, 760, 748, 771, 755, 740],
      },
      offensive_width: {
        frames: [0, 60, 120, 180, 240, 300, 360, 420],
        team1: [31, 33, 35, 36, 34, 35, 37, 34],
        team2: [26, 27, 29, 31, 30, 31, 32, 29],
      },
    },
    possession: {
      team1_pct: 54,
      team2_pct: 39,
      contested_pct: 7,
    },
    scouting: {
      confidence: {
        team1: { label: 'Alta', score: 84 },
        team2: { label: 'Media', score: 68 },
      },
      bullets: {
        team1: [
          'Bloque medio-alto con tendencia a recuperar adelantado.',
          'Buena amplitud ofensiva para atacar por fuera.',
          'La línea defensiva se sostiene por encima del bloque medio.',
        ],
        team2: [
          'Bloque medio con menor altura de presión.',
          'Compactación razonable, pero menos agresiva tras pérdida.',
          'Fase ofensiva más contenida y menos ancha.',
        ],
      },
    },
    exports: {
      json: true,
      csv: true,
      pdf: false,
    },
    warnings: [
      'Demo degradado: las métricas son aproximadas si la homografía no es estable.',
      config.enable_possession
        ? 'La posesión depende de señal de balón consistente.'
        : 'La posesión está desactivada para este procesamiento.',
    ],
    interpretation: [
      'La lectura táctica debe empezar por homografía y tracking antes de interpretar métricas avanzadas.',
      'Si el clip presenta zoom extremo o pocas líneas de campo, la confiabilidad baja.',
      'Usá los gráficos temporales para detectar cambios de estructura y no sólo promedios.',
    ],
  }
}

export async function analyzeComputerVisionVideo(params: {
  source: VideoSourceInput
  config: ComputerVisionConfig
}): Promise<ComputerVisionResult> {
  const formData = new FormData()
  formData.append('source_mode', params.source.source_mode)
  formData.append('config', JSON.stringify(params.config))

  if (params.source.file) {
    formData.append('file', params.source.file)
  }

  if (params.source.soccernet_path) {
    formData.append('soccernet_path', params.source.soccernet_path)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/computer-vision/analyze`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const message = await response.text()
      throw new ApiError(message || 'No se pudo procesar el video.', response.status)
    }

    return (await response.json()) as ComputerVisionResult
  } catch {
    return buildMockResult(params.source, params.config)
  }
}

function buildFormData(params: { source: VideoSourceInput; config: ComputerVisionConfig }) {
  const formData = new FormData()
  formData.append('source_mode', params.source.source_mode)
  formData.append('config', JSON.stringify(params.config))

  if (params.source.file) {
    formData.append('file', params.source.file)
  }

  if (params.source.soccernet_path) {
    formData.append('soccernet_path', params.source.soccernet_path)
  }

  return formData
}

export async function createComputerVisionJob(params: {
  source: VideoSourceInput
  config: ComputerVisionConfig
}): Promise<ComputerVisionJob> {
  const response = await fetch(`${API_BASE_URL}/api/v1/computer-vision/jobs`, {
    method: 'POST',
    body: buildFormData(params),
  })

  if (!response.ok) {
    const message = await response.text()
    throw new ApiError(message || 'No se pudo crear el job.', response.status)
  }

  return (await response.json()) as ComputerVisionJob
}

export async function getComputerVisionJob(jobId: string): Promise<ComputerVisionJob> {
  const response = await fetch(`${API_BASE_URL}/api/v1/computer-vision/jobs/${jobId}`)

  if (!response.ok) {
    const message = await response.text()
    throw new ApiError(message || 'No se pudo consultar el job.', response.status)
  }

  return (await response.json()) as ComputerVisionJob
}
