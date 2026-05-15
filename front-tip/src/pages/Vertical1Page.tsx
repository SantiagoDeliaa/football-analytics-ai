import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import { z } from 'zod'
import { loadTimelineCharts, scheduleModulePrefetch } from '../app/modulePreload'
import { ErrorState } from '../components/common/ErrorState'
import { LoadingState } from '../components/common/LoadingState'
import { Tabs } from '../components/common/Tabs'
import { ComputerVisionHistoryPanel } from '../components/vertical1/ComputerVisionHistoryPanel'
import { ExportsPanel } from '../components/vertical1/ExportsPanel'
import { PipelineHealthPanel } from '../components/vertical1/PipelineHealthPanel'
import { ProcessingConfigPanel } from '../components/vertical1/ProcessingConfigPanel'
import {
  InterpretationPanel,
  PossessionPanel,
  ScoutingPanels,
} from '../components/vertical1/ScoutingPanels'
import { TeamComparisonTable } from '../components/vertical1/TeamComparisonTable'
import { VideoSourcePanel } from '../components/vertical1/VideoSourcePanel'
import { useAsync } from '../hooks/useAsync'
import {
  createComputerVisionJob,
  deleteComputerVisionHistoryItem,
  getComputerVisionHistory,
  getComputerVisionHistoryItem,
  getComputerVisionJob,
} from '../services/computerVisionApi'
import type {
  ComputerVisionConfig,
  ComputerVisionHistoryItem,
  ComputerVisionJob,
  ComputerVisionModelAssets,
  ComputerVisionResult,
  VideoSourceInput,
} from '../types/computerVision'

const TimelineCharts = lazy(async () => {
  const module = await loadTimelineCharts()
  return { default: module.TimelineCharts }
})

type Vertical1Tab =
  | 'video'
  | 'stats'
  | 'charts'
  | 'exports'
  | 'scouting'
  | 'interpretation'
  | 'possession'

const initialSource: VideoSourceInput = {
  source_mode: 'upload',
}

const initialConfig: ComputerVisionConfig = {
  model_name: 'yolov8n.pt',
  player_model_source: 'builtin',
  ball_model_source: 'heuristic',
  pitch_source: 'homography',
  confidence: 0.25,
  image_size: 640,
  only_person: true,
  segment_mode: false,
  start_seconds: 0,
  duration_seconds: 10,
  full_field_approx: false,
  enable_radar: true,
  enable_analytics: true,
  enable_possession: true,
  disable_inertia: false,
  export_profile: 'debug_sampled',
  sample_stride: 10,
  topk_frames: 20,
  enable_compression: true,
}

const uploadSourceSchema = z.object({
  source_mode: z.literal('upload'),
  file: z.instanceof(File, { message: 'Debés seleccionar un archivo de video.' }),
})

const soccernetSourceSchema = z.object({
  source_mode: z.literal('soccernet'),
  soccernet_path: z.string().min(3, 'Debés ingresar la ruta del video en SoccerNet/local.'),
})

export function Vertical1Page() {
  const [activeTab, setActiveTab] = useState<Vertical1Tab>('video')
  const [source, setSource] = useState<VideoSourceInput>(initialSource)
  const [config, setConfig] = useState<ComputerVisionConfig>(initialConfig)
  const [assets, setAssets] = useState<ComputerVisionModelAssets>({})
  const [validationError, setValidationError] = useState<string | undefined>(undefined)
  const [job, setJob] = useState<ComputerVisionJob | undefined>(undefined)
  const [result, setResult] = useState<ComputerVisionResult | undefined>(undefined)
  const [jobError, setJobError] = useState<string | undefined>(undefined)
  const [historyEntries, setHistoryEntries] = useState<ComputerVisionHistoryItem[]>([])
  const [historyFeedback, setHistoryFeedback] = useState<string | undefined>(undefined)
  const [activeProcessingId, setActiveProcessingId] = useState<string | undefined>(undefined)
  const [loadingHistoryProcessingId, setLoadingHistoryProcessingId] = useState<string | undefined>(undefined)
  const [deletingHistoryProcessingId, setDeletingHistoryProcessingId] = useState<string | undefined>(undefined)
  const createJobTask = useAsync<ComputerVisionJob>()
  const historyTask = useAsync<{ items: ComputerVisionHistoryItem[] }>()
  const historyDetailTask = useAsync<{ metadata: ComputerVisionHistoryItem; result: ComputerVisionResult }>()
  const deleteHistoryTask = useAsync<{ ok: boolean; message: string }>()
  const busy = createJobTask.loading || job?.status === 'queued' || job?.status === 'running'

  const previewUrl = useMemo(() => {
    if (!source.file) {
      return undefined
    }

    return URL.createObjectURL(source.file)
  }, [source.file])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  useEffect(() => {
    if (!result) {
      return
    }

    scheduleModulePrefetch(loadTimelineCharts)
  }, [result])

  useEffect(() => {
    void refreshHistory()
  }, [])

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) {
      return
    }

    const activeJob = job
    let cancelled = false

    async function pollJob() {
      try {
        const nextJob = await getComputerVisionJob(activeJob.job_id)
        if (cancelled) {
          return
        }

        setJob(nextJob)

        if (nextJob.status === 'completed') {
          if (nextJob.result) {
            setResult(nextJob.result)
            setJobError(undefined)
            setActiveProcessingId(nextJob.processing_id ?? undefined)
            setHistoryFeedback(
              nextJob.processing_id ? 'El procesamiento quedó guardado en el historial local.' : undefined,
            )
            setActiveTab('video')
            void refreshHistory()
            return
          }

          setJobError('El job terminó sin resultado.')
        }

        if (nextJob.status === 'failed') {
          setJobError(nextJob.error ?? 'El procesamiento del job falló.')
        }
      } catch (error) {
        if (!cancelled) {
          setJobError(error instanceof Error ? error.message : 'No se pudo consultar el job.')
        }
      }
    }

    void pollJob()
    const intervalId = window.setInterval(() => {
      void pollJob()
    }, 2000)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [job])

  function validateSource() {
    if (source.source_mode === 'upload') {
      const parsed = uploadSourceSchema.safeParse(source)
      return parsed.success ? undefined : parsed.error.issues[0]?.message
    }

    const parsed = soccernetSourceSchema.safeParse(source)
    return parsed.success ? undefined : parsed.error.issues[0]?.message
  }

  function validateAdvancedAssets() {
    if (config.player_model_source === 'custom' && !assets.playerModelFile) {
      return 'Debés subir un modelo custom de jugadores (.pt).'
    }
    if (config.ball_model_source === 'custom' && !assets.ballModelFile) {
      return 'Debés subir un modelo custom de pelota (.pt).'
    }
    return undefined
  }

  async function refreshHistory() {
    const history = await historyTask.run(() => getComputerVisionHistory())
    if (history) {
      setHistoryEntries(history.items)
    }
  }

  async function handleSubmit() {
    const nextError = validateSource()
    if (nextError) {
      setValidationError(nextError)
      return
    }

    const advancedError = validateAdvancedAssets()
    if (advancedError) {
      setValidationError(advancedError)
      return
    }

    setValidationError(undefined)
    setJobError(undefined)
    setResult(undefined)
    setActiveProcessingId(undefined)
    setHistoryFeedback(undefined)

    const createdJob = await createJobTask.run(() =>
      createComputerVisionJob({
        source,
        config,
        assets,
      }),
    )

    if (createdJob) {
      setJob(createdJob)
    }
  }

  async function handleLoadHistory(processingId: string) {
    setLoadingHistoryProcessingId(processingId)
    const detail = await historyDetailTask.run(() => getComputerVisionHistoryItem(processingId))
    setLoadingHistoryProcessingId(undefined)
    if (!detail) {
      return
    }

    setResult(detail.result)
    setJob(undefined)
    setJobError(undefined)
    setValidationError(undefined)
    setActiveProcessingId(processingId)
    setActiveTab('video')
    setHistoryFeedback(`Se cargó el procesamiento guardado para ${detail.metadata.video_name}.`)
    setSource(
      detail.metadata.source_mode === 'soccernet'
        ? { source_mode: 'soccernet', soccernet_path: detail.metadata.source_label }
        : { source_mode: 'upload' },
    )
  }

  async function handleDeleteHistory(processingId: string) {
    setDeletingHistoryProcessingId(processingId)
    const response = await deleteHistoryTask.run(() => deleteComputerVisionHistoryItem(processingId))
    setDeletingHistoryProcessingId(undefined)
    if (!response) {
      return
    }

    if (activeProcessingId === processingId) {
      setActiveProcessingId(undefined)
    }

    setHistoryFeedback(response.message)
    await refreshHistory()
  }

  const historyError = historyTask.error ?? historyDetailTask.error ?? deleteHistoryTask.error

  return (
    <section className="space-y-6">
      <header className="relative overflow-hidden rounded-[28px] border border-sky-500/20 bg-slate-950/70 p-6 shadow-2xl md:p-8">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.12),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(29,215,144,0.08),_transparent_30%)]" />
        <div className="relative">
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-300">
            Tactical Intelligence Platform
          </p>
          <h1 className="mt-3 text-3xl font-extrabold tracking-tight text-slate-100 md:text-4xl">
            Computer Vision
          </h1>
          <p className="mt-3 max-w-3xl text-sm text-slate-300 md:text-base">
            Tracking y métricas tácticas desde video broadcast, con foco en salud del pipeline, lectura
            temporal y exportes reutilizables.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_1.2fr]">
        <VideoSourcePanel
          loading={busy}
          onFileChange={(file) => setSource({ source_mode: 'upload', file })}
          onModeChange={(mode) =>
            setSource(mode === 'upload' ? { source_mode: mode } : { source_mode: mode, soccernet_path: '' })
          }
          onSoccernetPathChange={(path) => setSource({ source_mode: 'soccernet', soccernet_path: path })}
          source={source}
        />
        <ProcessingConfigPanel
          assets={assets}
          config={config}
          loading={busy}
          onAssetsChange={setAssets}
          onChange={setConfig}
          onSubmit={handleSubmit}
        />
      </div>

      <ComputerVisionHistoryPanel
        activeProcessingId={activeProcessingId}
        deletingProcessingId={deletingHistoryProcessingId}
        entries={historyEntries}
        feedbackMessage={historyFeedback}
        loading={historyTask.loading}
        loadingProcessingId={loadingHistoryProcessingId}
        onDelete={handleDeleteHistory}
        onLoad={handleLoadHistory}
      />

      {validationError ? <ErrorState message={validationError} /> : null}
      {busy ? (
        <LoadingState
          label={
            job?.status === 'running'
              ? `Procesando video en background${job.video_name ? `: ${job.video_name}` : ''}...`
              : 'Creando job de procesamiento...'
          }
        />
      ) : null}
      {createJobTask.error ? <ErrorState message={createJobTask.error} onRetry={handleSubmit} /> : null}
      {jobError ? <ErrorState message={jobError} onRetry={handleSubmit} /> : null}
      {historyError ? <ErrorState message={historyError} onRetry={refreshHistory} /> : null}

      {result ? (
        <>
          <SessionOverview result={result} />

          <Tabs
            onChange={(value: string) => setActiveTab(value as Vertical1Tab)}
            options={[
              { value: 'video', label: 'Video' },
              { value: 'stats', label: 'Estadísticas' },
              { value: 'charts', label: 'Gráficos' },
              { value: 'exports', label: 'Exportar' },
              { value: 'scouting', label: 'Scouting' },
              { value: 'interpretation', label: 'Interpretación' },
              { value: 'possession', label: 'Posesión' },
            ]}
            value={activeTab}
          />

          {activeTab === 'video' ? (
            <VideoTab previewUrl={previewUrl} result={result} source={source} />
          ) : null}
          {activeTab === 'stats' ? (
            <div className="space-y-4">
              <PipelineHealthPanel result={result} />
              <TeamComparisonTable result={result} />
            </div>
          ) : null}
          {activeTab === 'charts' ? (
            <Suspense fallback={<LoadingState label="Cargando gráficos temporales..." />}>
              <TimelineCharts result={result} />
            </Suspense>
          ) : null}
          {activeTab === 'exports' ? <ExportsPanel result={result} /> : null}
          {activeTab === 'scouting' ? <ScoutingPanels result={result} /> : null}
          {activeTab === 'interpretation' ? <InterpretationPanel result={result} /> : null}
          {activeTab === 'possession' ? <PossessionPanel result={result} /> : null}
        </>
      ) : (
        <section className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/50 p-6 text-sm text-slate-300">
          Configurá el pipeline y procesá un video para habilitar video comparado, métricas, scouting y exportes.
        </section>
      )}
    </section>
  )
}

function SessionOverview({ result }: { result: ComputerVisionResult }) {
  const cards = [
    { label: 'Duración analizada', value: `${result.duration_seconds.toFixed(1)} s` },
    { label: 'Frames procesados', value: `${result.total_frames}` },
    { label: 'Rendimiento', value: `${result.fps.toFixed(1)} FPS` },
    { label: 'Origen', value: result.source === 'mock' ? 'Mock demo' : 'Backend API' },
  ]

  return (
    <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map((card) => (
        <article className="rounded-xl border border-slate-700 bg-slate-900/70 p-4" key={card.label}>
          <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{card.label}</p>
          <p className="mt-2 text-2xl font-semibold text-slate-100">{card.value}</p>
        </article>
      ))}
    </section>
  )
}

function VideoTab({
  source,
  result,
  previewUrl,
}: {
  source: VideoSourceInput
  result: ComputerVisionResult
  previewUrl?: string
}) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Video original</h3>
        {source.source_mode === 'upload' && previewUrl ? (
          <video className="mt-4 max-h-[420px] w-full rounded-xl bg-slate-950/70" controls src={previewUrl} />
        ) : (
          <div className="mt-4 rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-4 text-sm text-slate-300">
            {source.source_mode === 'soccernet'
              ? `Fuente remota/local: ${source.soccernet_path}`
              : 'No hay preview local disponible.'}
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
        <h3 className="text-lg font-semibold text-slate-100">Salida procesada</h3>
        <div className="mt-4 space-y-3">
          <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <p className="text-sm font-semibold text-slate-100">{result.video_name}</p>
            <p className="mt-2 text-sm text-slate-300">
              {result.source === 'mock'
                ? 'Mostrando una salida demo estable mientras se define el endpoint final del backend.'
                : 'El video fue procesado por el backend y consolidado en la sesión actual.'}
            </p>
          </div>
          {result.status_message ? (
            <div className="rounded-xl border border-amber-700/40 bg-amber-900/20 p-4 text-sm text-amber-50">
              {result.status_message}
            </div>
          ) : null}
        </div>
      </section>
    </div>
  )
}
