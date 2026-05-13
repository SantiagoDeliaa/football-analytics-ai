import type { ComputerVisionConfig, ComputerVisionModelAssets } from '../../types/computerVision'

interface ProcessingConfigPanelProps {
  assets: ComputerVisionModelAssets
  config: ComputerVisionConfig
  loading: boolean
  onAssetsChange: (next: ComputerVisionModelAssets) => void
  onChange: (next: ComputerVisionConfig) => void
  onSubmit: () => void
}

export function ProcessingConfigPanel({
  assets,
  config,
  loading,
  onAssetsChange,
  onChange,
  onSubmit,
}: ProcessingConfigPanelProps) {
  function patch<K extends keyof ComputerVisionConfig>(key: K, value: ComputerVisionConfig[K]) {
    onChange({ ...config, [key]: value })
  }

  function patchAsset<K extends keyof ComputerVisionModelAssets>(key: K, value: ComputerVisionModelAssets[K]) {
    onAssetsChange({ ...assets, [key]: value })
  }

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div>
        <h3 className="text-base font-semibold text-slate-100">Configuración del pipeline</h3>
        <p className="mt-1 text-sm text-slate-300">
          Replica los controles críticos del Streamlit para radar, analítica, posesión y calidad.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <SelectField
          label="Modelo de jugadores"
          onChange={(value) => patch('player_model_source', value as ComputerVisionConfig['player_model_source'])}
          options={[
            { label: 'YOLOv8 genérico', value: 'builtin' },
            { label: 'Modelo custom (.pt)', value: 'custom' },
          ]}
          value={config.player_model_source}
        />
        <SelectField
          label="Modelo YOLOv8"
          onChange={(value) => patch('model_name', value as ComputerVisionConfig['model_name'])}
          options={[
            { label: 'yolov8n.pt', value: 'yolov8n.pt' },
            { label: 'yolov8s.pt', value: 'yolov8s.pt' },
            { label: 'yolov8m.pt', value: 'yolov8m.pt' },
            { label: 'yolov8l.pt', value: 'yolov8l.pt' },
            { label: 'yolov8x.pt', value: 'yolov8x.pt' },
          ]}
          value={config.model_name}
        />
        <SelectField
          label="Modelo de pelota"
          onChange={(value) => patch('ball_model_source', value as ComputerVisionConfig['ball_model_source'])}
          options={[
            { label: 'Heurística sports ball', value: 'heuristic' },
            { label: 'Modelo custom (.pt)', value: 'custom' },
          ]}
          value={config.ball_model_source}
        />
        <SelectField
          label="Modelo de campo"
          onChange={(value) => patch('pitch_source', value as ComputerVisionConfig['pitch_source'])}
          options={[
            { label: 'Homography (recomendado)', value: 'homography' },
            { label: 'Soccana keypoint', value: 'soccana' },
            { label: 'Full field approx', value: 'full_field_approx' },
          ]}
          value={config.pitch_source}
        />
      </div>

      {(config.player_model_source === 'custom' || config.ball_model_source === 'custom') ? (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {config.player_model_source === 'custom' ? (
            <FileField
              file={assets.playerModelFile}
              label="Modelo jugadores (.pt)"
              onChange={(file) => patchAsset('playerModelFile', file)}
            />
          ) : (
            <InfoCard
              label="Modelo jugadores"
              text="Usa el preset YOLOv8 seleccionado arriba para la detección de jugadores."
            />
          )}
          {config.ball_model_source === 'custom' ? (
            <FileField
              file={assets.ballModelFile}
              label="Modelo pelota (.pt)"
              onChange={(file) => patchAsset('ballModelFile', file)}
            />
          ) : (
            <InfoCard
              label="Modelo pelota"
              text="Se mantiene la heurística basada en la clase sports ball."
            />
          )}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <RangeField
          label="Umbral de confianza"
          max={0.9}
          min={0.1}
          onChange={(value) => patch('confidence', value)}
          step={0.05}
          value={config.confidence}
        />
        <SelectField
          label="Tamaño de imagen"
          onChange={(value) => patch('image_size', Number(value) as ComputerVisionConfig['image_size'])}
          options={[
            { label: '640', value: '640' },
            { label: '720', value: '720' },
            { label: '960', value: '960' },
          ]}
          value={`${config.image_size}`}
        />
        <SelectField
          label="Perfil de exportación"
          onChange={(value) => patch('export_profile', value as ComputerVisionConfig['export_profile'])}
          options={[
            { label: 'summary', value: 'summary' },
            { label: 'debug_sampled', value: 'debug_sampled' },
            { label: 'full', value: 'full' },
          ]}
          value={config.export_profile}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <NumberField
          label="Sample stride"
          onChange={(value) => patch('sample_stride', value)}
          value={config.sample_stride}
        />
        <NumberField
          label="Top frames"
          onChange={(value) => patch('topk_frames', value)}
          value={config.topk_frames}
        />
        <CheckboxField
          checked={config.full_field_approx}
          label="Full field approx"
          onChange={(checked) => patch('full_field_approx', checked)}
        />
        <CheckboxField
          checked={config.only_person}
          label="Solo personas"
          onChange={(checked) => patch('only_person', checked)}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        <CheckboxField
          checked={config.enable_radar}
          label="Radar táctico"
          onChange={(checked) => patch('enable_radar', checked)}
        />
        <CheckboxField
          checked={config.enable_analytics}
          label="Análisis táctico"
          onChange={(checked) => patch('enable_analytics', checked)}
        />
        <CheckboxField
          checked={config.enable_possession}
          label="Posesión"
          onChange={(checked) => patch('enable_possession', checked)}
        />
        <CheckboxField
          checked={config.disable_inertia}
          label="Desactivar inercia"
          onChange={(checked) => patch('disable_inertia', checked)}
        />
        <CheckboxField
          checked={config.enable_compression}
          label="Compresión"
          onChange={(checked) => patch('enable_compression', checked)}
        />
        <CheckboxField
          checked={config.segment_mode}
          label="Procesar segmento"
          onChange={(checked) => patch('segment_mode', checked)}
        />
      </div>

      {config.segment_mode ? (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <NumberField
            label="Inicio (seg)"
            onChange={(value) => patch('start_seconds', value)}
            step={1}
            value={config.start_seconds}
          />
          <NumberField
            label="Duración (seg)"
            onChange={(value) => patch('duration_seconds', value)}
            step={1}
            value={config.duration_seconds}
          />
        </div>
      ) : null}

      <button
        className="w-full rounded-md border border-emerald-500 bg-emerald-700 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
        disabled={loading}
        onClick={onSubmit}
        type="button"
      >
        {loading ? 'Procesando video...' : 'Procesar video'}
      </button>
    </section>
  )
}

function FileField({
  label,
  file,
  onChange,
}: {
  label: string
  file?: File
  onChange: (file?: File) => void
}) {
  return (
    <label className="text-sm text-slate-200">
      {label}
      <input
        accept=".pt"
        className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 file:mr-3 file:rounded-md file:border-0 file:bg-slate-800 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-slate-200"
        onChange={(event) => onChange(event.target.files?.[0])}
        type="file"
      />
      <span className="mt-2 block text-xs text-slate-400">
        {file ? `Archivo listo: ${file.name}` : 'Todavía no se cargó ningún archivo.'}
      </span>
    </label>
  )
}

function InfoCard({ label, text }: { label: string; text: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-3">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm text-slate-300">{text}</p>
    </div>
  )
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: { label: string; value: string }[]
  onChange: (value: string) => void
}) {
  const safeValue = value ?? ''

  return (
    <label className="text-sm text-slate-200">
      {label}
      <select
        className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
        onChange={(event) => onChange(event.target.value)}
        value={safeValue}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function RangeField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  const safeValue = Number.isFinite(value) ? value : min

  return (
    <label className="text-sm text-slate-200">
      <span className="flex items-center justify-between">
        {label}
        <span className="text-xs text-slate-400">{safeValue.toFixed(2)}</span>
      </span>
      <input
        className="mt-2 w-full accent-emerald-400"
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="range"
        value={safeValue}
      />
    </label>
  )
}

function NumberField({
  label,
  value,
  onChange,
  step = 5,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  step?: number
}) {
  const safeValue = Number.isFinite(value) ? value : 0

  return (
    <label className="text-sm text-slate-200">
      {label}
      <input
        className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
        min={0}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="number"
        value={safeValue}
      />
    </label>
  )
}

function CheckboxField({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  const safeChecked = Boolean(checked)

  return (
    <label className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm text-slate-200">
      <input checked={safeChecked} onChange={(event) => onChange(event.target.checked)} type="checkbox" />
      <span>{label}</span>
    </label>
  )
}
