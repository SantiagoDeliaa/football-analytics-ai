import type { ChangeEvent } from 'react'
import type { VideoSourceInput, VideoSourceMode } from '../../types/computerVision'
import { Tabs } from '../common/Tabs'

interface VideoSourcePanelProps {
  source: VideoSourceInput
  loading: boolean
  onModeChange: (mode: VideoSourceMode) => void
  onFileChange: (file?: File) => void
  onSoccernetPathChange: (path: string) => void
}

export function VideoSourcePanel({
  source,
  loading,
  onModeChange,
  onFileChange,
  onSoccernetPathChange,
}: VideoSourcePanelProps) {
  const safeMode: VideoSourceMode = source.source_mode === 'soccernet' ? 'soccernet' : 'upload'
  const safeSoccernetPath = source.soccernet_path ?? ''

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    onFileChange(event.target.files?.[0])
  }

  return (
    <section className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <div>
        <h3 className="text-base font-semibold text-slate-100">Origen del video</h3>
        <p className="mt-1 text-sm text-slate-300">
          Elegí si querés cargar un clip propio o apuntar a un video local de SoccerNet en el backend.
        </p>
      </div>

      <Tabs
        onChange={onModeChange}
        options={[
          { value: 'upload', label: 'Subir archivo' },
          { value: 'soccernet', label: 'SoccerNet local' },
        ]}
        value={safeMode}
      />

      {safeMode === 'upload' ? (
        <label className="block text-sm text-slate-200">
          Archivo de video
          <input
            accept=".mp4,.mov,.avi,.mkv,video/*"
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            disabled={loading}
            onChange={handleFileChange}
            type="file"
          />
          <span className="mt-2 block text-xs text-slate-400">
            {source.file ? `Archivo seleccionado: ${source.file.name}` : 'Formatos soportados: mp4, mov, avi, mkv.'}
          </span>
        </label>
      ) : (
        <label className="block text-sm text-slate-200">
          Ruta del video en SoccerNet/local
          <input
            className="mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2"
            disabled={loading}
            onChange={(event) => onSoccernetPathChange(event.target.value)}
            placeholder="videos/match_001/1_224p.mkv"
            type="text"
            value={safeSoccernetPath}
          />
          <span className="mt-2 block text-xs text-slate-400">
            La ruta se envía al backend para que resuelva el archivo dentro del entorno del servidor.
          </span>
        </label>
      )}
    </section>
  )
}
