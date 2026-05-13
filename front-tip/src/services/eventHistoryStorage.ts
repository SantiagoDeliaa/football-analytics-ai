import type { EventDataResult, EventHistoryEntry, EventSelection, ProviderOption } from '../types/eventData'

const STORAGE_KEY = 'tip:event-history'
const MAX_ENTRIES = 5

function canUseStorage() {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isValidHistoryEntry(value: unknown): value is EventHistoryEntry {
  if (!isRecord(value) || !isRecord(value.result) || !isRecord(value.selection)) {
    return false
  }

  return (
    typeof value.id === 'string' &&
    typeof value.provider === 'string' &&
    typeof value.competition_name === 'string' &&
    typeof value.match_label === 'string' &&
    typeof value.saved_at === 'string' &&
    typeof value.selection.team === 'string' &&
    typeof value.selection.player === 'string' &&
    typeof value.result.match_id === 'string'
  )
}

function readHistory() {
  if (!canUseStorage()) {
    return []
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter(isValidHistoryEntry)
  } catch {
    return []
  }
}

function writeHistory(entries: EventHistoryEntry[]) {
  if (!canUseStorage()) {
    return
  }

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
}

export function listEventHistoryEntries() {
  return readHistory().sort((left, right) => right.saved_at.localeCompare(left.saved_at))
}

export function getEventHistoryEntry(matchId: string) {
  return readHistory().find((entry) => entry.result.match_id === matchId)
}

export function saveEventHistoryEntry(params: {
  provider: ProviderOption
  result: EventDataResult
  selection: EventSelection
}) {
  const currentEntries = readHistory().filter((entry) => entry.result.match_id !== params.result.match_id)

  const nextEntry: EventHistoryEntry = {
    id: `${params.result.match_id}-${Date.now()}`,
    provider: params.provider,
    competition_name: params.result.competition_name,
    match_label: params.result.match_label,
    saved_at: new Date().toISOString(),
    selection: params.selection,
    result: params.result,
  }

  try {
    writeHistory([nextEntry, ...currentEntries].slice(0, MAX_ENTRIES))
  } catch {
    // Best effort persistence for demo UX; the app should continue if storage is full.
  }
}
