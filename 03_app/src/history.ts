import type { KeypointResult } from "./api";

// Saved analysis history (Worker -> D1). Only the result summary and per-window scores/times are
// sent; never video, keypoints or the file name. Records are scoped to a random browser ID kept in
// localStorage. That is a demo scoping mechanism, not authentication.
// (Type-only imports: this module also runs under `node --test`.)

export interface HistoryItem {
  id: string;
  created_at: string;            // ISO 8601
  analysis_type: string;
  label: string;                 // "Video analysis"
  duration_seconds: number | null;
  window_count: number;
  flagged_count: number;
  flagged_percent: number;
}

export interface SavedAnalysis {
  item: HistoryItem;
  result: KeypointResult;
}

export type SaveOutcome = { saved: true; item: HistoryItem } | { saved: false; message: string };

interface Options {
  fetch?: typeof fetch;
  browserId?: string;
}

type KeyValueStore = Pick<Storage, "getItem" | "setItem">;

export const BROWSER_ID_KEY = "motion-atlas.browser-id";
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
let memoryId: string | null = null;

function defaultStore(): KeyValueStore | null {
  try { return globalThis.localStorage ?? null; } catch { return null; }
}

/** Random per-browser ID used to scope history. Falls back to a per-page ID if storage is blocked. */
export function getBrowserId(store: KeyValueStore | null = defaultStore()): string {
  try {
    const existing = store?.getItem(BROWSER_ID_KEY);
    if (existing && UUID_RE.test(existing)) return existing;
    const id = crypto.randomUUID();
    store?.setItem(BROWSER_ID_KEY, id);
    if (store) return id;
  } catch { /* storage blocked: use an in-memory ID */ }
  memoryId ??= crypto.randomUUID();
  return memoryId;
}

/** The exact body sent to POST /api/history: summary + per-window times/scores only. */
export function toSaveRequest(result: KeypointResult, durationS: number | null, windowS: number) {
  return {
    analysis_type: "video",
    duration_seconds: durationS && durationS > 0 ? Math.round(durationS * 1000) / 1000 : null,
    model: result.model,
    threshold: result.threshold,
    interpretation: result.interpretation,
    posture_summary: result.posture_summary,
    windows: result.windows.map((w) => ({
      window: w.window,
      start_seconds: w.start_seconds,
      end_seconds: w.end_seconds ?? Math.round((w.start_seconds + windowS) * 1000) / 1000,
      reconstruction_score: w.reconstruction_score,
      above_threshold: w.above_threshold,
    })),
  };
}

async function call<T>(path: string, init: RequestInit, opts: Options): Promise<T> {
  const doFetch = opts.fetch ?? globalThis.fetch.bind(globalThis);
  const headers = new Headers(init.headers);
  headers.set("x-browser-id", opts.browserId ?? getBrowserId());
  const res = await doFetch(path, { ...init, headers });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    throw new HistoryError(typeof detail === "string" ? detail : `History request failed (HTTP ${res.status}).`, res.status);
  }
  return data as T;
}

export class HistoryError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export function listHistory(opts: Options = {}): Promise<HistoryItem[]> {
  return call<{ items: HistoryItem[] }>("/api/history?limit=20", { method: "GET" }, opts).then((d) => d.items);
}

export async function loadHistory(id: string, opts: Options = {}): Promise<SavedAnalysis> {
  const d = await call<HistoryItem & Omit<KeypointResult, "windows_analyzed" | "windows_above_threshold" | "percentage_above_threshold">>(
    `/api/history/${encodeURIComponent(id)}`, { method: "GET" }, opts);
  const { model, threshold, interpretation, posture_summary, windows, ...item } = d;
  return {
    item,
    result: {
      model, threshold, interpretation, posture_summary,
      windows: windows.map((w) => ({ ...w, start_sample: Math.round(w.start_seconds * 25) })),
      windows_analyzed: item.window_count,
      windows_above_threshold: item.flagged_count,
      percentage_above_threshold: item.flagged_percent,
    },
  };
}

export async function deleteHistory(id: string, opts: Options = {}): Promise<void> {
  await call(`/api/history/${encodeURIComponent(id)}`, { method: "DELETE" }, opts);
}

/** Save a finished analysis. Never throws: a failed save must not affect the shown analysis. */
export async function persistAnalysis(result: KeypointResult, durationS: number | null, windowS: number, opts: Options = {}): Promise<SaveOutcome> {
  try {
    const item = await call<HistoryItem>("/api/history", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(toSaveRequest(result, durationS, windowS)),
    }, opts);
    return { saved: true, item };
  } catch (e) {
    const message = e instanceof HistoryError ? e.message : "Could not reach the history service.";
    return { saved: false, message };
  }
}
