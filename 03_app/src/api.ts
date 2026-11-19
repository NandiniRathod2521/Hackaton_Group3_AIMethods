import type { Frame } from "./pose/coco";

/** One scored window as returned by Render's /predict/keypoints (via the Worker). */
export interface ScoredWindow {
  window: number;
  start_sample: number;      // original first frame (25 fps) of this window in the clip
  start_seconds: number;
  end_seconds?: number;      // added with the window-start fix; older servers omit it
  reconstruction_score: number;
  above_threshold: boolean;
}

export interface KeypointResult {
  model: string;
  windows_analyzed: number;
  threshold: number;
  windows_above_threshold: number;
  percentage_above_threshold: number;
  interpretation: string;
  windows: ScoredWindow[];
  posture_summary: Record<string, number | string>;
}

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;

/** Wake the Render service early (free tier sleeps); errors are ignored. */
export function wakeModelServer(): void {
  fetch("/api/health").catch(() => undefined);
}

/** Send keypoints only (never video) to the Worker, which forwards them to Render. */
export async function analyzeKeypoints(frames: Frame[]): Promise<KeypointResult> {
  const body = JSON.stringify({
    frames: frames.map((f) => f.map(([x, y, p]) => [round(x, 2), round(y, 2), round(p, 4)])),
  });
  const res = await fetch("/api/analyze/keypoints", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(errorMessage(res.status, data), res.status);
  return data as KeypointResult;
}

/** A failed analysis request; `status` is the HTTP status (422 = the input had no scoreable window). */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

function errorMessage(status: number, data: unknown): string {
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => (d as { msg?: string }).msg ?? String(d)).join("; ");
  return `The analysis request failed (HTTP ${status}).`;
}
