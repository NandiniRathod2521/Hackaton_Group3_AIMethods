// Pure helpers for in-browser recording. The recorded video never leaves the device: it is handed
// to the same local extraction as a chosen file, and only keypoints are sent for scoring.

/**
 * Shortest recording worth analysing: one scoring window (WINDOW / FPS s) plus a margin, because
 * sampling at 25 fps loses part of a source frame at the end of the clip.
 */
export function minRecordSeconds(windowS: number): number {
  return windowS + 0.3;
}

/**
 * Container/codec candidates in order of preference. VP8 WebM first: it is Chrome/Edge/Firefox's
 * default MediaRecorder format and the one their <video> decoders handle most reliably; VP9 next;
 * MP4 for Safari. A type whose recording fails to decode is skipped on the next attempt.
 */
export const RECORDING_TYPES = [
  "video/webm;codecs=vp8",
  "video/webm;codecs=vp9",
  "video/webm",
  "video/mp4;codecs=avc1",
  "video/mp4",
] as const;

/** First type both MediaRecorder can record and a <video> element can play back; null if none. */
export function pickRecordingType(
  canRecord: (type: string) => boolean,
  canPlay: (type: string) => boolean = () => true,
  exclude: ReadonlySet<string> = new Set(),
): string | null {
  return RECORDING_TYPES.find((t) => !exclude.has(t) && canRecord(t) && canPlay(t)) ?? null;
}

/** Message when a finished recording can't be opened by this browser's video decoder. */
export function unreadableRecordingMessage(type: string, detail: string, hasFallback: boolean): string {
  return `This browser couldn't read back its own ${type} recording (${detail}). ` +
    (hasFallback ? "Try again: the next recording will use a different format." : "Choose a video file instead.");
}

export function extensionFor(type: string): string {
  return type.startsWith("video/mp4") ? "mp4" : "webm";
}

export function recordingFileName(type: string, at: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${at.getFullYear()}-${pad(at.getMonth() + 1)}-${pad(at.getDate())} ${pad(at.getHours())}.${pad(at.getMinutes())}.${pad(at.getSeconds())}`;
  return `Recording ${stamp}.${extensionFor(type)}`;
}

export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Why recording can't start in this browser at all, or null if it can. */
export function recordingSupportProblem(env: {
  isSecureContext: boolean;
  hasGetUserMedia: boolean;
  hasMediaRecorder: boolean;
}): string | null {
  if (!env.isSecureContext) return "Camera recording needs a secure (https) connection. Open the app over https, or choose a video file instead.";
  if (!env.hasGetUserMedia) return "This browser can't access a camera. Choose a video file instead.";
  if (!env.hasMediaRecorder) return "This browser can't record video. Choose a video file instead, or try a current Chrome, Edge, Firefox or Safari.";
  return null;
}

/** Clear message for a getUserMedia failure. */
export function cameraErrorMessage(err: unknown): string {
  const name = typeof err === "object" && err && "name" in err ? String((err as { name: unknown }).name) : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
    case "PermissionDeniedError":
      return "Camera access was blocked. Allow camera access for this site in your browser's address bar or settings, then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "OverconstrainedError":
      return "No camera was found. Connect a camera, or choose a video file instead.";
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      return "The camera couldn't be started. It may be in use by another app or browser tab; close it and try again.";
    default:
      return `The camera couldn't be started: ${err instanceof Error ? err.message : String(err)}`;
  }
}

export function tooShortMessage(seconds: number, minS: number): string {
  return `The recording was ${seconds.toFixed(1)} s long; at least ${minS.toFixed(1)} s is needed for one analysis window. Record a longer clip.`;
}
