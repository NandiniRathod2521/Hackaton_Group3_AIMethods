import { FPS, type Frame } from "./coco";
import { detectFrame, type Detector } from "./movenet";

export interface VideoExtraction {
  frames: Frame[];
  detection: number[];
  /** Presentation time (s) of the source frame actually analysed for each 25 fps sample. */
  mediaTimes: (number | null)[];
  width: number;
  height: number;
  durationS: number;
  sourceFrameS: number;      // estimated source frame duration
  nameMismatches: number;
  elapsedMs: number;
}

type RVFCVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
};

function once(el: EventTarget, type: string): Promise<void> {
  return new Promise((resolve, reject) => {
    el.addEventListener(type, () => resolve(), { once: true });
    el.addEventListener("error", () => reject(new Error(`video error while waiting for ${type}`)), { once: true });
  });
}

/** Resolve with the presented frame's mediaTime, or null if no new frame shows up in time. */
function nextFrameTime(video: RVFCVideo, timeoutMs: number): Promise<number | null> {
  return new Promise((resolve) => {
    if (!video.requestVideoFrameCallback) return resolve(null);
    const timer = setTimeout(() => resolve(null), timeoutMs);
    video.requestVideoFrameCallback((_now, meta) => { clearTimeout(timer); resolve(meta.mediaTime); });
  });
}

async function seek(video: RVFCVideo, t: number): Promise<number | null> {
  const frameTime = nextFrameTime(video, 300);
  const seeked = once(video, "seeked");
  video.currentTime = t;
  await seeked;
  return frameTime;
}

/** Estimate the source frame duration from frame presentation times during a short muted play. */
async function estimateFrameDuration(video: RVFCVideo): Promise<number> {
  const times: number[] = [];
  if (video.requestVideoFrameCallback) {
    const collect = (_n: number, meta: { mediaTime: number }) => {
      times.push(meta.mediaTime);
      if (times.length < 20) video.requestVideoFrameCallback!(collect);
    };
    video.requestVideoFrameCallback(collect);
    await video.play().catch(() => undefined);
    const deadline = performance.now() + 1500;
    while (times.length < 20 && performance.now() < deadline) await new Promise((r) => setTimeout(r, 30));
    video.pause();
  }
  const diffs = times.slice(1).map((t, i) => t - times[i]).filter((d) => d > 0.004).sort((a, b) => a - b);
  if (!diffs.length) return 1 / 30;
  const smallest = diffs[0];
  const cluster = diffs.filter((d) => d < smallest * 1.5);   // ignore skipped frames during playback
  return cluster[Math.floor(cluster.length / 2)];
}

/**
 * Sample a local video at 25 fps the way video_to_keypoints.py does: for output tick k take the
 * source frame nearest to k/25 s (seek to k/25 + half a source frame, which shows that frame).
 * The video stays in the browser; only keypoints come out.
 */
export async function extractVideo(
  src: string,
  detector: Detector,
  onProgress?: (done: number, total: number) => void,
): Promise<VideoExtraction> {
  const t0 = performance.now();
  const video = document.createElement("video") as RVFCVideo;
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = src;
  await once(video, "loadeddata");
  const width = video.videoWidth;   // already upright: browsers apply the rotation tag
  const height = video.videoHeight;
  const durationS = video.duration;
  const frameS = await estimateFrameDuration(video);

  const ticks = Math.floor((durationS - frameS) * FPS + 1e-6) + 1;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

  detector.reset();
  const frames: Frame[] = [];
  const detection: number[] = [];
  const mediaTimes: (number | null)[] = [];
  let nameMismatches = 0;
  for (let k = 0; k < ticks; k++) {
    const target = Math.min(k / FPS + frameS / 2 - 1e-4, durationS - 1e-3);
    const shown = await seek(video, Math.max(0, target));   // null = no new frame (same frame still shown)
    ctx.drawImage(video, 0, 0, width, height);
    const r = await detectFrame(detector, canvas, (k * 1000) / FPS);
    frames.push(r.frame);
    detection.push(r.detection);
    mediaTimes.push(shown);
    if (r.nameMismatch) nameMismatches++;
    onProgress?.(k + 1, ticks);
  }
  video.removeAttribute("src");
  video.load();
  return {
    frames, detection, mediaTimes, width, height, durationS, sourceFrameS: frameS,
    nameMismatches, elapsedMs: performance.now() - t0,
  };
}
