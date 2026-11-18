// Test hook for tools/browser_extract.mjs: runs the same browser code on (a) pre-decoded
// frames and (b) the original video, so detector and sampling can be checked separately.
import { framesToCsv, type Frame } from "./coco";
import { createDetector, detectFrame, EXTRACTOR, type Detector } from "./movenet";
import { extractVideo } from "./sampleVideo";

let detectorPromise: Promise<{ detector: Detector; backend: string }> | null = null;
const getDetector = () => (detectorPromise ??= createDetector());

async function info() {
  const { backend } = await getDetector();
  return { backend, extractor: EXTRACTOR, userAgent: navigator.userAgent };
}

/** MoveNet on JPEG frames <base>/<tick:06d>.jpg (written by tools/decode_frames.py). */
async function extractImages(base: string, ticks: number) {
  const { detector } = await getDetector();
  detector.reset();
  const t0 = performance.now();
  const frames: Frame[] = [];
  const detection: number[] = [];
  let nameMismatches = 0;
  for (let k = 0; k < ticks; k++) {
    const blob = await (await fetch(`${base}/${String(k).padStart(6, "0")}.jpg`)).blob();
    const bitmap = await createImageBitmap(blob);
    const r = await detectFrame(detector, bitmap, (k * 1000) / 25);
    bitmap.close();
    frames.push(r.frame);
    detection.push(r.detection);
    if (r.nameMismatch) nameMismatches++;
  }
  return {
    csv: framesToCsv(frames, detection), nameMismatches, elapsedMs: performance.now() - t0,
  };
}

/** The real app path: video as a local object URL, sampled in the browser. */
async function extractVideoUrl(url: string) {
  const { detector } = await getDetector();
  const blob = await (await fetch(url)).blob();          // like a File picked by the user
  const objectUrl = URL.createObjectURL(blob);
  try {
    const r = await extractVideo(objectUrl, detector);
    return {
      csv: framesToCsv(r.frames, r.detection), nameMismatches: r.nameMismatches, mediaTimes: r.mediaTimes,
      width: r.width, height: r.height, durationS: r.durationS, sourceFrameS: r.sourceFrameS,
      elapsedMs: r.elapsedMs,
    };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

declare global {
  interface Window { __motionAtlas?: unknown }
}
window.__motionAtlas = { info, extractImages, extractVideoUrl };
