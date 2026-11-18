import { useEffect, useRef, useState } from "react";
import { analyzeKeypoints, wakeModelServer, type KeypointResult, type ScoredWindow } from "./api";
import { ScoreTimeline } from "./components/ScoreTimeline";
import { VideoPlayer, type VideoPlayerHandle } from "./components/VideoPlayer";
import { COCO_NAMES, FPS, WINDOW, framesToCsv } from "./pose/coco";
import { createDetector, EXTRACTOR, type Detector } from "./pose/movenet";
import { assessQuality, type Quality } from "./pose/quality";
import { extractVideo, type VideoExtraction } from "./pose/sampleVideo";

// Video-first keypoint analysis. The video stays in this browser: MoveNet extracts one pose per
// frame here, and only the keypoints are sent to the Worker -> Render model API.

type Stage =
  | { kind: "idle" }
  | { kind: "extracting"; done: number; total: number }
  | { kind: "analyzing" }
  | { kind: "done" }
  | { kind: "error"; message: string };

const WINDOW_S = WINDOW / FPS;
const clock = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export default function App() {
  const [detector, setDetector] = useState<{ detector: Detector; backend: string } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [stage, setStage] = useState<Stage>({ kind: "idle" });
  const [extraction, setExtraction] = useState<VideoExtraction | null>(null);
  const [quality, setQuality] = useState<Quality | null>(null);
  const [result, setResult] = useState<KeypointResult | null>(null);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const player = useRef<VideoPlayerHandle>(null);

  useEffect(() => { createDetector().then(setDetector, (e) => setLoadError(String(e))); }, []);
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);

  const busy = stage.kind === "extracting" || stage.kind === "analyzing";

  async function run(f: File) {
    if (!detector) return;
    wakeModelServer();
    setFile(f); setExtraction(null); setQuality(null); setResult(null);
    const url = URL.createObjectURL(f);
    setVideoUrl(url);
    let ex: VideoExtraction;
    try {
      setStage({ kind: "extracting", done: 0, total: 0 });
      ex = await extractVideo(url, detector.detector, (done, total) => setStage({ kind: "extracting", done, total }));
    } catch (e) {
      setStage({ kind: "error", message: `Could not read this video in the browser: ${String(e)}` });
      return;
    }
    setExtraction(ex);
    setQuality(assessQuality(ex.frames, ex.detection));
    await analyze(ex);
  }

  async function analyze(ex: VideoExtraction) {
    if (ex.frames.length < WINDOW) {
      setStage({ kind: "error", message: `The video is ${(ex.frames.length / FPS).toFixed(1)} s long; at least ${WINDOW_S.toFixed(2)} s is needed.` });
      return;
    }
    setStage({ kind: "analyzing" });
    try {
      setResult(await analyzeKeypoints(ex.frames));
      setStage({ kind: "done" });
    } catch (e) {
      setStage({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  const flagged = result?.windows.filter((w) => w.above_threshold) ?? [];
  const jump = (w: ScoredWindow) => player.current?.playFrom(w.start_seconds);

  return (
    <main>
      <header>
        <h1>Motion Atlas</h1>
        <p className="muted">Video movement analysis. Your video is processed in this browser and is never uploaded; only body keypoints are sent for scoring.</p>
      </header>

      <section className="picker">
        <label className="button">
          {detector ? "Choose a video" : "Loading pose model…"}
          <input type="file" accept="video/*" disabled={!detector || busy} hidden
            onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) run(f); }} />
        </label>
        {file && <span className="muted">{file.name}</span>}
        {loadError && <p className="error">The pose model could not be loaded: {loadError}</p>}
      </section>

      {stage.kind === "extracting" && (
        <p role="status">Finding body keypoints… {stage.total ? `${stage.done} / ${stage.total} frames` : "opening video"}</p>
      )}
      {stage.kind === "analyzing" && (
        <p role="status">Scoring movement… The model server may need up to a minute to wake up.</p>
      )}
      {stage.kind === "error" && (
        <div className="error-box" role="alert">
          <p>{stage.message}</p>
          {extraction && <button onClick={() => analyze(extraction)}>Try the analysis again</button>}
        </div>
      )}

      {extraction && videoUrl && (
        <section className="limits" aria-label="Limitations">
          <p>
            <strong>One pose per frame.</strong> The analysis follows a single detected body. It does not reliably keep track of
            one person: if other people are in view, the analysed pose may switch between them. Check the skeleton on the video below.
          </p>
          {quality?.warnings.map((w) => <p key={w.id} className="warning">⚠ {w.text}</p>)}
        </section>
      )}

      {result && extraction && (
        <section aria-label="Result">
          <div className="tiles">
            <div className="tile">
              <span className="tile-label">Windows above threshold</span>
              <span className="tile-value">{result.windows_above_threshold} of {result.windows_analyzed}</span>
              <span className="muted">{result.percentage_above_threshold.toFixed(0)}% of scored {WINDOW_S.toFixed(2)} s windows</span>
            </div>
            <div className="tile">
              <span className="tile-label">Video analysed</span>
              <span className="tile-value">{(extraction.frames.length / FPS).toFixed(1)} s</span>
              <span className="muted">{extraction.frames.length} frames at {FPS} fps</span>
            </div>
          </div>
          <p className="muted">{result.interpretation} “Above threshold” means a window differs from the model’s reference movements (walking and pushing a cart) more than 90% of reference windows did.</p>

          <ScoreTimeline windows={result.windows} threshold={result.threshold} durationS={extraction.frames.length / FPS}
            windowS={WINDOW_S} onSelect={jump} />

          {flagged.length > 0 && (
            <div className="moments">
              <h2>Flagged moments</h2>
              <ul>
                {flagged.map((w) => (
                  <li key={w.window}>
                    <button onClick={() => jump(w)}>▶ {clock(w.start_seconds)}–{clock(w.end_seconds ?? w.start_seconds + WINDOW_S)}</button>
                    <span className="muted"> score {w.reconstruction_score.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {extraction && videoUrl && (
        <section aria-label="Video">
          <label className="toggle">
            <input type="checkbox" checked={showSkeleton} onChange={(e) => setShowSkeleton(e.target.checked)} /> Show the skeleton used for analysis
          </label>
          <VideoPlayer ref={player} src={videoUrl} frames={extraction.frames} width={extraction.width}
            height={extraction.height} showSkeleton={showSkeleton} />
        </section>
      )}

      {result && <Posture summary={result.posture_summary} />}

      {extraction && file && <Diagnostics extraction={extraction} quality={quality} backend={detector?.backend ?? "?"} fileName={file.name} />}
    </main>
  );
}

function Posture({ summary }: { summary: KeypointResult["posture_summary"] }) {
  const num = (k: string) => (typeof summary[k] === "number" ? `${(summary[k] as number).toFixed(0)}` : "—");
  return (
    <section className="posture" aria-label="Posture">
      <h2>Approximate 2D posture</h2>
      <dl>
        <div><dt>Median trunk lean</dt><dd>{num("median trunk lean (degrees)")}°</dd></div>
        <div><dt>Median arm elevation</dt><dd>{num("median arm elevation (degrees)")}°</dd></div>
        <div><dt>Frames with shoulders and hips visible</dt><dd>{num("frames with shoulders/hips visible")}%</dd></div>
      </dl>
      <details>
        <summary>Experimental: knee bend (MoveNet estimate)</summary>
        <p>Median knee bend: {num("median knee bend (degrees)")}°</p>
        <p className="muted">
          Experimental. On our test clips MoveNet reported more knee bend than our YOLO extraction (about 5–6° more on median, up to 16°),
          and the pose detector behind the training data is unknown. Don’t compare this value with reference data or other detectors.
        </p>
      </details>
      <p className="muted">{String(summary.note ?? "")}</p>
    </section>
  );
}

function Diagnostics({ extraction: r, quality, backend, fileName }: { extraction: VideoExtraction; quality: Quality | null; backend: string; fileName: string }) {
  const conf = r.frames.flat().map((p) => p[2]);
  const min = (a: number[]) => a.reduce((m, v) => Math.min(m, v), Infinity);
  const max = (a: number[]) => a.reduce((m, v) => Math.max(m, v), -Infinity);
  function download() {
    const blob = new Blob([framesToCsv(r.frames, r.detection)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fileName.replace(/\.[^.]+$/, "") + ".movenet.keypoints.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }
  const rows: [string, string][] = [
    ["Extractor", `${EXTRACTOR} (${backend})`],
    ["Video (upright)", `${r.width} × ${r.height} px, ${r.durationS.toFixed(2)} s, source ≈ ${(1 / r.sourceFrameS).toFixed(1)} fps`],
    ["Samples", `${r.frames.length} at ${FPS} fps`],
    ["Point order", r.nameMismatches ? `${r.nameMismatches} frames not in COCO order` : `COCO-17 (${COCO_NAMES[0]} … ${COCO_NAMES[16]})`],
    ["Person found", quality ? `${(quality.personFound * 100).toFixed(1)}% of frames` : "—"],
    ["Legs detected", quality ? `${(quality.legsDetected * 100).toFixed(1)}% of frames with a person` : "—"],
    ["Confidence range", `${min(conf).toFixed(3)} – ${max(conf).toFixed(3)}`],
    ["Extraction time", `${(r.elapsedMs / 1000).toFixed(1)} s`],
  ];
  return (
    <details className="diagnostics">
      <summary>Diagnostics</summary>
      <table><tbody>{rows.map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}</tbody></table>
      <button onClick={download}>Download keypoints CSV</button>
    </details>
  );
}
