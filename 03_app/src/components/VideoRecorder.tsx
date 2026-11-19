import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import {
  cameraErrorMessage, formatElapsed, pickRecordingType, recordingFileName, recordingSupportProblem, tooShortMessage,
  unreadableRecordingMessage,
} from "../recording";
import { mediaErrorDetail } from "../pose/sampleVideo";

// Records a clip from the camera with MediaRecorder. The recording stays in this browser: it is
// handed to the same local keypoint extraction as a chosen file. Only video is requested (no
// microphone), and every track is stopped when recording ends, fails, is cancelled or unmounts.

type Phase =
  | { kind: "starting" }
  | { kind: "preview" }
  | { kind: "recording" }
  | { kind: "finishing" }
  | { kind: "error"; message: string; canRetry: boolean };

interface Props {
  /** Shortest clip that can be analysed, in seconds. */
  minSeconds: number;
  onRecorded: (file: File, durationS: number) => void;
  onCancel: () => void;
}

const CONSTRAINTS: MediaStreamConstraints = {
  audio: false,
  video: { width: { ideal: 1280 }, height: { ideal: 720 } },
};

/** Open the clip in a <video> element, as extraction will; resolves with null if decodable, else the reason. */
function probeDecodable(file: File, timeoutMs = 10000): Promise<string | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    const done = (reason: string | null) => {
      clearTimeout(timer);
      v.onloadeddata = v.onerror = null;
      v.removeAttribute("src"); v.load();
      URL.revokeObjectURL(url);
      resolve(reason);
    };
    const timer = setTimeout(() => done("the video did not load within 10 s"), timeoutMs);
    v.muted = true; v.playsInline = true; v.preload = "auto";
    v.onloadeddata = () => done(null);
    v.onerror = () => done(mediaErrorDetail(v));
    v.src = url;
  });
}

export function VideoRecorder({ minSeconds, onRecorded, onCancel }: Props) {
  const [phase, setPhase] = useState<Phase>({ kind: "starting" });
  const [elapsed, setElapsed] = useState(0);
  const preview = useRef<HTMLVideoElement>(null);
  const startButton = useRef<HTMLButtonElement>(null);
  const stopButton = useRef<HTMLButtonElement>(null);
  const retryButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const discard = useRef(false);        // true: the clip must not be analysed (cancel, error, unmount)
  const startedAt = useRef(0);
  const stoppedAt = useRef(0);
  const timer = useRef<number | null>(null);
  const attempt = useRef(0);            // ignores camera requests that finish after a newer one or unmount
  const mounted = useRef(false);
  const unreadable = useRef(new Set<string>());   // formats whose recordings this browser couldn't decode
  const onRecordedRef = useRef(onRecorded);
  onRecordedRef.current = onRecorded;

  function clearTimer() {
    if (timer.current != null) window.clearInterval(timer.current);
    timer.current = null;
  }

  function releaseCamera() {
    stream.current?.getTracks().forEach((t) => { t.onended = null; t.stop(); });
    stream.current = null;
    if (preview.current) preview.current.srcObject = null;
  }

  /** Stop everything without analysing. Safe to call in any state. */
  function teardown() {
    discard.current = true;
    clearTimer();
    const rec = recorder.current;
    if (rec && rec.state !== "inactive") {
      try { rec.stop(); } catch { /* already stopping */ }
    }
    recorder.current = null;
    releaseCamera();
  }

  function fail(message: string, canRetry = true) {
    teardown();
    if (mounted.current) setPhase({ kind: "error", message, canRetry });
  }

  async function startCamera() {
    const id = ++attempt.current;
    teardown();
    setElapsed(0);
    setPhase({ kind: "starting" });
    const problem = recordingSupportProblem({
      isSecureContext: window.isSecureContext,
      hasGetUserMedia: !!navigator.mediaDevices?.getUserMedia,
      hasMediaRecorder: typeof MediaRecorder !== "undefined",
    });
    if (problem) return fail(problem, false);

    let s: MediaStream;
    try {
      s = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
    } catch (e) {
      if (id === attempt.current && mounted.current) fail(cameraErrorMessage(e));
      return;
    }
    if (id !== attempt.current || !mounted.current) {
      s.getTracks().forEach((t) => t.stop());   // a newer attempt or unmount superseded this one
      return;
    }
    stream.current = s;
    s.getVideoTracks().forEach((t) => {
      t.onended = () => fail("The camera stopped (it may have been disconnected or used by another app). Try again.");
    });
    if (preview.current) {
      preview.current.srcObject = s;
      preview.current.play().catch(() => undefined);
    }
    setPhase({ kind: "preview" });
  }

  function startRecording() {
    const s = stream.current;
    if (!s) return;
    const probe = document.createElement("video");
    const type = pickRecordingType(
      (t) => MediaRecorder.isTypeSupported(t),
      (t) => probe.canPlayType(t) !== "",
      unreadable.current,
    );
    if (!type) {
      return fail("This browser can't record video in a format it can also read back (WebM or MP4). Choose a video file instead.", false);
    }
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(s, { mimeType: type });
    } catch (e) {
      return fail(`This browser can't record ${type} video: ${e instanceof Error ? e.message : String(e)}. Choose a video file instead.`, false);
    }
    chunks.current = [];
    discard.current = false;
    rec.ondataavailable = (ev) => { if (ev.data.size > 0) chunks.current.push(ev.data); };
    rec.onerror = (ev) => {
      const err = (ev as Event & { error?: DOMException }).error;
      fail(`Recording failed${err?.message ? `: ${err.message}` : ""}. Try again, or choose a video file instead.`);
    };
    rec.onstop = () => finish(rec, type);
    recorder.current = rec;
    try {
      rec.start(1000);   // collect data every second so a long recording isn't held in one buffer
    } catch (e) {
      return fail(`Recording couldn't start: ${e instanceof Error ? e.message : String(e)}`);
    }
    startedAt.current = performance.now();
    setElapsed(0);
    clearTimer();
    timer.current = window.setInterval(() => setElapsed((performance.now() - startedAt.current) / 1000), 250);
    setPhase({ kind: "recording" });
  }

  function stopAndAnalyze() {
    const rec = recorder.current;
    if (!rec || rec.state === "inactive") return;
    stoppedAt.current = performance.now();
    clearTimer();
    setPhase({ kind: "finishing" });
    rec.stop();   // -> onstop -> finish
  }

  async function finish(rec: MediaRecorder, type: string) {
    if (recorder.current === rec) recorder.current = null;
    clearTimer();
    releaseCamera();
    if (discard.current || !mounted.current) return;
    const durationS = (stoppedAt.current - startedAt.current) / 1000;
    const mime = rec.mimeType || type;
    const blob = new Blob(chunks.current, { type: mime });
    chunks.current = [];
    if (blob.size === 0) return fail("The recording is empty. Try again.");
    if (durationS < minSeconds) return fail(tooShortMessage(durationS, minSeconds));
    const file = new File([blob], recordingFileName(mime), { type: mime });
    // Check the browser can decode its own recording before handing it to extraction.
    const problem = await probeDecodable(file);
    if (discard.current || !mounted.current) return;
    if (problem) {
      unreadable.current.add(type);
      const probe = document.createElement("video");
      const next = pickRecordingType((t) => MediaRecorder.isTypeSupported(t), (t) => probe.canPlayType(t) !== "", unreadable.current);
      return fail(unreadableRecordingMessage(mime, problem, next != null), next != null);
    }
    onRecordedRef.current(file, durationS);
  }

  function cancel() {
    attempt.current++;
    teardown();
    onCancel();
  }

  useEffect(() => {
    mounted.current = true;
    startCamera();
    return () => {
      mounted.current = false;
      attempt.current++;
      teardown();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Move keyboard focus to the control for the next step.
  useEffect(() => {
    if (phase.kind === "preview") startButton.current?.focus();
    else if (phase.kind === "recording") stopButton.current?.focus();
    else if (phase.kind === "error") (retryButton.current ?? closeButton.current)?.focus();
  }, [phase.kind]);

  const longEnough = elapsed >= minSeconds;
  const status =
    phase.kind === "starting" ? "Requesting camera access…"
    : phase.kind === "preview" ? "Camera ready. Frame your whole body, then start recording."
    : phase.kind === "recording" ? (longEnough
      ? "Recording. Minimum length reached; stop whenever you're ready."
      : `Recording. Keep going for at least ${minSeconds.toFixed(1)} seconds.`)
    : phase.kind === "finishing" ? "Recording stopped. Preparing the clip for analysis…"
    : "";

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") { e.preventDefault(); cancel(); }
  }

  return (
    <section className="recorder" aria-label="Record a video" onKeyDown={onKeyDown}>
      <div className="recorder-preview" hidden={phase.kind === "error"}>
        <video ref={preview} muted playsInline autoPlay aria-label="Live camera preview" />
        {phase.kind === "recording" && (
          <span className="recorder-badge" role="timer" aria-label={`Recording time ${formatElapsed(elapsed)}`}>
            <span className="recorder-dot" aria-hidden="true" /> REC {formatElapsed(elapsed)}
          </span>
        )}
      </div>

      <p role="status" className="recorder-status">{status}</p>

      {phase.kind === "error" && (
        <div className="error-box" role="alert">
          <p>{phase.message}</p>
        </div>
      )}

      <div className="recorder-actions">
        {phase.kind === "preview" && (
          <button ref={startButton} className="button" onClick={startRecording}>Start recording</button>
        )}
        {(phase.kind === "recording" || phase.kind === "finishing") && (
          <button ref={stopButton} className="button" onClick={stopAndAnalyze} disabled={phase.kind === "finishing"}>
            Stop and analyze
          </button>
        )}
        {phase.kind === "error" && phase.canRetry && (
          <button ref={retryButton} className="button" onClick={startCamera}>Try again</button>
        )}
        <button ref={closeButton} onClick={cancel} disabled={phase.kind === "finishing"}>
          {phase.kind === "error" ? "Close" : "Cancel"}
        </button>
      </div>
      <p className="muted recorder-note">
        The recording stays on this device; only body keypoints are sent for scoring. Microphone is not used.
        Press Esc to cancel.
      </p>
    </section>
  );
}
