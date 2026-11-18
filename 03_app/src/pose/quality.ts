import { FPS, type Frame } from "./coco";

// Input-quality warnings shown next to the result. Thresholds were set on the 13 prototype
// clips (03_app/prototype_out): legs hidden behind plants, a close-up without legs, and a
// two-worker clip where the single-pose detector jumped between people.

const LEG_POINTS = [13, 14, 15, 16];     // knees, ankles
const LEG_MIN_CONF = 0.2;
const LEGS_WARN_BELOW = 0.8;             // share of detected frames with all leg points found
const FOUND_WARN_BELOW = 0.9;
const JUMP_TORSOS = 1.0;                 // hip centre moving > 1 torso length in 40 ms

export interface QualityWarning { id: string; text: string }

export interface Quality {
  personFound: number;
  legsDetected: number;
  jumpSamples: number[];
  warnings: QualityWarning[];
}

const clock = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export function assessQuality(frames: Frame[], detection: number[]): Quality {
  const found = detection.map((d) => d > 0);
  const nFound = found.filter(Boolean).length;
  const personFound = frames.length ? nFound / frames.length : 0;

  const legsOk = frames.filter((f, t) => found[t] && LEG_POINTS.every((k) => f[k][2] >= LEG_MIN_CONF)).length;
  const legsDetected = nFound ? legsOk / nFound : 0;

  const hip = frames.map((f) => [(f[11][0] + f[12][0]) / 2, (f[11][1] + f[12][1]) / 2]);
  const torsos = frames
    .filter((f, t) => found[t] && [5, 6, 11, 12].every((k) => f[k][2] >= 0.3))
    .map((f) => Math.hypot((f[5][0] + f[6][0]) / 2 - (f[11][0] + f[12][0]) / 2, (f[5][1] + f[6][1]) / 2 - (f[11][1] + f[12][1]) / 2))
    .sort((a, b) => a - b);
  const torso = torsos.length ? torsos[Math.floor(torsos.length / 2)] : NaN;
  const jumpSamples: number[] = [];
  if (Number.isFinite(torso) && torso > 1) {
    for (let t = 1; t < frames.length; t++) {
      if (found[t] && found[t - 1] && Math.hypot(hip[t][0] - hip[t - 1][0], hip[t][1] - hip[t - 1][1]) > JUMP_TORSOS * torso) {
        jumpSamples.push(t);
      }
    }
  }

  const warnings: QualityWarning[] = [];
  if (personFound < FOUND_WARN_BELOW) {
    warnings.push({ id: "found", text: `A person was detected in only ${Math.round(personFound * 100)}% of the video. Windows without enough visible points are not scored.` });
  }
  if (nFound && legsDetected < LEGS_WARN_BELOW) {
    warnings.push({
      id: "legs",
      text: `Knees and ankles were not clearly detected in ${Math.round((1 - legsDetected) * 100)}% of frames with a person. ` +
        "Scores and posture are less reliable when the full body is not visible; film the whole body, head to feet.",
    });
  }
  if (jumpSamples.length) {
    warnings.push({
      id: "jump",
      text: `The tracked pose jumped suddenly ${jumpSamples.length} time${jumpSamples.length > 1 ? "s" : ""} ` +
        `(first at ${clock(jumpSamples[0] / FPS)}). The analysis may have switched to another person or object.`,
    });
  }
  return { personFound, legsDetected, jumpSamples, warnings };
}
