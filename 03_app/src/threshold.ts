// Exploratory threshold: re-classifies already-scored windows for display only. It never changes
// the model's scores, its trained threshold, the scoring API, or what is saved to History.

export interface ScoreLike {
  reconstruction_score: number;
  above_threshold: boolean;   // the server's flag at the trained threshold
}

export interface Classification {
  flagged: boolean[];
  count: number;
  percent: number;   // 0..100
}

/** True when the exploratory value is (numerically) the trained threshold. */
export function isTrained(threshold: number, trained: number): boolean {
  return Math.abs(threshold - trained) <= Math.max(1e-9, Math.abs(trained) * 1e-9);
}

/**
 * Which windows are above `threshold`. At the trained threshold the server's own flags are used,
 * so the default view always matches the scoring API exactly; otherwise the server's rule
 * (score > threshold) is applied to the stored scores.
 */
export function classify(windows: ScoreLike[], threshold: number, trained: number): Classification {
  const flagged = isTrained(threshold, trained)
    ? windows.map((w) => w.above_threshold)
    : windows.map((w) => w.reconstruction_score > threshold);
  const count = flagged.filter(Boolean).length;
  return { flagged, count, percent: windows.length ? (count / windows.length) * 100 : 0 };
}

/** Round up to 1, 2, 2.5 or 5 × 10^k so the slider ends on a readable number. */
export function niceCeil(v: number): number {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const m = [1, 2, 2.5, 5, 10].find((s) => s * p >= v * (1 - 1e-12))!;
  return m * p;
}

export interface SliderRange { min: number; max: number; step: number; }

/** 0 … a readable value above both the highest score and the trained threshold. */
export function sliderRange(scores: number[], trained: number): SliderRange {
  const top = Math.max(trained * 1.5, ...scores.map((s) => s * 1.05));
  const max = niceCeil(top);
  return { min: 0, max, step: max / 500 };
}

/** Keep a typed/slid value in range; the trained value itself is always allowed. */
export function clampThreshold(v: number, range: SliderRange): number {
  if (!Number.isFinite(v)) return range.min;
  return Math.min(range.max, Math.max(range.min, v));
}

/** Digits needed to show values at this step without losing the trained threshold's shape. */
export function decimalsFor(step: number): number {
  return Math.min(4, Math.max(2, Math.ceil(-Math.log10(step)) + 1));
}
