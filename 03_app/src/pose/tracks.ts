import type { Frame } from "./coco";

// EXPERIMENTAL multi-person post-processing (pure; no DOM). Turns MoveNet MultiPose output with
// tracker IDs into per-person tracks labelled A, B, C … in order of first appearance.
//
// Guarantees:
//  - A track's frames only ever come from poses the tracker gave that track. Frames from two
//    people are never mixed into one frame.
//  - Scoring happens per *segment*: a run of one tracker ID with no long dropout and no sudden
//    jump. A scored window therefore never spans a tracker ID change, a rejoin or a jump, and each
//    segment is sent to the API on its own so the server's clip-wide normalisation only sees that
//    segment's frames.
//  - Rejoining split fragments ("relink") only affects the label shown, never what is scored.
//
// Labels are temporary and only mean something within one clip; they do not identify a worker.
// Window rules mirror 02_model_server/keypoint_utils.py (WINDOW 48, PMIN 0.02, MAX_MISSING 0.30).

export interface RawPose { id: number; score: number; kp: Frame }

export interface TrackSettings {
  window: number;          // frames per scored window (keypoint_utils.WINDOW)
  pmin: number;            // point counts as visible at this confidence (PMIN)
  maxMissing: number;      // window skipped when more than this share of points is missing (MAX_MISSING)
  minLabelFrames: number;  // shorter tracks are treated as brief detections, not labelled
  maxDropout: number;      // a gap of more than this many frames ends a segment
  jumpTorsos: number;      // hip-centre move per frame (in torso lengths) treated as a jump
  relink: boolean;         // rejoin fragments for labelling
  relinkMaxGap: number;    // frames
  relinkMaxDist: number;   // hip-centre distance in torso lengths
  duplicateDist: number;   // median hip distance (torso lengths) for "probably the same person twice"
}

export const DEFAULT_TRACK_SETTINGS: TrackSettings = {
  window: 48, pmin: 0.02, maxMissing: 0.3,
  minLabelFrames: 12, maxDropout: 5, jumpTorsos: 1.0,
  relink: true, relinkMaxGap: 25, relinkMaxDist: 1.5, duplicateDist: 0.5,
};

export interface Segment {
  start: number;            // first frame (inclusive)
  end: number;              // last frame + 1
  rawId: number;
  fullWindows: number;      // floor(length / window)
  scoreableWindows: number; // full windows with enough visible points (the server's rule)
}

export type TrackIssue =
  | { kind: "dropout"; start: number; end: number }   // frames [start, end) without this track
  | { kind: "rejoin"; frame: number; gap: number }    // fragments joined for labelling at `frame`
  | { kind: "jump"; frame: number }
  | { kind: "duplicate"; of: string };                // overlaps another track (same person twice?)

export interface Track {
  label: string;
  rawIds: number[];
  frames: (Frame | null)[];   // per clip frame; null where this track wasn't detected
  rawIdAt: (number | null)[]; // which tracker ID supplied each frame
  present: number;            // frames with a pose
  first: number;
  last: number;
  segments: Segment[];
  issues: TrackIssue[];
  scoreable: boolean;         // at least one segment has a scoreable window and the track isn't a duplicate
}

export interface TrackResult {
  frameCount: number;
  tracks: Track[];
  brief: { rawId: number; frames: number }[];   // too short to label
  maxPeople: number;
}

const hip = (f: Frame) => [(f[11][0] + f[12][0]) / 2, (f[11][1] + f[12][1]) / 2] as const;
const torsoLen = (f: Frame) => Math.hypot((f[5][0] + f[6][0]) / 2 - (f[11][0] + f[12][0]) / 2, (f[5][1] + f[6][1]) / 2 - (f[11][1] + f[12][1]) / 2);
const coreVisible = (f: Frame) => [5, 6, 11, 12].every((k) => f[k][2] >= 0.3);
const median = (a: number[]) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

export function trackLabel(i: number): string {
  let s = "", n = i;
  do { s = String.fromCharCode(65 + (n % 26)) + s; n = Math.floor(n / 26) - 1; } while (n >= 0);
  return s;
}

interface Raw { id: number; at: Map<number, Frame>; first: number; last: number }

function typicalTorso(frames: Frame[]): number {
  return median(frames.filter(coreVisible).map(torsoLen).filter((t) => t > 1));
}

/** Share of points visible (p >= pmin) over frames [s, e), missing frames counting as invisible. */
function visibleShare(frames: (Frame | null)[], s: number, e: number, pmin: number): number {
  let ok = 0;
  for (let t = s; t < e; t++) { const f = frames[t]; if (f) for (const p of f) if (p[2] >= pmin) ok++; }
  return ok / ((e - s) * 17);
}

export function buildTracks(poses: RawPose[][], settings: Partial<TrackSettings> = {}): TrackResult {
  const S = { ...DEFAULT_TRACK_SETTINGS, ...settings };
  const T = poses.length;
  const raws = new Map<number, Raw>();
  let maxPeople = 0;
  poses.forEach((ps, t) => {
    maxPeople = Math.max(maxPeople, ps.length);
    for (const p of ps) {
      let r = raws.get(p.id);
      if (!r) raws.set(p.id, (r = { id: p.id, at: new Map(), first: t, last: t }));
      r.at.set(t, p.kp); r.last = t;
    }
  });

  // Chains of fragments (one per labelled person). Without relink each raw ID is its own chain.
  const sorted = [...raws.values()].sort((a, b) => a.first - b.first || a.id - b.id);
  const chains: { parts: Raw[]; joins: { frame: number; gap: number }[] }[] = [];
  for (const r of sorted) {
    let best: (typeof chains)[number] | null = null, bestD = Infinity;
    if (S.relink) {
      const rFirst = r.at.get(r.first)!;
      for (const c of chains) {
        const tail = c.parts[c.parts.length - 1];
        const gap = r.first - tail.last - 1;
        if (gap < 0 || gap > S.relinkMaxGap) continue;
        const lastF = tail.at.get(tail.last)!;
        const tc = typicalTorso([...tail.at.values()]), tr = typicalTorso([...r.at.values()]);
        if (!(tc > 1)) continue;
        if (tr > 1 && (tr / tc < 0.67 || tr / tc > 1.5)) continue;
        const [ax, ay] = hip(lastF), [bx, by] = hip(rFirst);
        const d = Math.hypot(ax - bx, ay - by) / tc;
        if (d <= S.relinkMaxDist && d < bestD) { best = c; bestD = d; }
      }
    }
    if (best) {
      const tail = best.parts[best.parts.length - 1];
      best.joins.push({ frame: r.first, gap: r.first - tail.last - 1 });
      best.parts.push(r);
    } else chains.push({ parts: [r], joins: [] });
  }

  const brief: TrackResult["brief"] = [];
  const kept = chains.filter((c) => {
    const n = c.parts.reduce((a, p) => a + p.at.size, 0);
    if (n < S.minLabelFrames) { c.parts.forEach((p) => brief.push({ rawId: p.id, frames: p.at.size })); return false; }
    return true;
  });

  const tracks: Track[] = kept.map((c, i) => {
    const frames: (Frame | null)[] = Array(T).fill(null);
    const rawIdAt: (number | null)[] = Array(T).fill(null);
    for (const p of c.parts) for (const [t, f] of p.at) { frames[t] = f; rawIdAt[t] = p.id; }
    const issues: TrackIssue[] = c.joins.map((j) => ({ kind: "rejoin" as const, frame: j.frame, gap: j.gap }));
    const first = c.parts[0].first, last = c.parts[c.parts.length - 1].last;

    // jumps within one tracker ID
    const torso = typicalTorso(frames.filter((f): f is Frame => f != null));
    const jumpAt = new Set<number>();
    let prev = -1;
    for (let t = first; t <= last; t++) {
      const f = frames[t];
      if (!f) continue;
      if (prev >= 0 && t - prev <= 2 && rawIdAt[prev] === rawIdAt[t] && torso > 1) {
        const [ax, ay] = hip(frames[prev]!), [bx, by] = hip(f);
        if (Math.hypot(ax - bx, ay - by) > S.jumpTorsos * torso * (t - prev)) { jumpAt.add(t); issues.push({ kind: "jump", frame: t }); }
      }
      prev = t;
    }

    // segments: same tracker ID, dropouts <= maxDropout, broken at jumps
    const segments: Segment[] = [];
    let segStart = -1, segId: number | null = null, lastSeen = -1;
    const close = (end: number) => {
      if (segStart < 0) return;
      const len = end - segStart, full = Math.floor(len / S.window);
      let ok = 0;
      for (let w = 0; w < full; w++) {
        const s = segStart + w * S.window;
        if (visibleShare(frames, s, s + S.window, S.pmin) >= 1 - S.maxMissing) ok++;
      }
      segments.push({ start: segStart, end, rawId: segId!, fullWindows: full, scoreableWindows: ok });
      segStart = -1;
    };
    for (let t = first; t <= last; t++) {
      const f = frames[t];
      if (!f) continue;
      const gap = lastSeen >= 0 ? t - lastSeen - 1 : 0;
      if (gap > 0 && gap > S.maxDropout) issues.push({ kind: "dropout", start: lastSeen + 1, end: t });
      if (segStart >= 0 && (rawIdAt[t] !== segId || gap > S.maxDropout || jumpAt.has(t))) close(lastSeen + 1);
      if (segStart < 0) { segStart = t; segId = rawIdAt[t]; }
      lastSeen = t;
    }
    close(lastSeen + 1);

    const present = frames.filter(Boolean).length;
    return {
      label: trackLabel(i), rawIds: c.parts.map((p) => p.id), frames, rawIdAt, present, first, last,
      segments, issues, scoreable: segments.some((s) => s.scoreableWindows > 0),
    };
  });

  // Likely duplicates: a track that mostly sits on top of a longer one at the same time.
  for (const a of tracks) {
    for (const b of tracks) {
      if (a === b || b.present < a.present || (b.present === a.present && b.label > a.label)) continue;
      const d: number[] = [];
      for (let t = 0; t < T; t++) {
        const fa = a.frames[t], fb = b.frames[t];
        if (!fa || !fb) continue;
        const tb = torsoLen(fb);
        if (tb > 1) { const [ax, ay] = hip(fa), [bx, by] = hip(fb); d.push(Math.hypot(ax - bx, ay - by) / tb); }
      }
      if (d.length >= 10 && d.length >= 0.5 * a.present && median(d) < S.duplicateDist) {
        a.issues.push({ kind: "duplicate", of: b.label });
        a.scoreable = false;
        break;
      }
    }
  }
  return { frameCount: T, tracks, brief, maxPeople };
}

/** The frames sent to the API for one segment: only this tracker ID's poses; gaps are empty frames. */
export function segmentFrames(track: Track, seg: Segment, empty: () => Frame): Frame[] {
  const out: Frame[] = [];
  for (let t = seg.start; t < seg.end; t++) out.push(track.rawIdAt[t] === seg.rawId && track.frames[t] ? track.frames[t]! : empty());
  return out;
}

export function trackSummary(t: Track) {
  const windows = t.segments.reduce((a, s) => a + s.fullWindows, 0);
  const scoreable = t.segments.reduce((a, s) => a + s.scoreableWindows, 0);
  return { windows, scoreable, duplicate: t.issues.some((i) => i.kind === "duplicate") };
}

const clockAt = (frame: number, fps = 25) => { const s = frame / fps; return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`; };

/** Plain-language continuity warnings for one track, with times to check on the overlay. */
export function describeIssues(t: Track, frameCount: number, fps = 25): string[] {
  const out: string[] = [];
  const cover = t.present / frameCount;
  if (cover < 0.9) out.push(`Detected in ${Math.round(cover * 100)}% of the clip (${clockAt(t.first, fps)}–${clockAt(t.last + 1, fps)}).`);
  const drops = t.issues.filter((i): i is Extract<TrackIssue, { kind: "dropout" }> => i.kind === "dropout");
  if (drops.length) {
    const longest = drops.reduce((a, d) => (d.end - d.start > a.end - a.start ? d : a));
    out.push(`Lost ${drops.length} time${drops.length > 1 ? "s" : ""} for more than ${((DEFAULT_TRACK_SETTINGS.maxDropout + 1) / fps).toFixed(2)} s; ` +
      `longest gap ${((longest.end - longest.start) / fps).toFixed(1)} s at ${clockAt(longest.start, fps)}.`);
  }
  for (const r of t.issues) {
    if (r.kind === "rejoin") out.push(`The tracker split this person at ${clockAt(r.frame, fps)}; the pieces were joined under one label because they line up. Check the overlay there — it may be a different person.`);
  }
  const jumps = t.issues.filter((i) => i.kind === "jump");
  if (jumps.length) out.push(`The pose jumped suddenly ${jumps.length} time${jumps.length > 1 ? "s" : ""} (first at ${clockAt((jumps[0] as { frame: number }).frame, fps)}); it may have switched to someone else.`);
  // Same rule as the single-person quality check (src/pose/quality.ts): knees and ankles at >= 0.2.
  const posed = t.frames.filter((f): f is Frame => f != null);
  const legsMissing = posed.length ? posed.filter((f) => ![13, 14, 15, 16].every((k) => f[k][2] >= 0.2)).length / posed.length : 0;
  if (legsMissing > 0.2) {
    out.push(`Knees and ankles weren’t clearly detected in ${Math.round(legsMissing * 100)}% of this track’s frames. ` +
      "If the person is only partly in view, the skeleton and its scores are less reliable.");
  }
  const dup = t.issues.find((i): i is Extract<TrackIssue, { kind: "duplicate" }> => i.kind === "duplicate");
  if (dup) out.push(`Mostly overlaps Track ${dup.of}; it is probably the same person detected twice, so it isn’t scored.`);
  return out;
}
