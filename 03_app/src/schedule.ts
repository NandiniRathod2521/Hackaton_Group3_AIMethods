// Draft work-schedule planner. Pure functions only: the suggestions follow the rules and values the
// user entered and nothing else. Analysis scores are never used here. Nothing is saved or sent.

export type BlockKind = "task" | "break" | "meal" | "open";

export interface Block {
  id: string;
  kind: BlockKind;
  label: string;
  minutes: number;
  demanding?: boolean;   // user-marked "physically demanding" task
}

export interface TaskInput {
  id: string;
  name: string;
  minutes: number;
  demanding: boolean;
}

export interface PlannerSettings {
  startMin: number;        // shift start, minutes after midnight
  endMin: number;          // shift end (same day, after start)
  shortBreakMin: number;   // short break length
  workStretchMin: number;  // longest stretch of work before a short break
  mealBreakMin: number;    // 0 = no meal break
  alternate: boolean;      // alternate demanding and lighter tasks where possible
}

export interface DayPlan {
  day: string;
  blocks: Block[];
  unscheduled: { name: string; minutes: number }[];   // task time that didn't fit in the shift
}

export const WEEK_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const;
export const DAY_NAMES: Record<string, string> = {
  Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday", Fri: "Friday", Sat: "Saturday", Sun: "Sunday", Day: "Day plan",
};

let seq = 0;
export const newId = (prefix = "b") => `${prefix}${++seq}`;

export function parseTime(s: string): number | null {
  const m = /^(\d{1,2}):(\d{2})$/.exec(s.trim());
  if (!m) return null;
  const h = Number(m[1]), min = Number(m[2]);
  return h < 24 && min < 60 ? h * 60 + min : null;
}

export function formatTime(min: number): string {
  const m = ((Math.round(min) % 1440) + 1440) % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

export function formatDuration(min: number): string {
  const h = Math.floor(min / 60), m = Math.round(min % 60);
  return h ? (m ? `${h} h ${m} min` : `${h} h`) : `${m} min`;
}

/** Problems with the inputs, in plain language; empty when a draft can be generated. */
export function validate(s: PlannerSettings, tasks: TaskInput[]): string[] {
  const errs: string[] = [];
  if (!(s.endMin > s.startMin)) errs.push("The shift end must be later than the start on the same day (overnight shifts aren’t supported yet).");
  if (!(s.workStretchMin >= 15)) errs.push("Set the longest work stretch to at least 15 minutes.");
  if (!(s.shortBreakMin >= 0)) errs.push("Break length can’t be negative.");
  if (!(s.mealBreakMin >= 0)) errs.push("Meal break length can’t be negative.");
  if (!tasks.some((t) => t.name.trim() && t.minutes > 0)) errs.push("Add at least one task with a name and a duration.");
  return errs;
}

/**
 * Task order: as entered, or — when `alternate` is on — demanding and lighter tasks interleaved
 * (each group keeps its entered order), starting with whichever group came first.
 */
export function orderTasks(tasks: TaskInput[], alternate: boolean): TaskInput[] {
  const valid = tasks.filter((t) => t.name.trim() && t.minutes > 0);
  if (!alternate) return valid;
  const hard = valid.filter((t) => t.demanding), light = valid.filter((t) => !t.demanding);
  if (!hard.length || !light.length) return valid;
  const out: TaskInput[] = [];
  let useHard = valid[0].demanding;
  while (hard.length || light.length) {
    const from = (useHard && hard.length) || !light.length ? hard : light;
    out.push(from.shift()!);
    useHard = !useHard;
  }
  return out;
}

/**
 * One day's draft. Rules (all from the user's inputs):
 *  1. Tasks in order; a task longer than the work stretch is split around breaks.
 *  2. A short break after each work stretch.
 *  3. One meal break at the first break due around the middle of the shift.
 *  4. Time left at the end is "Open time"; task time that doesn't fit is listed as not scheduled.
 */
export function generateDay(day: string, s: PlannerSettings, tasks: TaskInput[]): DayPlan {
  const blocks: Block[] = [];
  const unscheduled: DayPlan["unscheduled"] = [];
  const mid = s.startMin + (s.endMin - s.startMin) / 2;
  let cursor = s.startMin, sinceBreak = 0, mealPlaced = s.mealBreakMin <= 0;

  const placeBreak = () => {
    const meal = !mealPlaced && cursor >= mid - s.workStretchMin / 2;
    const want = meal ? s.mealBreakMin : s.shortBreakMin;
    const len = Math.min(want, s.endMin - cursor);
    if (len > 0) {
      blocks.push({ id: newId(), kind: meal ? "meal" : "break", label: meal ? "Meal break" : "Break", minutes: len });
      cursor += len;
    }
    if (meal) mealPlaced = true;
    sinceBreak = 0;
  };

  const segments = new Map<string, Block[]>();
  for (const t of orderTasks(tasks, s.alternate)) {
    let remaining = t.minutes;
    while (remaining > 0) {
      const left = s.workStretchMin - sinceBreak;
      if (sinceBreak > 0 && (left <= 0 || (left < 10 && remaining >= 10))) placeBreak();
      else if (!mealPlaced && sinceBreak > 0 && cursor >= mid + s.workStretchMin / 2) placeBreak();
      const room = s.endMin - cursor;
      if (room <= 0) break;
      const chunk = Math.min(remaining, s.workStretchMin - sinceBreak, room);
      const b: Block = { id: newId(), kind: "task", label: t.name.trim(), minutes: chunk, demanding: t.demanding };
      blocks.push(b);
      segments.set(t.id, [...(segments.get(t.id) ?? []), b]);
      cursor += chunk; sinceBreak += chunk; remaining -= chunk;
    }
    if (remaining > 0) unscheduled.push({ name: t.name.trim(), minutes: remaining });
  }
  for (const parts of segments.values()) {
    if (parts.length > 1) parts.forEach((b, i) => { b.label = `${b.label} (part ${i + 1} of ${parts.length})`; });
  }
  while (blocks.length && blocks[blocks.length - 1].kind !== "task") cursor -= blocks.pop()!.minutes;
  if (cursor < s.endMin) blocks.push({ id: newId(), kind: "open", label: "Open time", minutes: s.endMin - cursor });
  return { day, blocks, unscheduled };
}

export function generatePlan(days: string[], s: PlannerSettings, tasks: TaskInput[]): DayPlan[] {
  return days.map((d) => generateDay(d, s, tasks));
}

export interface TimedBlock extends Block { start: number; end: number; }

/** Blocks run back to back from the shift start; editing a duration moves everything after it. */
export function withTimes(blocks: Block[], startMin: number): TimedBlock[] {
  let t = startMin;
  return blocks.map((b) => { const start = t; t += b.minutes; return { ...b, start, end: t }; });
}

export function dayTotals(blocks: Block[], s: Pick<PlannerSettings, "startMin" | "endMin">) {
  const sum = (k: BlockKind[]) => blocks.filter((b) => k.includes(b.kind)).reduce((a, b) => a + b.minutes, 0);
  const endMin = s.startMin + sum(["task", "break", "meal", "open"]);
  return {
    taskMin: sum(["task"]), breakMin: sum(["break", "meal"]), openMin: sum(["open"]),
    endMin, overrunMin: Math.max(0, endMin - s.endMin), earlyMin: Math.max(0, s.endMin - endMin),
  };
}

// ---- editing (immutable) ----
export function updateBlock(blocks: Block[], id: string, patch: Partial<Omit<Block, "id">>): Block[] {
  return blocks.map((b) => (b.id === id ? { ...b, ...patch } : b));
}
export function removeBlock(blocks: Block[], id: string): Block[] {
  return blocks.filter((b) => b.id !== id);
}
export function moveBlock(blocks: Block[], id: string, delta: -1 | 1): Block[] {
  const i = blocks.findIndex((b) => b.id === id), j = i + delta;
  if (i < 0 || j < 0 || j >= blocks.length) return blocks;
  const out = blocks.slice();
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}
export function addBlock(blocks: Block[], kind: BlockKind, minutes: number): Block[] {
  const label = kind === "task" ? "New task" : kind === "meal" ? "Meal break" : kind === "open" ? "Open time" : "Break";
  return [...blocks, { id: newId(), kind, label, minutes }];
}

/** Plain-text copy of the draft, for the user to paste wherever they choose. */
export function planToText(plans: DayPlan[], s: PlannerSettings): string {
  const lines = ["Draft work schedule (Motion Atlas planner — a draft, not a final or approved schedule)", ""];
  for (const p of plans) {
    lines.push(`${DAY_NAMES[p.day] ?? p.day} · shift ${formatTime(s.startMin)}–${formatTime(s.endMin)}`);
    for (const b of withTimes(p.blocks, s.startMin)) {
      lines.push(`  ${formatTime(b.start)}–${formatTime(b.end)}  ${b.label}${b.kind === "task" && b.demanding ? " (demanding)" : ""}`);
    }
    for (const u of p.unscheduled) lines.push(`  Not scheduled: ${u.name}, ${formatDuration(u.minutes)}`);
    lines.push("");
  }
  return lines.join("\n").trimEnd() + "\n";
}
