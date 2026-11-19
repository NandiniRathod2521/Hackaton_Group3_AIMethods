import { useEffect, useId, useRef, useState } from "react";
import {
  addBlock, dayTotals, DAY_NAMES, formatDuration, formatTime, generatePlan, moveBlock, newId, parseTime, planToText,
  removeBlock, updateBlock, validate, WEEK_DAYS, withTimes, type Block, type BlockKind, type DayPlan, type PlannerSettings,
  type TaskInput,
} from "../schedule";

// Draft schedule planner. Everything stays in this page's memory: nothing is saved to History,
// stored in the browser, or sent anywhere. Suggestions follow only the user's inputs.

export interface AnalysisContext {
  flagged: number;          // windows above the trained threshold
  total: number;
  durationS: number | null;
  saved: boolean;           // from History rather than the current video
}

const KIND_LABEL: Record<BlockKind, string> = { task: "Task", break: "Break", meal: "Meal break", open: "Open time" };

export function SchedulePlanner({ analysis }: { analysis: AnalysisContext | null }) {
  const uid = useId();
  const [mode, setMode] = useState<"day" | "week">("day");
  const [start, setStart] = useState("07:00");
  const [end, setEnd] = useState("15:30");
  const [days, setDays] = useState<string[]>(["Mon", "Tue", "Wed", "Thu", "Fri"]);
  const [tasks, setTasks] = useState<TaskInput[]>([{ id: newId("t"), name: "", minutes: 60, demanding: false }]);
  const [shortBreak, setShortBreak] = useState(10);
  const [stretch, setStretch] = useState(90);
  const [meal, setMeal] = useState(30);
  const [alternate, setAlternate] = useState(true);
  const [plan, setPlan] = useState<{ settings: PlannerSettings; days: DayPlan[] } | null>(null);
  const [edited, setEdited] = useState(false);
  const [current, setCurrent] = useState(0);
  const [errors, setErrors] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [copyText, setCopyText] = useState<string | null>(null);
  const draftHeading = useRef<HTMLHeadingElement>(null);
  const focusAfter = useRef<string | null>(null);

  const settings: PlannerSettings = {
    startMin: parseTime(start) ?? NaN, endMin: parseTime(end) ?? NaN,
    shortBreakMin: shortBreak, workStretchMin: stretch, mealBreakMin: meal, alternate,
  };
  const dayList = mode === "day" ? ["Day"] : WEEK_DAYS.filter((d) => days.includes(d));

  function generate() {
    const errs = validate(settings, tasks);
    if (mode === "week" && dayList.length === 0) errs.push("Choose at least one work day.");
    setErrors(errs);
    if (errs.length) { setStatus(""); return; }
    if (plan && edited && !window.confirm("Generate a new draft? This replaces the changes you made to the current draft.")) return;
    setPlan({ settings, days: generatePlan(dayList, settings, tasks) });
    setEdited(false); setCurrent(0); setCopyText(null);
    setStatus(`Draft created for ${dayList.length === 1 ? DAY_NAMES[dayList[0]].toLowerCase() : `${dayList.length} days`}. Review and edit it below.`);
    requestAnimationFrame(() => draftHeading.current?.focus());
  }

  function editDay(i: number, fn: (b: Block[]) => Block[]) {
    setPlan((p) => p && { ...p, days: p.days.map((d, j) => (j === i ? { ...d, blocks: fn(d.blocks) } : d)) });
    setEdited(true);
  }

  function discard() {
    if (edited && !window.confirm("Discard this draft and your changes to it?")) return;
    setPlan(null); setEdited(false); setCopyText(null); setStatus("Draft discarded.");
  }

  async function copy() {
    if (!plan) return;
    const text = planToText(plan.days, plan.settings);
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Draft copied as text. Paste it wherever you need it.");
      setCopyText(null);
    } catch {
      setCopyText(text);
      setStatus("Copying isn’t available here; the text is shown below to select and copy.");
    }
  }

  const setTask = (id: string, patch: Partial<TaskInput>) => setTasks((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  const taskTotal = tasks.reduce((a, t) => a + (t.name.trim() ? t.minutes || 0 : 0), 0);
  const shiftLen = Number.isFinite(settings.endMin - settings.startMin) ? settings.endMin - settings.startMin : 0;
  const day = plan?.days[current] ?? null;

  return (
    <div className="planner">
      <div className="planner-intro">
        <span className="eyebrow">Planner · draft only</span>
        <h2 className="planner-title">Plan a shift with regular breaks</h2>
        <p className="muted">
          Enter the shift, the tasks and the breaks you want. The planner suggests an order with breaks spread through the shift.
          You can change or remove anything. The draft stays on this page: it isn’t saved, shared or sent anywhere, and reloading the page clears it.
        </p>
      </div>

      {analysis && (
        <details className="card context-card">
          <summary>Optional context: latest movement analysis</summary>
          <p>
            {analysis.saved ? "Saved analysis" : "Current video"}: {analysis.flagged} of {analysis.total} windows above the trained threshold
            {analysis.durationS != null ? ` in a ${analysis.durationS.toFixed(1)} s clip` : ""}.
          </p>
          <p className="muted">
            This compares a short clip with the model’s reference movements (walking and pushing a cart). It isn’t a diagnosis, an injury-risk
            rating or a safety certification, and a short clip can’t show how a whole shift or week will go. It doesn’t change the suggestions below.
            You might use it as one prompt for a conversation about task variety or breaks.
          </p>
        </details>
      )}

      <form className="card planner-form" onSubmit={(e) => { e.preventDefault(); generate(); }} noValidate>
        <fieldset className="field-group">
          <legend>Plan for</legend>
          <div className="segmented">
            {(["day", "week"] as const).map((m) => (
              <label key={m} className={`segment ${mode === m ? "on" : ""}`}>
                <input type="radio" name={`${uid}-mode`} value={m} checked={mode === m} onChange={() => setMode(m)} />
                {m === "day" ? "One day" : "One week"}
              </label>
            ))}
          </div>
          {mode === "week" && (
            <div className="day-picks" role="group" aria-label="Work days">
              {WEEK_DAYS.map((d) => (
                <label key={d} className={`chip ${days.includes(d) ? "on" : ""}`}>
                  <input type="checkbox" checked={days.includes(d)}
                    onChange={(e) => setDays((ds) => (e.target.checked ? [...ds, d] : ds.filter((x) => x !== d)))} />
                  <span aria-hidden="true">{d}</span><span className="sr-only">{DAY_NAMES[d]}</span>
                </label>
              ))}
            </div>
          )}
        </fieldset>

        <fieldset className="field-group">
          <legend>Shift{mode === "week" ? " (same each work day)" : ""}</legend>
          <div className="field-row">
            <label className="field"><span>Starts</span><input type="time" value={start} onChange={(e) => setStart(e.target.value)} required /></label>
            <label className="field"><span>Ends</span><input type="time" value={end} onChange={(e) => setEnd(e.target.value)} required /></label>
            <p className="field-note muted">{shiftLen > 0 ? `${formatDuration(shiftLen)} shift` : ""}</p>
          </div>
        </fieldset>

        <fieldset className="field-group">
          <legend>Tasks{mode === "week" ? " (each work day)" : ""}</legend>
          <ol className="task-inputs">
            {tasks.map((t, i) => (
              <li key={t.id}>
                <label className="field grow"><span>Task {i + 1}</span>
                  <input type="text" value={t.name} placeholder="e.g. Rebar tying" onChange={(e) => setTask(t.id, { name: e.target.value })} />
                </label>
                <label className="field narrow"><span>Minutes</span>
                  <input type="number" inputMode="numeric" min={5} max={720} step={5} value={t.minutes}
                    onChange={(e) => setTask(t.id, { minutes: Math.max(0, Number(e.target.value) || 0) })} />
                </label>
                <label className="check"><input type="checkbox" checked={t.demanding} onChange={(e) => setTask(t.id, { demanding: e.target.checked })} />
                  Physically demanding</label>
                <button type="button" className="icon-btn" aria-label={`Remove task ${i + 1}${t.name ? `, ${t.name}` : ""}`}
                  disabled={tasks.length === 1} onClick={() => setTasks((ts) => ts.filter((x) => x.id !== t.id))}>✕</button>
              </li>
            ))}
          </ol>
          <div className="field-row">
            <button type="button" onClick={() => setTasks((ts) => [...ts, { id: newId("t"), name: "", minutes: 60, demanding: false }])}>+ Add task</button>
            <p className="field-note muted">{formatDuration(taskTotal)} of tasks{shiftLen > 0 ? ` for a ${formatDuration(shiftLen)} shift` : ""}</p>
          </div>
        </fieldset>

        <fieldset className="field-group">
          <legend>Breaks</legend>
          <div className="field-row">
            <label className="field narrow"><span>Short break (min)</span>
              <input type="number" min={0} max={60} step={5} value={shortBreak} onChange={(e) => setShortBreak(Number(e.target.value) || 0)} /></label>
            <label className="field narrow"><span>After every (min of work)</span>
              <input type="number" min={15} max={480} step={5} value={stretch} onChange={(e) => setStretch(Number(e.target.value) || 0)} /></label>
            <label className="field narrow"><span>Meal break (min, 0 = none)</span>
              <input type="number" min={0} max={120} step={5} value={meal} onChange={(e) => setMeal(Number(e.target.value) || 0)} /></label>
          </div>
          <label className="check"><input type="checkbox" checked={alternate} onChange={(e) => setAlternate(e.target.checked)} />
            Alternate physically demanding and lighter tasks where possible</label>
        </fieldset>

        {errors.length > 0 && (
          <div className="error-box" role="alert"><ul>{errors.map((e) => <li key={e}>{e}</li>)}</ul></div>
        )}
        <div className="form-actions">
          <button type="submit" className="button-cta">{plan ? "Generate a new draft" : "Generate draft schedule"}</button>
          {plan && edited && <span className="muted">This replaces your edits (you’ll be asked first).</span>}
        </div>
      </form>

      <p role="status" className="planner-status">{status}</p>

      {plan && day && (
        <section className="card draft" aria-labelledby={`${uid}-draft`}>
          <div className="draft-head">
            <div>
              <span className="eyebrow">Draft{edited ? " · edited" : ""}</span>
              <h3 id={`${uid}-draft`} ref={draftHeading} tabIndex={-1}>
                {plan.days.length > 1 ? "Week draft" : "Day draft"} · {formatTime(plan.settings.startMin)}–{formatTime(plan.settings.endMin)}
              </h3>
            </div>
            <div className="draft-actions">
              <button type="button" onClick={copy}>Copy as text</button>
              <button type="button" onClick={discard}>Discard draft</button>
            </div>
          </div>

          {plan.days.length > 1 && (
            <ul className="week-overview" aria-label="Days in this draft">
              {plan.days.map((d, i) => {
                const tot = dayTotals(d.blocks, plan.settings);
                return (
                  <li key={d.day}>
                    <button type="button" className={`week-day ${i === current ? "on" : ""}`} aria-pressed={i === current}
                      onClick={() => setCurrent(i)}>
                      <span className="week-day-name">{DAY_NAMES[d.day]}</span>
                      <DayBar blocks={d.blocks} />
                      <span className="muted week-day-meta">
                        {formatDuration(tot.taskMin)} tasks · {formatDuration(tot.breakMin)} breaks{tot.overrunMin ? ` · ${tot.overrunMin} min over` : ""}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          <DayEditor key={day.day} day={day} settings={plan.settings} focusAfter={focusAfter}
            onChange={(fn) => editDay(current, fn)} />

          {copyText && (
            <label className="field"><span>Draft as text</span>
              <textarea readOnly rows={Math.min(16, copyText.split("\n").length)} value={copyText} onFocus={(e) => e.target.select()} />
            </label>
          )}
        </section>
      )}

      <aside className="card limits-card" aria-label="About these suggestions">
        <h3>What the suggestions are, and aren’t</h3>
        <ul>
          <li>They follow only the shift, tasks and break settings you entered: tasks in order, a short break after each work stretch, and one meal break around the middle of the shift.</li>
          <li>They don’t know about site rules, legal or contractual break requirements, weather and heat, workload, staffing, or anyone’s individual needs. Check those with your supervisor and the rules that apply to you.</li>
          <li>Movement analysis results aren’t used. A short video can’t determine a working schedule.</li>
          <li>This is a starting point to discuss and adjust, not an approved schedule.</li>
        </ul>
      </aside>
    </div>
  );
}

function DayBar({ blocks }: { blocks: Block[] }) {
  const total = blocks.reduce((a, b) => a + b.minutes, 0) || 1;
  return (
    <span className="day-bar" aria-hidden="true">
      {blocks.map((b) => <span key={b.id} className={`seg ${b.kind} ${b.demanding ? "demanding" : ""}`} style={{ flexGrow: b.minutes / total }} />)}
    </span>
  );
}

function DayEditor({ day, settings, focusAfter, onChange }: {
  day: DayPlan; settings: PlannerSettings; focusAfter: { current: string | null };
  onChange: (fn: (b: Block[]) => Block[]) => void;
}) {
  const timed = withTimes(day.blocks, settings.startMin);
  const tot = dayTotals(day.blocks, settings);
  const inputs = useRef<Record<string, HTMLInputElement | null>>({});
  const addTaskBtn = useRef<HTMLButtonElement>(null);

  // After a remove or move, keep keyboard focus on a sensible control.
  useEffect(() => {
    const id = focusAfter.current;
    if (id == null) return;
    focusAfter.current = null;
    (id === "add" ? addTaskBtn.current : inputs.current[id])?.focus();
  });

  function remove(i: number) {
    const next = day.blocks[i + 1] ?? day.blocks[i - 1];
    focusAfter.current = next ? next.id : "add";
    onChange((bs) => removeBlock(bs, day.blocks[i].id));
  }

  return (
    <div className="day-editor">
      <div className="day-editor-head">
        <h4>{DAY_NAMES[day.day]}</h4>
        <p className="muted">
          {formatDuration(tot.taskMin)} tasks · {formatDuration(tot.breakMin)} breaks{tot.openMin ? ` · ${formatDuration(tot.openMin)} open` : ""} · ends {formatTime(tot.endMin)}
        </p>
      </div>
      <DayBar blocks={day.blocks} />
      {tot.overrunMin > 0 && <p className="note-warn">This day now runs {formatDuration(tot.overrunMin)} past the shift end ({formatTime(settings.endMin)}). Shorten or remove a block if that isn’t intended.</p>}
      {tot.earlyMin > 0 && !day.blocks.some((b) => b.kind === "open") && <p className="muted">The blocks end {formatDuration(tot.earlyMin)} before the shift end.</p>}
      {day.unscheduled.length > 0 && (
        <p className="note-warn">
          Didn’t fit in the shift: {day.unscheduled.map((u) => `${u.name} (${formatDuration(u.minutes)})`).join(", ")}. Shorten tasks or extend the shift and generate again, or add them by hand.
        </p>
      )}

      <ol className="blocks">
        {timed.map((b, i) => (
          <li key={b.id} className={`block ${b.kind}`}>
            <span className="block-time">{formatTime(b.start)}–{formatTime(b.end)}</span>
            <div className="block-fields">
              <label className="field grow"><span className="sr-only">Label for {formatTime(b.start)} block</span>
                <input ref={(el) => { inputs.current[b.id] = el; }} type="text" value={b.label}
                  onChange={(e) => onChange((bs) => updateBlock(bs, b.id, { label: e.target.value }))} />
              </label>
              <label className="field kind"><span className="sr-only">Type</span>
                <select value={b.kind} onChange={(e) => onChange((bs) => updateBlock(bs, b.id, { kind: e.target.value as BlockKind }))}>
                  {(Object.keys(KIND_LABEL) as BlockKind[]).map((k) => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}
                </select>
              </label>
              <label className="field mins"><span className="sr-only">Minutes</span>
                <input type="number" inputMode="numeric" min={5} max={720} step={5} value={b.minutes}
                  onChange={(e) => onChange((bs) => updateBlock(bs, b.id, { minutes: Math.max(0, Number(e.target.value) || 0) }))} />
                <span aria-hidden="true" className="unit">min</span>
              </label>
              {b.kind === "task" && b.demanding && <span className="tag">demanding</span>}
            </div>
            <div className="block-actions">
              <button type="button" className="icon-btn" aria-label={`Move ${b.label} earlier`} disabled={i === 0}
                onClick={() => { focusAfter.current = b.id; onChange((bs) => moveBlock(bs, b.id, -1)); }}>↑</button>
              <button type="button" className="icon-btn" aria-label={`Move ${b.label} later`} disabled={i === timed.length - 1}
                onClick={() => { focusAfter.current = b.id; onChange((bs) => moveBlock(bs, b.id, 1)); }}>↓</button>
              <button type="button" className="icon-btn" aria-label={`Remove ${b.label}`} onClick={() => remove(i)}>✕</button>
            </div>
          </li>
        ))}
      </ol>
      <div className="field-row">
        <button type="button" ref={addTaskBtn} onClick={() => onChange((bs) => addBlock(bs, "task", 30))}>+ Add task</button>
        <button type="button" onClick={() => onChange((bs) => addBlock(bs, "break", settings.shortBreakMin || 10))}>+ Add break</button>
      </div>
    </div>
  );
}
