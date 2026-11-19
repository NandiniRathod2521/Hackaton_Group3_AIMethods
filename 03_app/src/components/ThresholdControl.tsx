import { useId, useState } from "react";
import { decimalsFor, isTrained, type SliderRange } from "../threshold";

// Exploratory threshold slider. It only changes which windows the page marks as above threshold;
// the model, its scores and its trained threshold are untouched and nothing is saved.

interface Props {
  value: number;
  trained: number;
  range: SliderRange;
  count: number;          // windows above `value`
  total: number;
  trainedCount: number;   // windows above the trained threshold (server flags)
  onChange: (v: number) => void;
}

export function ThresholdControl({ value, trained, range, count, total, trainedCount, onChange }: Props) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);   // text being typed in the number box
  const atTrained = isTrained(value, trained);
  const digits = decimalsFor(range.step);
  const pct = (v: number) => ((v - range.min) / (range.max - range.min)) * 100;
  const valueText = `${value.toFixed(digits)}${atTrained ? " (trained threshold)" : ""}, ${count} of ${total} windows above`;

  return (
    <div className={`threshold-control ${atTrained ? "" : "is-custom"}`}>
      <div className="threshold-head">
        <label htmlFor={`${id}-slider`} className="threshold-title">Explore the threshold</label>
        <span className={`pill ${atTrained ? "pill-trained" : "pill-custom"}`}>
          {atTrained ? "Trained threshold" : "Exploratory threshold"}
        </span>
      </div>
      <p className="muted threshold-help" id={`${id}-help`}>
        Moving this only changes which windows are <em>marked</em> on this page. The model, its scores and its trained
        threshold ({trained.toFixed(3)}) don’t change, and nothing here is saved.
      </p>

      <div className="threshold-slider">
        <input id={`${id}-slider`} type="range" min={range.min} max={range.max} step={range.step} value={value}
          aria-describedby={`${id}-help`} aria-valuetext={valueText}
          onChange={(e) => onChange(Number(e.target.value))} />
        <span className="trained-tick" style={{ left: `${pct(trained)}%` }} aria-hidden="true">
          <span className="trained-tick-mark" />
          <span className="trained-tick-label">trained</span>
        </span>
      </div>

      <div className="threshold-row">
        <label className="threshold-number">
          <span className="sr-only">Threshold value</span>
          <input type="number" inputMode="decimal" min={range.min} max={range.max} step={range.step}
            value={draft ?? value.toFixed(digits)}
            onChange={(e) => {
              setDraft(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value !== "" && Number.isFinite(v)) onChange(v);
            }}
            onBlur={() => setDraft(null)}
            onKeyDown={(e) => { if (e.key === "Enter") setDraft(null); }} />
        </label>
        <button type="button" onClick={() => onChange(trained)} disabled={atTrained}>Reset to trained</button>
        <p className="threshold-summary">
          <strong>{count}</strong> of {total} windows above
          {!atTrained && <span className="muted"> · {trainedCount} at the trained threshold</span>}
        </p>
      </div>
    </div>
  );
}
