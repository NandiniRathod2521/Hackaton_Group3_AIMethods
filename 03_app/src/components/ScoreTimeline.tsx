import { useState } from "react";
import type { ScoredWindow } from "../api";

// Window scores over the clip timeline. Each bar spans the window's real start..end time
// (from the API), so gaps show windows the server did not score. Above-threshold bars use the
// status colour plus a ▲ marker and legend label (never colour alone). Click a bar to jump.

const W = 720, H = 180, PAD = { l: 44, r: 12, t: 18, b: 28 };

interface Props {
  windows: ScoredWindow[];
  threshold: number;
  durationS: number;
  windowS: number;
  onSelect: (w: ScoredWindow) => void;
}

const endOf = (w: ScoredWindow, windowS: number) => w.end_seconds ?? w.start_seconds + windowS;

function barPath(x: number, y: number, w: number, h: number): string {
  const r = Math.min(4, w / 2, h);   // rounded data end at the top, square on the baseline
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

export function ScoreTimeline({ windows, threshold, durationS, windowS, onSelect }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const maxScore = Math.max(threshold, ...windows.map((w) => w.reconstruction_score)) * 1.15;
  const span = Math.max(durationS, ...windows.map((w) => endOf(w, windowS)));
  const x = (s: number) => PAD.l + (s / span) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / maxScore) * (H - PAD.t - PAD.b);
  const ticks = Array.from({ length: Math.floor(span) + 1 }, (_, i) => i).filter((s) => span <= 12 || s % 2 === 0);
  const hovered = hover == null ? null : windows[hover];

  return (
    <figure className="timeline">
      <figcaption>
        <strong>Movement deviation per window</strong>
        <span className="legend">
          <span className="swatch normal" /> At or below threshold
          <span className="swatch flagged" /> ▲ Above threshold
          <span className="swatch line" /> Threshold
        </span>
      </figcaption>
      <div className="timeline-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
          aria-label={`Reconstruction score for ${windows.length} windows; ${windows.filter((w) => w.above_threshold).length} above threshold`}>
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line className="grid" x1={PAD.l} x2={W - PAD.r} y1={y(maxScore * f)} y2={y(maxScore * f)} />
              <text className="axis" x={PAD.l - 6} y={y(maxScore * f) + 4} textAnchor="end">{(maxScore * f).toFixed(2)}</text>
            </g>
          ))}
          {ticks.map((s) => (
            <text key={s} className="axis" x={x(s)} y={H - 8} textAnchor="middle">{s}s</text>
          ))}
          {windows.map((w, i) => {
            const x0 = x(w.start_seconds) + 1, x1 = x(endOf(w, windowS)) - 1;
            const top = y(w.reconstruction_score);
            return (
              <g key={w.window} className={`bar ${w.above_threshold ? "flagged" : "normal"} ${hover === i ? "hover" : ""}`}
                onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onClick={() => onSelect(w)}>
                {/* hit target larger than the mark: full plot height */}
                <rect className="hit" x={x0} y={PAD.t} width={Math.max(2, x1 - x0)} height={H - PAD.t - PAD.b} />
                <path d={barPath(x0, top, Math.max(2, x1 - x0), y(0) - top)} />
                {w.above_threshold && <text className="marker" x={(x0 + x1) / 2} y={top - 4} textAnchor="middle">▲</text>}
              </g>
            );
          })}
          <line className="baseline" x1={PAD.l} x2={W - PAD.r} y1={y(0)} y2={y(0)} />
          <line className="threshold" x1={PAD.l} x2={W - PAD.r} y1={y(threshold)} y2={y(threshold)} />
          <text className="axis" x={W - PAD.r} y={y(threshold) - 5} textAnchor="end">threshold {threshold.toFixed(3)}</text>
        </svg>
        {hovered && (
          <div className="tooltip" style={{ left: `${(((x(hovered.start_seconds) + x(endOf(hovered, windowS))) / 2) / W) * 100}%` }}>
            <div>Window {hovered.window}: {hovered.start_seconds.toFixed(2)}–{endOf(hovered, windowS).toFixed(2)} s</div>
            <div>Score <strong>{hovered.reconstruction_score.toFixed(3)}</strong> {hovered.above_threshold ? "▲ above threshold" : "at or below threshold"}</div>
            <div className="muted">Click to play this moment</div>
          </div>
        )}
      </div>
      <details className="table-view">
        <summary>Show as table</summary>
        <table>
          <thead><tr><th>Window</th><th>Start (s)</th><th>End (s)</th><th>Score</th><th>Above threshold</th></tr></thead>
          <tbody>
            {windows.map((w) => (
              <tr key={w.window}>
                <td>{w.window}</td><td>{w.start_seconds.toFixed(2)}</td><td>{endOf(w, windowS).toFixed(2)}</td>
                <td>{w.reconstruction_score.toFixed(3)}</td><td>{w.above_threshold ? "▲ yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
