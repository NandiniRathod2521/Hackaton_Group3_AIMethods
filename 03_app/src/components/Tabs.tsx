import { useRef, type KeyboardEvent, type ReactNode } from "react";

// WAI-ARIA tabs: one tab stop, ←/→ (Home/End) move and select. Panels stay mounted (hidden) so
// switching tabs never interrupts an analysis or loses planner edits.

export interface TabDef { id: string; label: ReactNode; }

interface Props {
  tabs: TabDef[];
  selected: string;
  onSelect: (id: string) => void;
  label: string;
}

export const tabId = (id: string) => `tab-${id}`;
export const panelId = (id: string) => `panel-${id}`;

export function Tabs({ tabs, selected, onSelect, label }: Props) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  function onKeyDown(e: KeyboardEvent) {
    const i = tabs.findIndex((t) => t.id === selected);
    const to = e.key === "ArrowRight" ? (i + 1) % tabs.length
      : e.key === "ArrowLeft" ? (i - 1 + tabs.length) % tabs.length
      : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : null;
    if (to == null) return;
    e.preventDefault();
    onSelect(tabs[to].id);
    refs.current[tabs[to].id]?.focus();
  }
  return (
    <div className="tabs" role="tablist" aria-label={label} onKeyDown={onKeyDown}>
      {tabs.map((t) => (
        <button key={t.id} ref={(el) => { refs.current[t.id] = el; }} role="tab" id={tabId(t.id)}
          aria-selected={t.id === selected} aria-controls={panelId(t.id)} tabIndex={t.id === selected ? 0 : -1}
          className="tab" onClick={() => onSelect(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}
