-- Migration 0001: saved analysis summaries (Motion Atlas history).
--
-- Stores only what is needed to redraw a results view: the summary and, per scored window,
-- its index, start/end time, score and flag. Never video, raw keypoints, raw IMU samples or
-- file names. browser_id is a random ID generated in the visitor's browser (demo scoping,
-- not authentication).

CREATE TABLE analysis_history (
  id               TEXT    PRIMARY KEY NOT NULL,                     -- random UUID set by the Worker
  browser_id       TEXT    NOT NULL,                                 -- random UUID from the browser's localStorage
  created_at       INTEGER NOT NULL,                                 -- Unix epoch milliseconds (UTC)
  analysis_type    TEXT    NOT NULL CHECK (analysis_type IN ('video', 'imu')),
  model            TEXT    NOT NULL,
  duration_seconds REAL    CHECK (duration_seconds IS NULL OR duration_seconds > 0),
  window_count     INTEGER NOT NULL CHECK (window_count > 0),
  flagged_count    INTEGER NOT NULL CHECK (flagged_count >= 0 AND flagged_count <= window_count),
  flagged_percent  REAL    NOT NULL CHECK (flagged_percent >= 0 AND flagged_percent <= 100),
  threshold        REAL    NOT NULL,
  interpretation   TEXT    NOT NULL DEFAULT '',
  summary_json     TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(summary_json)),  -- posture summary metrics
  windows_json     TEXT    NOT NULL CHECK (json_valid(windows_json))               -- [[window, start_s, end_s, score, flagged 0/1], ...]
);

-- Recent history for one browser (list, newest first) and per-browser pruning.
CREATE INDEX idx_analysis_history_browser_created
  ON analysis_history (browser_id, created_at DESC, id DESC);

-- Age-based retention clean-up across all browsers.
CREATE INDEX idx_analysis_history_created
  ON analysis_history (created_at);
