// App-side history helpers: the browser ID, what is sent when saving, and that a failed save is
// reported as a value (so the analysis that was just shown stays on screen).
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { BROWSER_ID_KEY, getBrowserId, persistAnalysis, toSaveRequest } from "../src/history.ts";
import { sampleResult } from "./helpers.mjs";

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

function memoryStore(initial = {}) {
  const data = { ...initial };
  return { data, getItem: (k) => data[k] ?? null, setItem: (k, v) => { data[k] = String(v); } };
}

describe("browser ID", () => {
  test("is a random UUID kept in localStorage and reused", () => {
    const store = memoryStore();
    const id = getBrowserId(store);
    assert.match(id, UUID_V4);
    assert.equal(store.data[BROWSER_ID_KEY], id);
    assert.equal(getBrowserId(store), id);
    assert.notEqual(getBrowserId(memoryStore()), id, "a different browser gets a different ID");
  });

  test("replaces a tampered value", () => {
    const store = memoryStore({ [BROWSER_ID_KEY]: "' OR 1=1 --" });
    assert.match(getBrowserId(store), UUID_V4);
  });

  test("falls back to a stable in-page ID when storage is blocked", () => {
    const blocked = { getItem() { throw new Error("SecurityError"); }, setItem() { throw new Error("SecurityError"); } };
    const a = getBrowserId(blocked);
    assert.match(a, UUID_V4);
    assert.equal(getBrowserId(blocked), a);
    assert.equal(getBrowserId(null), a);
  });
});

describe("save request", () => {
  test("contains only the summary and per-window score/time/flag", () => {
    const body = toSaveRequest(sampleResult(), 8.2004, 1.92);
    assert.deepEqual(Object.keys(body).sort(),
      ["analysis_type", "duration_seconds", "interpretation", "model", "posture_summary", "threshold", "windows"]);
    assert.equal(body.analysis_type, "video");
    assert.equal(body.duration_seconds, 8.2);
    for (const w of body.windows) {
      assert.deepEqual(Object.keys(w).sort(), ["above_threshold", "end_seconds", "reconstruction_score", "start_seconds", "window"]);
    }
  });

  test("fills a missing end time from the window length (older Render API)", () => {
    const r = sampleResult();
    delete r.windows[1].end_seconds;
    assert.equal(toSaveRequest(r, 8, 1.92).windows[1].end_seconds, 3.84);
  });

  test("sends no duration when it is unknown", () => {
    assert.equal(toSaveRequest(sampleResult(), 0, 1.92).duration_seconds, null);
  });
});

describe("persistAnalysis", () => {
  test("posts JSON with the browser ID header", async () => {
    let seen;
    const fetch = async (path, init) => {
      seen = { path, init };
      return new Response(JSON.stringify({ id: crypto.randomUUID() }), { status: 201 });
    };
    const browserId = crypto.randomUUID();
    const outcome = await persistAnalysis(sampleResult(), 8.2, 1.92, { fetch, browserId });
    assert.equal(outcome.saved, true);
    assert.equal(seen.path, "/api/history");
    assert.equal(seen.init.method, "POST");
    assert.equal(new Headers(seen.init.headers).get("x-browser-id"), browserId);
    assert.equal(new Headers(seen.init.headers).get("content-type"), "application/json");
    // No raw-data keys. (The API's posture note itself mentions "keypoints" in prose, so match keys.)
    assert.doesNotMatch(seen.init.body, /"(start_sample|frames|keypoints|file_?name|name)"\s*:|\.mp4/i);
  });

  test("network failure resolves to { saved: false } and leaves the result untouched", async () => {
    const result = sampleResult();
    const before = structuredClone(result);
    const outcome = await persistAnalysis(result, 8.2, 1.92, { fetch: async () => { throw new TypeError("Failed to fetch"); }, browserId: crypto.randomUUID() });
    assert.deepEqual(outcome, { saved: false, message: "Could not reach the history service." });
    assert.deepEqual(result, before);
  });

  test("server errors resolve to { saved: false } with the server's message", async () => {
    const fetch = async () => new Response(JSON.stringify({ detail: "History storage failed." }), { status: 500 });
    const outcome = await persistAnalysis(sampleResult(), 8.2, 1.92, { fetch, browserId: crypto.randomUUID() });
    assert.deepEqual(outcome, { saved: false, message: "History storage failed." });
  });

  test("non-JSON error pages resolve to { saved: false }", async () => {
    const fetch = async () => new Response("<html>Bad gateway</html>", { status: 502 });
    const outcome = await persistAnalysis(sampleResult(), 8.2, 1.92, { fetch, browserId: crypto.randomUUID() });
    assert.equal(outcome.saved, false);
    assert.match(outcome.message, /HTTP 502/);
  });
});
