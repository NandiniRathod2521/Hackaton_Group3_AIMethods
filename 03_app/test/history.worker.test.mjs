// Worker history routes against a real local D1: save/list/load/delete, validation, browser
// scoping, payload limits, and that analysis keeps working when history storage is unavailable.
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { after, before, describe, test } from "node:test";
import worker from "../worker/index.ts";
import { LIST_MAX, MAX_RECORDS_PER_BROWSER, MAX_WINDOWS } from "../worker/history.ts";
import { deleteHistory, listHistory, loadHistory, persistAnalysis } from "../src/history.ts";
import { caller, localD1, newBrowserId, sampleResult, sampleSave } from "./helpers.mjs";

let proxy, env, call, render, renderUrl;

before(async () => {
  proxy = await localD1();
  // Stand-in for the Render model API, to check analysis routing is independent of history.
  render = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(sampleResult()));
    });
  });
  await new Promise((r) => render.listen(0, "127.0.0.1", r));
  renderUrl = `http://127.0.0.1:${render.address().port}`;
  env = { DB: proxy.env.DB, ASSETS: { fetch: async () => new Response("app") }, RENDER_API_URL: renderUrl };
  call = caller(worker, env);
});
after(async () => { await proxy?.dispose(); render?.close(); });

describe("save, list, load, delete", () => {
  test("round trip keeps the summary and per-window data", async () => {
    const browserId = newBrowserId();
    const saved = await call("/api/history", { method: "POST", browserId, body: sampleSave() });
    assert.equal(saved.status, 201, saved.text);
    assert.match(saved.data.id, /^[0-9a-f-]{36}$/);
    assert.equal(saved.data.label, "Video analysis");
    assert.equal(saved.data.analysis_type, "video");
    assert.equal(saved.data.duration_seconds, 8.2);
    assert.equal(saved.data.window_count, 3);
    assert.equal(saved.data.flagged_count, 1);
    assert.equal(saved.data.flagged_percent, 33.33);
    assert.ok(!Number.isNaN(Date.parse(saved.data.created_at)));

    const list = await call("/api/history", { browserId });
    assert.equal(list.status, 200);
    assert.deepEqual(list.data.items, [saved.data]);
    assert.equal(list.data.items[0].windows, undefined, "the list never carries per-window data");

    const loaded = await call(`/api/history/${saved.data.id}`, { browserId });
    assert.equal(loaded.status, 200);
    assert.deepEqual(loaded.data.windows, sampleSave().windows);
    assert.deepEqual(loaded.data.posture_summary, sampleSave().posture_summary);
    assert.equal(loaded.data.threshold, 0.42);
    assert.equal(loaded.data.model, "keypoint_autoencoder");

    const del = await call(`/api/history/${saved.data.id}`, { method: "DELETE", browserId });
    assert.equal(del.status, 200);
    assert.equal((await call(`/api/history/${saved.data.id}`, { browserId })).status, 404);
    assert.equal((await call(`/api/history/${saved.data.id}`, { method: "DELETE", browserId })).status, 404);
    assert.deepEqual((await call("/api/history", { browserId })).data.items, []);
  });

  test("flag counts are computed from the windows, not trusted from the client", async () => {
    const browserId = newBrowserId();
    const windows = sampleSave().windows.map((w) => ({ ...w, above_threshold: true }));
    const saved = await call("/api/history", { method: "POST", browserId, body: sampleSave({ windows }) });
    assert.equal(saved.data.flagged_count, 3);
    assert.equal(saved.data.flagged_percent, 100);
  });

  test("duration is optional", async () => {
    const saved = await call("/api/history", { method: "POST", browserId: newBrowserId(), body: sampleSave({ duration_seconds: null }) });
    assert.equal(saved.status, 201, saved.text);
    assert.equal(saved.data.duration_seconds, null);
  });

  test("list is newest first and limited", async () => {
    const browserId = newBrowserId();
    const ids = [];
    for (let i = 0; i < 3; i++) {
      ids.push((await call("/api/history", { method: "POST", browserId, body: sampleSave() })).data.id);
      await new Promise((r) => setTimeout(r, 5));
    }
    const all = await call("/api/history", { browserId });
    assert.deepEqual(all.data.items.map((i) => i.id), ids.reverse());
    const one = await call("/api/history?limit=1", { browserId });
    assert.equal(one.data.items.length, 1);
    for (const bad of ["0", String(LIST_MAX + 1), "abc", "1.5", "-1"]) {
      assert.equal((await call(`/api/history?limit=${bad}`, { browserId })).status, 400, `limit=${bad}`);
    }
  });

  test(`keeps only the newest ${MAX_RECORDS_PER_BROWSER} records per browser`, async () => {
    const browserId = newBrowserId();
    for (let i = 0; i < MAX_RECORDS_PER_BROWSER + 2; i++) {
      assert.equal((await call("/api/history", { method: "POST", browserId, body: sampleSave() })).status, 201);
    }
    const { count } = await env.DB.prepare("SELECT COUNT(*) AS count FROM analysis_history WHERE browser_id = ?1").bind(browserId).first();
    assert.equal(count, MAX_RECORDS_PER_BROWSER);
  });
});

describe("browser scoping", () => {
  test("another browser can't list, load or delete a record", async () => {
    const owner = newBrowserId(), other = newBrowserId();
    const { data: item } = await call("/api/history", { method: "POST", browserId: owner, body: sampleSave() });
    assert.deepEqual((await call("/api/history", { browserId: other })).data.items, []);
    assert.equal((await call(`/api/history/${item.id}`, { browserId: other })).status, 404);
    assert.equal((await call(`/api/history/${item.id}`, { method: "DELETE", browserId: other })).status, 404);
    assert.equal((await call(`/api/history/${item.id}`, { browserId: owner })).status, 200, "still there for its owner");
  });

  test("every history route requires a valid browser ID", async () => {
    const id = crypto.randomUUID();
    for (const browserId of [undefined, "", "not-a-uuid", "00000000-0000-0000-0000-000000000000", "' OR 1=1 --"]) {
      for (const [path, method, body] of [
        ["/api/history", "GET"], ["/api/history", "POST", sampleSave()], [`/api/history/${id}`, "GET"], [`/api/history/${id}`, "DELETE"],
      ]) {
        const res = await call(path, { method, browserId, body });
        assert.equal(res.status, 400, `${method} ${path} with ${JSON.stringify(browserId)}`);
        assert.match(res.data.detail, /X-Browser-Id/);
      }
    }
  });
});

describe("request validation", () => {
  const browserId = newBrowserId();
  const post = (body, headers) => call("/api/history", { method: "POST", browserId, body, headers });

  test("rejects raw data and file names (unknown fields), storing nothing", async () => {
    for (const extra of [{ frames: [[[1, 2, 0.9]]] }, { keypoints: [] }, { imu_samples: [] }, { file_name: "site-a.mp4" }, { label: "x" }]) {
      const res = await post(sampleSave(extra));
      assert.equal(res.status, 400, JSON.stringify(extra));
      assert.match(res.data.detail, /Unexpected field/);
    }
    const windows = sampleSave().windows.map((w, i) => (i ? w : { ...w, keypoints: [[1, 2, 3]] }));
    assert.equal((await post(sampleSave({ windows }))).status, 400);
    assert.deepEqual((await call("/api/history", { browserId })).data.items, []);
  });

  test("rejects invalid values", async () => {
    const w = sampleSave().windows[0];
    for (const bad of [
      { analysis_type: "imu" }, { analysis_type: "../x" }, { duration_seconds: -1 }, { duration_seconds: "8" },
      { model: "" }, { threshold: "0.4" }, { threshold: -1 }, { interpretation: "x".repeat(501) },
      { windows: [] }, { windows: "[]" }, { windows: [{ ...w, end_seconds: w.start_seconds }] },
      { windows: [{ ...w, reconstruction_score: "NaN" }] }, { windows: [{ ...w, above_threshold: 1 }] },
      { windows: [{ ...w, window: 1.5 }] }, { posture_summary: [] }, { posture_summary: { a: { nested: 1 } } },
      { windows: Array.from({ length: MAX_WINDOWS + 1 }, (_, i) => ({ ...w, window: i })) },
    ]) {
      const res = await post(sampleSave(bad));
      assert.equal(res.status, 400, `${JSON.stringify(bad).slice(0, 80)} -> ${res.text}`);
    }
    assert.equal((await post(sampleSave({ model: undefined }))).status, 400);
  });

  test("rejects malformed and oversized bodies", async () => {
    assert.equal((await post("{not json")).status, 400);
    assert.equal((await post(JSON.stringify(sampleSave()), { "content-type": "text/plain" })).status, 415);
    const huge = JSON.stringify(sampleSave({ interpretation: "x".repeat(300 * 1024) }));
    assert.equal((await post(huge)).status, 413);
  });

  test("rejects bad IDs, routes and methods", async () => {
    assert.equal((await call("/api/history/123", { browserId })).status, 400);
    assert.equal((await call("/api/history/%27%20OR%201%3D1", { browserId })).status, 400);
    assert.equal((await call(`/api/history/${crypto.randomUUID()}/x`, { browserId })).status, 404);
    assert.equal((await call(`/api/history/${crypto.randomUUID()}`, { browserId })).status, 404);
    assert.equal((await call("/api/history", { method: "DELETE", browserId })).status, 405);
    assert.equal((await call(`/api/history/${crypto.randomUUID()}`, { method: "POST", browserId, body: {} })).status, 405);
  });
});

describe("history failures don't affect analysis", () => {
  // Node's fetch needs `duplex` to stream a request body; workerd doesn't. Test-only shim.
  const realFetch = globalThis.fetch;
  before(() => {
    globalThis.fetch = (input, init) => realFetch(input, init?.body instanceof ReadableStream ? { ...init, duplex: "half" } : init);
  });
  after(() => { globalThis.fetch = realFetch; });

  const frames = '{"frames":[[[0,0,0]]]}';
  const analyze = (e) => worker.fetch(new Request("http://app.test/api/analyze/keypoints", {
    method: "POST", headers: { "content-type": "application/json", "content-length": String(frames.length) }, body: frames,
  }), e);
  const brokenDb = { prepare() { throw new Error("D1_ERROR: database unavailable"); }, batch() { throw new Error("D1_ERROR"); } };

  test("no DB binding: history answers 503, analysis still works", async () => {
    const e = { ...env, DB: undefined };
    const res = await caller(worker, e)("/api/history", { browserId: newBrowserId() });
    assert.equal(res.status, 503);
    assert.match(res.data.detail, /not configured/);
    const a = await analyze(e);
    assert.equal(a.status, 200);
    assert.equal((await a.json()).windows_analyzed, 3);
  });

  test("failing D1: history answers a JSON 500, analysis still works", async () => {
    const e = { ...env, DB: brokenDb };
    const c = caller(worker, e);
    for (const [path, method, body] of [["/api/history", "GET"], ["/api/history", "POST", sampleSave()], [`/api/history/${crypto.randomUUID()}`, "GET"]]) {
      const res = await c(path, { method, browserId: newBrowserId(), body });
      assert.equal(res.status, 500, `${method} ${path}`);
      assert.match(res.data.detail, /analysis itself is not affected/);
    }
    assert.equal((await analyze(e)).status, 200);
  });
});

describe("app history client against the Worker", () => {
  const viaWorker = (browserId) => ({
    browserId,
    fetch: (path, init) => worker.fetch(new Request(new URL(path, "http://app.test"), init), env),
  });

  test("persist, list, restore and delete with the app's own functions", async () => {
    const opts = viaWorker(newBrowserId());
    const result = sampleResult();
    const outcome = await persistAnalysis(result, 8.2, 1.92, opts);
    assert.equal(outcome.saved, true);
    const items = await listHistory(opts);
    assert.deepEqual(items.map((i) => i.id), [outcome.item.id]);

    const restored = await loadHistory(outcome.item.id, opts);
    assert.equal(restored.item.duration_seconds, 8.2);
    assert.equal(restored.result.windows_analyzed, result.windows_analyzed);
    assert.equal(restored.result.windows_above_threshold, result.windows_above_threshold);
    assert.equal(restored.result.threshold, result.threshold);
    assert.deepEqual(restored.result.posture_summary, result.posture_summary);
    assert.deepEqual(
      restored.result.windows.map(({ start_sample: _s, ...w }) => w),
      result.windows.map(({ start_sample: _s, ...w }) => w),
      "timeline and jump-to-moment data are restored exactly");

    await deleteHistory(outcome.item.id, opts);
    assert.deepEqual(await listHistory(opts), []);
    await assert.rejects(loadHistory(outcome.item.id, opts), (e) => e.status === 404);
  });

  test("a save that fails on the Worker is reported, not thrown", async () => {
    const opts = { browserId: newBrowserId(), fetch: (path, init) => worker.fetch(new Request(new URL(path, "http://app.test"), init), { ...env, DB: undefined }) };
    const outcome = await persistAnalysis(sampleResult(), 8.2, 1.92, opts);
    assert.equal(outcome.saved, false);
    assert.match(outcome.message, /not configured/);
  });
});
