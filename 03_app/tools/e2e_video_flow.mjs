#!/usr/bin/env node
// End-to-end check of the video flow in headless Edge against a running app
// (`npx vite` with .dev.vars -> local or Render model API).
//
//   node tools/e2e_video_flow.mjs --url http://127.0.0.1:5179 --videos "<dir>" --clips "4.03.49,4.03.52" [--shots <dir>]
//
// For each clip: choose the file, wait for the result, then check
//  - only keypoint JSON is sent (no request carries video; every POST goes to /api/analyze/keypoints)
//  - timeline bars and flagged moments use the API's start/end times
//  - clicking a flagged moment (or the first bar) seeks the video to that window's start
//  - quality warnings shown
import { readdir, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright-core";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const VIDEOS = resolve(args.videos);
const wanted = (args.clips ?? "").split(",").filter(Boolean);
const files = (await readdir(VIDEOS)).filter((f) => /\.(mp4|mov|webm)$/i.test(f) && (!wanted.length || wanted.some((w) => f.includes(w))));
if (args.shots) await mkdir(args.shots, { recursive: true });

const browser = await chromium.launch({ channel: args.channel ?? "msedge", headless: true, args: ["--enable-unsafe-swiftshader"] });
const failures = [];
const check = (clip, ok, msg) => { console.log(`  ${ok ? "PASS" : "FAIL"} ${msg}`); if (!ok) failures.push(`${clip}: ${msg}`); };

for (const file of files) {
  const clip = file.replace(/^WhatsApp Video 2026-11-17 at /, "").replace(/\.mp4$/, "");
  console.log(`\n${clip}`);
  const page = await browser.newPage({ viewport: { width: 1000, height: 1400 } });
  const requests = [];
  page.on("request", (r) => requests.push({ url: r.url(), method: r.method(), type: r.headers()["content-type"] ?? "", bytes: r.postDataBuffer()?.length ?? 0 }));
  page.on("pageerror", (e) => console.log(`  [page error] ${e.message}`));
  await page.goto(args.url);
  await page.getByText("Choose a video").waitFor({ timeout: 120_000 });

  const apiResponse = page.waitForResponse((r) => r.url().endsWith("/api/analyze/keypoints"), { timeout: 300_000 });
  await page.setInputFiles('input[type="file"]', join(VIDEOS, file));
  const res = await apiResponse;
  const api = await res.json();
  await page.locator(".tiles, .error-box").first().waitFor({ timeout: 60_000 });
  check(clip, res.status() === 200, `API answered ${res.status()} with ${api.windows?.length ?? 0} windows`);

  // Privacy: no video bytes leave the browser.
  const origin = new URL(args.url).origin;
  const keypointUrl = `${origin}/api/analyze/keypoints`;
  const appPosts = requests.filter((r) => r.method !== "GET" && r.url.startsWith(origin));
  check(clip, appPosts.length === 1 && appPosts[0].url === keypointUrl && appPosts[0].type.includes("application/json"),
    `the app sends exactly one POST: keypoint JSON to /api/analyze/keypoints (${(appPosts[0]?.bytes / 1024).toFixed(0)} KiB)`);
  check(clip, !requests.some((r) => r.url !== keypointUrl && r.bytes > 64 * 1024),
    "no other request anywhere carries more than 64 KiB");
  check(clip, !requests.some((r) => /video\//.test(r.type)), "no request has a video content type");
  const foreignPosts = requests.filter((r) => r.method !== "GET" && !r.url.startsWith(origin));
  const hosts = [...new Set(requests.map((r) => new URL(r.url).host).filter((h) => h && h !== new URL(args.url).host))];
  console.log(`  external hosts contacted: ${hosts.join(", ") || "none"}`);
  if (foreignPosts.length) {
    console.log(`  note: ${foreignPosts.length} POST(s) not from the app: ${[...new Set(foreignPosts.map((r) => `${new URL(r.url).host} (≤${Math.max(...foreignPosts.map((p) => p.bytes))} B)`))].join(", ")}`);
  }

  // Times shown come from the API.
  const starts = api.windows.map((w) => w.start_seconds);
  console.log(`  API windows start at ${starts.join(", ")} s; above threshold ${api.windows_above_threshold}/${api.windows_analyzed}`);
  const barCount = await page.locator(".timeline .bar").count();
  check(clip, barCount === api.windows.length, `timeline has one bar per API window (${barCount})`);
  const tableStarts = await page.locator(".table-view tbody tr td:nth-child(2)").allTextContents();
  check(clip, tableStarts.map(Number).every((s, i) => Math.abs(s - starts[i]) < 0.006), "table start times equal API start_seconds");

  // Jump to a flagged moment (or the first window) and check the video position.
  const flagged = api.windows.filter((w) => w.above_threshold);
  const target = flagged[0] ?? api.windows[0];
  if (flagged.length) await page.locator(".moments button").first().click();
  else await page.locator(".timeline .bar").first().click();
  await page.waitForTimeout(400);
  const t = await page.evaluate(() => document.querySelector(".player video").currentTime);
  check(clip, t >= target.start_seconds - 0.05 && t <= target.start_seconds + 1.0,
    `jump sets video to ${t.toFixed(2)} s (window starts ${target.start_seconds} s)`);

  const warnings = await page.locator(".warning").allTextContents();
  console.log(`  warnings: ${warnings.length ? warnings.map((w) => w.slice(2, 60) + "…").join(" | ") : "none"}`);
  const mainText = await page.locator(".posture dl").textContent();
  check(clip, !/knee/i.test(mainText), "knee bend is not in the main posture summary");
  if (args.shots) await page.screenshot({ path: join(args.shots, `${clip}.png`), fullPage: true });
  await page.close();
}
await browser.close();
console.log(failures.length ? `\n${failures.length} FAILED:\n${failures.join("\n")}` : "\nAll checks passed.");
process.exit(failures.length ? 1 : 0);
