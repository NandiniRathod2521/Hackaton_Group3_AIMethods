#!/usr/bin/env node
// Batch-run the browser MoveNet extractor (dist/ build) in headless Edge/Chrome.
// For each clip it runs (a) MoveNet on the frames decoded by decode_frames.py and
// (b) the real app path on the original video (object URL + 25 fps seeking in the browser).
// Everything stays on this machine: a local static server feeds the browser.
//
//   npm run build
//   node tools/browser_extract.mjs --frames <frames dir> --videos <videos dir> --out prototype_out
//   [--channel msedge|chrome] [--only video|frames]
import { createServer } from "node:http";
import { readFile, mkdir, writeFile, stat } from "node:fs/promises";
import { extname, join, resolve } from "node:path";
import { chromium } from "playwright-core";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const DIST = resolve("dist/client");   // client build (the Worker build is in dist/motion_atlas)
const FRAMES = resolve(args.frames);
const VIDEOS = resolve(args.videos);
const OUT = resolve(args.out ?? "prototype_out");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".jpg": "image/jpeg", ".mp4": "video/mp4", ".mov": "video/quicktime" };

const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const [root, rel] = path.startsWith("/frames/") ? [FRAMES, path.slice(8)]
    : path.startsWith("/videos/") ? [VIDEOS, path.slice(8)]
    : [DIST, path === "/" ? "index.html" : path.slice(1)];
  const file = resolve(root, rel);
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  try {
    await stat(file);
    res.writeHead(200, { "content-type": TYPES[extname(file).toLowerCase()] ?? "application/octet-stream" });
    res.end(await readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  channel: args.channel ?? "msedge",
  headless: true,
  args: ["--enable-unsafe-swiftshader", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") console.log(`[browser ${m.type()}] ${m.text()}`); });
page.on("pageerror", (e) => console.log(`[page error] ${e.message}`));
await page.goto(base);
await page.waitForFunction(() => window.__motionAtlas);
const info = await page.evaluate(() => window.__motionAtlas.info());
console.log("browser:", info);

const clips = JSON.parse(await readFile(join(FRAMES, "clips.json"), "utf8"));
for (const sub of ["movenet_frames", "movenet_video"]) await mkdir(join(OUT, sub), { recursive: true });
const runs = [];
for (const clip of clips) {
  const manifest = JSON.parse(await readFile(join(FRAMES, clip, "manifest.json"), "utf8"));
  const run = { clip, ticks: manifest.ticks };
  if (args.only !== "video") {
    const r = await page.evaluate(([b, n]) => window.__motionAtlas.extractImages(b, n),
      [`${base}/frames/${encodeURIComponent(clip)}`, manifest.ticks]);
    await writeFile(join(OUT, "movenet_frames", `${clip}.keypoints.csv`), r.csv);
    run.frames = { nameMismatches: r.nameMismatches, elapsedMs: r.elapsedMs };
  }
  if (args.only !== "frames") {
    const r = await page.evaluate((u) => window.__motionAtlas.extractVideoUrl(u),
      `${base}/videos/${encodeURIComponent(manifest.source_file)}`);
    await writeFile(join(OUT, "movenet_video", `${clip}.keypoints.csv`), r.csv);
    const { csv, ...meta } = r;
    run.video = meta;
  }
  console.log(`${clip}: frames ${run.frames ? (run.frames.elapsedMs / 1000).toFixed(1) + " s" : "-"}, `
    + `video ${run.video ? `${run.video.width}x${run.video.height}, ${(run.video.elapsedMs / 1000).toFixed(1)} s, `
    + `${run.video.mediaTimes.length} samples (manifest ${manifest.ticks})` : "-"}`);
  runs.push(run);
}
await writeFile(join(OUT, "browser_runs.json"), JSON.stringify({ info, runs }, null, 1));
await browser.close();
server.close();
