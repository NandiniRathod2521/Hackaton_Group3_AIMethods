# Motion Atlas app

Video-first movement analysis. You pick a video and the browser extracts one body pose per frame with MoveNet SinglePose Lightning, at 25 fps. Only the keypoints are sent. They go to the Cloudflare Worker, which forwards them to the existing Render model API (`POST /predict/keypoints`). The page then shows:

- the per-window scores on a timeline;
- the flagged moments, using the start and end times returned by the API;
- the local video with the analysed skeleton drawn over it;
- the input-quality warnings.

**The raw video never leaves the browser.** IMU CSV analysis isn't in this app yet.

## History (Cloudflare D1)

After each successful analysis the app saves a summary to a D1 database (binding `DB`) and lists recent saved analyses in the **History** section. Opening a row restores its summary, score timeline and flagged-moment times. To play those moments, open your local copy of the video; it's played in the browser only and isn't uploaded or re-analysed.

**What is stored** (`migrations/0001_create_analysis_history.sql`): analysis type, creation time, clip duration, window count, flagged-window count and share, threshold, model name, the interpretation sentence, the posture summary numbers, and per window: index, start/end time, score, and flag.
**What is never stored:** video, keypoints, raw IMU samples, file names. The Worker rejects save requests that contain any other field. Each record shows the generic label "Video analysis".

**Scoping (demo only, not authentication).** On first visit the browser generates a random ID and keeps it in `localStorage` (`motion-atlas.browser-id`). It's sent as an `X-Browser-Id` header, and every history query is limited to that ID. There's no endpoint that lists every visitor's history. However:
- this is **not a login**. Anyone who gets hold of a browser's ID can read and delete that browser's history;
- clearing site data, or switching browser or device, starts an empty history (the old records stay in D1 until they're pruned);
- **don't analyse confidential footage or sensitive personal data with this prototype.**

**Limits.** The list returns the 20 newest records (`?limit=` accepts 1–50). Each browser keeps its newest 100 records; older ones are deleted when a new one is saved. A save is at most 2000 windows and 256 KiB.

**If saving fails**, the analysis stays on screen with a "not saved" message and a **Try saving again** button. History errors never affect `/api/analyze/keypoints`.

| Route | Purpose |
| --- | --- |
| `GET /api/history?limit=20` | recent analyses for this browser (summary only) |
| `POST /api/history` | save one analysis summary |
| `GET /api/history/<id>` | one saved analysis with its per-window scores |
| `DELETE /api/history/<id>` | delete one saved analysis |

All four need the `X-Browser-Id` header. Another browser's record answers 404.

## Version 1 limitations shown in the app

- One pose per frame, with no reliable tracking of a single person. When other people are in view, the analysed pose may switch between them. The app warns when the pose jumps suddenly.
- It warns when knees and ankles aren't clearly detected, or when no person is found in parts of the video.
- Knee bend appears only as an experimental MoveNet estimate. On the test clips MoveNet reported about 5–6° more knee bend than YOLO, and the training data's detector is unknown.

## Local development

```powershell
# terminal 1, from the repository root: model API (same code as Render)
python -m uvicorn app:app --app-dir 02_model_server --host 127.0.0.1 --port 8000

# terminal 2
cd 03_app
npm install
copy .dev.vars.example .dev.vars      # RENDER_API_URL=http://127.0.0.1:8000
npm run db:migrate:local              # create the local D1 tables (once, and after adding a migration)
npm run dev                           # Vite + the Worker (workerd) on one origin
```

MoveNet weights are fetched from the model provider (tfhub.dev, which redirects to Kaggle storage) when the page loads.

### Local D1

`npm run dev` and `wrangler d1 ... --local` share the same local SQLite copy in `03_app/.wrangler/state` (ignored by Git). The `database_id` placeholder in `wrangler.jsonc` is enough for local work; no Cloudflare account is needed.

```powershell
npm run db:migrate:local                                                   # = npx wrangler d1 migrations apply DB --local
npx wrangler d1 migrations list DB --local                                 # show applied / pending migrations
npx wrangler d1 execute DB --local --command "SELECT id, created_at, window_count, flagged_count FROM analysis_history ORDER BY created_at DESC LIMIT 5"
```

To reset local history, stop `npm run dev`, delete `03_app/.wrangler/state`, and run `npm run db:migrate:local` again.

Add schema changes as new files (`migrations/0002_<name>.sql`, …). Don't edit an applied migration. Create one with `npx wrangler d1 migrations create DB <name>`.

## Deploy (not done yet)

1. Redeploy the Render service, so the window start-time fix and `end_seconds` are live.
2. Set `RENDER_API_URL` in `wrangler.jsonc` to the Render URL.
3. Create the production D1 database (once):
   ```powershell
   cd 03_app
   npx wrangler login
   npx wrangler d1 create motion-atlas-history
   ```
   Copy the printed `database_id` (a UUID) into `wrangler.jsonc` → `d1_databases[0].database_id`, replacing `REPLACE_WITH_D1_DATABASE_ID`. Keep `binding: "DB"` and `database_name: "motion-atlas-history"`. You can also create the database in the Cloudflare dashboard (**Storage & Databases → D1 SQL Database → Create**, name `motion-atlas-history`) and copy its ID from the database page.
4. Apply migrations to the production database:
   ```powershell
   npm run db:migrate:remote          # = npx wrangler d1 migrations apply DB --remote
   npx wrangler d1 migrations list DB --remote    # should list nothing left to apply
   ```
5. Run `npm run deploy`. Wrangler attaches the `DB` binding from `wrangler.jsonc`; you don't need to add it in the dashboard.
6. Check: open the site, analyse a clip, reload, and confirm it appears under History. In the dashboard, **Workers & Pages → motion-atlas → Settings → Bindings** should show `DB → motion-atlas-history`.

Apply new migrations with `npm run db:migrate:remote` **before** deploying code that needs them.

If the Render service is still on the old code, the jump-to-moment times will be wrong whenever windows are dropped.

## Checks

```powershell
python -m pytest 02_model_server/test_window_starts.py -q            # API window start times
cd 03_app; npm run build                                             # typecheck app + Worker, build
npm test                                                             # D1 history: migration, routes, scoping, failures
npm run db:migrate:local; npm run dev   # then, in another terminal:
node tools/e2e_video_flow.mjs --url http://127.0.0.1:5173 --videos "<videos dir>" --clips "4.03.49,4.03.52"
node tools/e2e_video_flow.mjs --url http://127.0.0.1:5173 --videos "<videos dir>" --clips "4.03.49" --fail-save 1
```

`npm test` uses Node's test runner with a real local D1 (Miniflare, via wrangler). It needs Node 22.6 or newer, and no Cloudflare account. It covers:
- the migration: columns, indexes, constraints, and that `wrangler d1 migrations apply --local` applies it once;
- save, list, load and delete, including limits, validation, and rejection of raw-data fields;
- browser scoping;
- the analysis route still working when D1 is missing or failing;
- the app's history client, including a failed save being reported rather than thrown.

The end-to-end script runs in headless Edge and checks:
- the app sends exactly one analysis POST, containing keypoint JSON, plus one small history save without keypoints or the file name; no request carries video;
- the timeline and table use the API's start times;
- jump-to-moment moves the video to the right time;
- knee bend isn't in the main summary;
- after a reload, the analysis is in History, and opening it restores the timeline and flagged moments;
- with `--fail-save 1`, the analysis stays on screen and a "not saved" message is shown.

## MoveNet vs YOLO comparison (prototype)

Keep frames and outputs out of Git (`prototype_out/` is ignored).

```powershell
# from the repository root
python 03_app/tools/decode_frames.py "<videos dir>" --out "<scratch>/frames"           # converter's exact 25 fps frames
python 03_app/tools/yolo_frames.py "<scratch>/frames" --out 03_app/prototype_out/yolo_frames_640
python 03_app/tools/yolo_frames.py "<scratch>/frames" --out 03_app/prototype_out/yolo_frames_960 --imgsz 960
cd 03_app; npm run build
node tools/browser_extract.mjs --frames "<scratch>/frames" --videos "<videos dir>" --out prototype_out
cd ..
python 03_app/tools/compare_extractors.py --frames "<scratch>/frames"
python 03_app/tools/compare_extractors.py --frames "<scratch>/frames" --exclude 4.03.44 4.03.47 4.03.35 --tag _single_person
```

YOLO is a comparison point, not ground truth. The numerical checks are prototype checks; they don't validate the anomaly scores.
