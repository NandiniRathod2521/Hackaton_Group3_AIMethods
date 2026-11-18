# Motion Atlas app

Video-first movement analysis. You pick a video and the browser extracts one body pose per frame with MoveNet SinglePose Lightning, at 25 fps. Only the keypoints are sent. They go to the Cloudflare Worker, which forwards them to the existing Render model API (`POST /predict/keypoints`). The page then shows:

- the per-window scores on a timeline;
- the flagged moments, using the start and end times returned by the API;
- the local video with the analysed skeleton drawn over it;
- the input-quality warnings.

**The raw video never leaves the browser.** There's no D1 history yet. IMU CSV analysis isn't in this app yet.

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
npm run dev                           # Vite + the Worker (workerd) on one origin
```

MoveNet weights are fetched from the model provider (tfhub.dev, which redirects to Kaggle storage) when the page loads.

## Deploy (not done yet)

1. Redeploy the Render service, so the window start-time fix and `end_seconds` are live.
2. Set `RENDER_API_URL` in `wrangler.jsonc` to the Render URL.
3. Run `npm run deploy`.

If the Render service is still on the old code, the jump-to-moment times will be wrong whenever windows are dropped.

## Checks

```powershell
python -m pytest 02_model_server/test_window_starts.py -q            # API window start times
cd 03_app; npm run build                                             # typecheck app + Worker, build
npm run dev   # then, in another terminal:
node tools/e2e_video_flow.mjs --url http://127.0.0.1:5173 --videos "<videos dir>" --clips "4.03.49,4.03.52"
```

The end-to-end script runs in headless Edge and checks four things:
- the app sends exactly one POST, containing keypoint JSON, and no request carries video;
- the timeline and table use the API's start times;
- jump-to-moment moves the video to the right time;
- knee bend isn't in the main summary.

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
