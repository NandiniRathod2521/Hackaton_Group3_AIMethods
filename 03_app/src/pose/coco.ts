// Contract with the trained keypoint model and the Render /predict/keypoints endpoint.
// Source of truth: 02_model_server/keypoint_utils.py and models/keypoint/config.json.

/** COCO-17 order used by the training CSVs, keypoint_utils.py and the API. */
export const COCO_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
] as const;

export const N_KP = 17;
export const FPS = 25;              // model frame rate; one sample every 40 ms
export const WINDOW = 48;           // frames per scored window (1.92 s), non-overlapping at inference
// Which windows are scored (and their start times) is decided by the server; the app uses the
// start_seconds/end_seconds it returns rather than re-implementing the filtering.

/** One frame: 17 × [x px, y px, confidence 0..1]. A frame with no person is all zeros. */
export type Frame = [number, number, number][];

export const EMPTY_FRAME = (): Frame => Array.from({ length: N_KP }, () => [0, 0, 0] as [number, number, number]);

/** VTT-ConIoT keypoint CSV, the format read by keypoint_utils.read_keypoints. */
export function framesToCsv(frames: Frame[], detection: number[]): string {
  const header = ["frame_number", "timestamp"];
  for (let k = 0; k < N_KP; k++) header.push(`x${k}`, `y${k}`, `prob${k}`);
  header.push("detection_score");
  const lines = [header.join(",")];
  frames.forEach((f, t) => {
    const row: (string | number)[] = [t, Math.floor((t * 1000) / FPS)];
    for (const [x, y, p] of f) row.push(x.toFixed(4), y.toFixed(4), p.toFixed(4));
    row.push(detection[t].toFixed(4));
    lines.push(row.join(","));
  });
  return lines.join("\n") + "\n";
}
