// pose-detection imports @mediapipe/pose for its BlazePose-MediaPipe runtime, which we do not use.
// That package is a global script without ES exports, so it breaks bundling; this stub replaces it.
export class Pose {
  constructor() {
    throw new Error("BlazePose (MediaPipe runtime) is not bundled; Motion Atlas uses MoveNet.");
  }
}
export const VERSION = "stub";
