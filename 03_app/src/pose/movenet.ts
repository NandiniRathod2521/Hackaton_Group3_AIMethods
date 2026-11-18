import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-converter";
import * as poseDetection from "@tensorflow-models/pose-detection";
import { COCO_NAMES, EMPTY_FRAME, type Frame } from "./coco";

export type Detector = poseDetection.PoseDetector;
export type DetectorInput = HTMLCanvasElement | ImageBitmap | HTMLImageElement;

export const EXTRACTOR = "movenet-singlepose-lightning-v4";

/**
 * MoveNet SinglePose Lightning, weights from the model provider (tfhub.dev -> Kaggle).
 * Smoothing is off: YOLO has no temporal filter and the autoencoder should see raw motion.
 * The library still carries a crop region from frame to frame (MoveNet's own tracking);
 * call detector.reset() between clips. Poses scoring < 0.25 (library default) are dropped
 * and written as an empty frame, like the YOLO converter does when nobody is detected.
 */
export async function createDetector(): Promise<{ detector: Detector; backend: string }> {
  await tf.setBackend("webgl").catch(() => false);
  await tf.ready();
  const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    enableSmoothing: false,
  });
  return { detector, backend: tf.getBackend() };
}

export interface FrameResult {
  frame: Frame;
  detection: number;      // MoveNet pose score (0 when no pose)
  nameMismatch: boolean;  // keypoint names not in COCO_NAMES order
}

export async function detectFrame(detector: Detector, input: DetectorInput, timestampMs: number): Promise<FrameResult> {
  const poses = await detector.estimatePoses(input, undefined, timestampMs);
  if (!poses.length) return { frame: EMPTY_FRAME(), detection: 0, nameMismatch: false };
  const pose = poses[0];
  let nameMismatch = pose.keypoints.length !== COCO_NAMES.length;
  const frame: Frame = pose.keypoints.map((kp, i) => {
    if (kp.name !== COCO_NAMES[i]) nameMismatch = true;
    const score = Math.min(1, Math.max(0, kp.score ?? 0));
    return [kp.x, kp.y, score];
  });
  return { frame, detection: pose.score ?? 0, nameMismatch };
}
