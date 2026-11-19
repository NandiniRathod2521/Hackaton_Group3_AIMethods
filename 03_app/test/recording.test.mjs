// In-browser recording helpers: format choice, minimum length, and the messages shown for
// camera/recording failures. The component itself needs a real camera, so it isn't run here.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import {
  RECORDING_TYPES, cameraErrorMessage, formatElapsed, minRecordSeconds, pickRecordingType,
  recordingFileName, recordingSupportProblem, tooShortMessage, unreadableRecordingMessage,
} from "../src/recording.ts";

const named = (name) => Object.assign(new Error("x"), { name });

describe("recording format", () => {
  test("prefers VP8 WebM, falls back to MP4 (Safari), and needs playback support too", () => {
    assert.equal(pickRecordingType(() => true), "video/webm;codecs=vp8");
    assert.equal(pickRecordingType((t) => t.startsWith("video/mp4")), "video/mp4;codecs=avc1");
    assert.equal(pickRecordingType(() => true, (t) => t === "video/webm"), "video/webm");
    assert.equal(pickRecordingType(() => false), null);
    assert.equal(pickRecordingType(() => true, () => false), null);
  });

  test("skips formats whose recordings could not be decoded", () => {
    const bad = new Set(["video/webm;codecs=vp8"]);
    assert.equal(pickRecordingType(() => true, () => true, bad), "video/webm;codecs=vp9");
    assert.equal(pickRecordingType(() => true, () => true, new Set(RECORDING_TYPES)), null);
    assert.match(unreadableRecordingMessage("video/webm", "MEDIA_ERR_DECODE", true), /MEDIA_ERR_DECODE.*different format/);
    assert.match(unreadableRecordingMessage("video/webm", "MEDIA_ERR_DECODE", false), /Choose a video file/);
  });

  test("file name extension follows the container", () => {
    const at = new Date(2026, 10, 19, 9, 5, 7);
    assert.equal(recordingFileName("video/webm;codecs=vp9", at), "Recording 2026-11-19 09.05.07.webm");
    assert.equal(recordingFileName("video/mp4", at), "Recording 2026-11-19 09.05.07.mp4");
  });
});

describe("minimum length", () => {
  test("is longer than one 48-frame window at 25 fps", () => {
    assert.ok(minRecordSeconds(48 / 25) > 48 / 25);
    assert.match(tooShortMessage(1.234, minRecordSeconds(1.92)), /1\.2 s long; at least 2\.2 s/);
  });
  test("timer format", () => {
    assert.equal(formatElapsed(0), "0:00");
    assert.equal(formatElapsed(65.9), "1:05");
  });
});

describe("messages", () => {
  test("permission denied, no camera, camera busy, other", () => {
    assert.match(cameraErrorMessage(named("NotAllowedError")), /blocked/);
    assert.match(cameraErrorMessage(named("SecurityError")), /blocked/);
    assert.match(cameraErrorMessage(named("NotFoundError")), /No camera/);
    assert.match(cameraErrorMessage(named("OverconstrainedError")), /No camera/);
    assert.match(cameraErrorMessage(named("NotReadableError")), /in use/);
    assert.match(cameraErrorMessage(new Error("boom")), /boom/);
  });
  test("unsupported environments", () => {
    const ok = { isSecureContext: true, hasGetUserMedia: true, hasMediaRecorder: true };
    assert.equal(recordingSupportProblem(ok), null);
    assert.match(recordingSupportProblem({ ...ok, isSecureContext: false }), /https/);
    assert.match(recordingSupportProblem({ ...ok, hasGetUserMedia: false }), /camera/);
    assert.match(recordingSupportProblem({ ...ok, hasMediaRecorder: false }), /record/);
  });
});
