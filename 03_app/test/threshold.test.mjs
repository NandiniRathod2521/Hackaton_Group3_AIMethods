// Exploratory threshold: display-only re-classification of already-scored windows.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { clampThreshold, classify, decimalsFor, isTrained, niceCeil, sliderRange } from "../src/threshold.ts";
import { sampleResult } from "./helpers.mjs";

const TRAINED = 0.6588426828384399;

describe("classify", () => {
  test("at the trained threshold uses the server's flags exactly", () => {
    // A server flag that disagrees with score > threshold must still win at the trained value.
    const windows = [{ reconstruction_score: TRAINED, above_threshold: true }, { reconstruction_score: 0.9, above_threshold: true }];
    assert.deepEqual(classify(windows, TRAINED, TRAINED).flagged, [true, true]);
    const r = sampleResult();
    const c = classify(r.windows, r.threshold, r.threshold);
    assert.equal(c.count, r.windows_above_threshold);
    assert.equal(c.percent.toFixed(2), r.percentage_above_threshold.toFixed(2));
  });

  test("other thresholds apply the server rule score > threshold", () => {
    const r = sampleResult();   // scores 0.31, 0.77, 0.12
    assert.deepEqual(classify(r.windows, 0.2, r.threshold).flagged, [true, true, false]);
    assert.deepEqual(classify(r.windows, 0.31, r.threshold).flagged, [false, true, false], "equal is not above");
    assert.equal(classify(r.windows, 1, r.threshold).count, 0);
    assert.equal(classify(r.windows, 0, r.threshold).percent, 100);
  });

  test("does not modify the result", () => {
    const r = sampleResult();
    const before = JSON.stringify(r);
    classify(r.windows, 0.1, r.threshold);
    assert.equal(JSON.stringify(r), before);
  });

  test("empty", () => assert.deepEqual(classify([], 1, 1), { flagged: [], count: 0, percent: 0 }));
});

describe("slider range", () => {
  test("covers every score and the trained threshold, ending on a readable number", () => {
    const r = sliderRange([0.2, 1.7, 0.4], TRAINED);
    assert.equal(r.min, 0);
    assert.equal(r.max, 2);
    assert.ok(r.max >= TRAINED && r.step > 0 && r.step <= r.max / 100);
    assert.ok(sliderRange([], TRAINED).max >= TRAINED * 1.5);
  });
  test("niceCeil, clamp, decimals, isTrained", () => {
    assert.deepEqual([0.66, 0.99, 1.7, 2.2, 3, 7, 12].map(niceCeil), [1, 1, 2, 2.5, 5, 10, 20]);
    const r = { min: 0, max: 2, step: 0.004 };
    assert.equal(clampThreshold(5, r), 2);
    assert.equal(clampThreshold(-1, r), 0);
    assert.equal(clampThreshold(NaN, r), 0);
    assert.equal(decimalsFor(0.004), 4);
    assert.equal(decimalsFor(0.5), 2);
    assert.ok(isTrained(TRAINED, TRAINED) && !isTrained(0.66, TRAINED));
  });
});
