// Draft schedule planner: suggestions follow only the user's inputs; editing is immutable.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import {
  addBlock, dayTotals, formatTime, generateDay, generatePlan, moveBlock, orderTasks, parseTime, planToText,
  removeBlock, updateBlock, validate, withTimes,
} from "../src/schedule.ts";

const S = { startMin: 7 * 60, endMin: 15 * 60 + 30, shortBreakMin: 10, workStretchMin: 90, mealBreakMin: 30, alternate: false };
const task = (id, name, minutes, demanding = false) => ({ id, name, minutes, demanding });
const kinds = (p) => p.blocks.map((b) => b.kind).join(",");

describe("time helpers", () => {
  test("parse / format", () => {
    assert.equal(parseTime("07:30"), 450);
    assert.equal(parseTime("7:05"), 425);
    assert.equal(parseTime("24:00"), null);
    assert.equal(parseTime("x"), null);
    assert.equal(formatTime(930), "15:30");
  });
});

describe("validate", () => {
  test("plain-language problems", () => {
    assert.deepEqual(validate(S, [task("a", "Dig", 60)]), []);
    assert.match(validate({ ...S, endMin: S.startMin }, [task("a", "Dig", 60)])[0], /later than the start/);
    assert.match(validate(S, [task("a", "  ", 60)])[0], /at least one task/);
    assert.match(validate({ ...S, workStretchMin: 5 }, [task("a", "Dig", 60)])[0], /15 minutes/);
  });
});

describe("generateDay", () => {
  test("breaks after each work stretch, one meal near the middle, open time at the end", () => {
    const p = generateDay("Day", S, [task("a", "Formwork", 180), task("b", "Rebar", 120), task("c", "Cleanup", 60)]);
    const t = withTimes(p.blocks, S.startMin);
    // no work stretch longer than 90 min
    let run = 0;
    for (const b of t) { run = b.kind === "task" ? run + b.minutes : 0; assert.ok(run <= 90, `stretch ${run}`); }
    const meals = t.filter((b) => b.kind === "meal");
    assert.equal(meals.length, 1);
    const mid = (S.startMin + S.endMin) / 2;
    assert.ok(Math.abs(meals[0].start - mid) <= 90, `meal at ${formatTime(meals[0].start)}`);
    assert.equal(t.at(-1).end, S.endMin, "fills exactly to the shift end");
    assert.equal(p.blocks.filter((b) => b.kind === "task").reduce((a, b) => a + b.minutes, 0), 360, "all task time kept");
    assert.deepEqual(p.unscheduled, []);
    assert.ok(p.blocks.some((b) => /Formwork \(part 1 of 2\)/.test(b.label)), "long task split and labelled");
  });

  test("never ends on a break; leftover becomes open time", () => {
    const p = generateDay("Day", S, [task("a", "Survey", 90)]);
    assert.equal(kinds(p), "task,open");
  });

  test("task time that doesn't fit is reported, not dropped silently", () => {
    const p = generateDay("Day", { ...S, endMin: S.startMin + 120 }, [task("a", "Pour", 100), task("b", "Finish", 60)]);
    assert.ok(dayTotals(p.blocks, { ...S, endMin: S.startMin + 120 }).overrunMin === 0);
    assert.equal(p.unscheduled.reduce((a, u) => a + u.minutes, 0) + p.blocks.filter((b) => b.kind === "task").reduce((a, b) => a + b.minutes, 0), 160);
    assert.ok(p.unscheduled.length >= 1);
  });

  test("no meal break when set to 0", () => {
    const p = generateDay("Day", { ...S, mealBreakMin: 0 }, [task("a", "Dig", 400)]);
    assert.ok(!p.blocks.some((b) => b.kind === "meal"));
  });
});

describe("orderTasks", () => {
  test("alternates demanding and lighter tasks, keeping each group's order", () => {
    const ts = [task("1", "Lift A", 30, true), task("2", "Lift B", 30, true), task("3", "Check", 30), task("4", "Log", 30)];
    assert.deepEqual(orderTasks(ts, true).map((t) => t.name), ["Lift A", "Check", "Lift B", "Log"]);
    assert.deepEqual(orderTasks(ts, false).map((t) => t.name), ["Lift A", "Lift B", "Check", "Log"]);
  });
});

describe("week + editing", () => {
  test("one plan per chosen day, independent blocks", () => {
    const plan = generatePlan(["Mon", "Wed"], S, [task("a", "Dig", 120)]);
    assert.deepEqual(plan.map((d) => d.day), ["Mon", "Wed"]);
    assert.notEqual(plan[0].blocks[0].id, plan[1].blocks[0].id);
  });

  test("change, move, remove, add — without mutating", () => {
    const p = generateDay("Day", S, [task("a", "Dig", 120), task("b", "Haul", 60)]);
    const before = JSON.stringify(p.blocks);
    const id = p.blocks[0].id;
    const changed = updateBlock(p.blocks, id, { label: "Dig trench", minutes: 45 });
    assert.equal(changed[0].label, "Dig trench");
    assert.equal(withTimes(changed, S.startMin)[1].start, S.startMin + 45, "later blocks move");
    assert.equal(moveBlock(p.blocks, id, 1)[1].id, id);
    assert.equal(moveBlock(p.blocks, id, -1), p.blocks, "can't move past the start");
    assert.ok(!removeBlock(p.blocks, id).some((b) => b.id === id));
    assert.equal(addBlock(p.blocks, "break", 10).at(-1).kind, "break");
    assert.equal(JSON.stringify(p.blocks), before);
  });

  test("overrun is reported when edits run past the shift end", () => {
    const p = generateDay("Day", S, [task("a", "Dig", 120)]);
    const longer = addBlock(p.blocks, "task", 60);
    assert.equal(dayTotals(longer, S).overrunMin, 60);
  });

  test("text copy is labelled as a draft", () => {
    const txt = planToText(generatePlan(["Mon"], S, [task("a", "Dig", 60)]), S);
    assert.match(txt, /draft, not a final or approved schedule/);
    assert.match(txt, /Monday · shift 07:00–15:30/);
    assert.match(txt, /07:00–08:00 {2}Dig/);
  });
});
