import test from "node:test";
import assert from "node:assert/strict";
import {
  defaultConfig,
  parameterBudget,
  applyDecision,
} from "../lib/project.ts";
test("baseline includes tied embeddings, all transformer matrices, and norms", () => {
  assert.equal(parameterBudget(defaultConfig).total, 24388992);
});
test("untying adds exactly one vocabulary table", () => {
  assert.equal(
    parameterBudget({ ...defaultConfig, tied: false }).total -
      parameterBudget(defaultConfig).total,
    8192 * 384,
  );
});
test("RoPE context length has no learned position parameters", () => {
  assert.equal(
    parameterBudget({ ...defaultConfig, context: 2048 }).total,
    parameterBudget(defaultConfig).total,
  );
});
test("vocabulary and depth decisions change only their intended settings", () => {
  assert.deepEqual(applyDecision(defaultConfig, "vocab", 2), {
    ...defaultConfig,
    vocab: 16384,
  });
  assert.deepEqual(applyDecision(defaultConfig, "layers", 0), {
    ...defaultConfig,
    layers: 8,
  });
  assert.deepEqual(
    applyDecision(defaultConfig, "evaluation", 0),
    defaultConfig,
  );
});
test("the largest vocabulary exceeds the target with the baseline body", () => {
  assert.ok(parameterBudget({ ...defaultConfig, vocab: 32768 }).total > 25e6);
});
