// Pure timing/accounting unit tests, not a backend substitute.
import test from "node:test";
import assert from "node:assert/strict";
import {
  BENCHMARK_CONFIG,
  benchmarkPhase,
  canStart,
  validResult,
  benchmarkSettings,
  workloadCompatible,
} from "../lib/benchmark.ts";
test("admin settings validate time bounds and select the correct payload", () => {
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 98304).batchTokens, 98304);
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 262656), null);
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 90000), null);
  assert.equal(benchmarkSettings(-1, 300, "synthetic"), null);
  assert.equal(benchmarkSettings(60, 9, "synthetic"), null);
  assert.equal(benchmarkSettings(60, 3601, "synthetic"), null);
  assert.equal(benchmarkSettings(1.5, 30, "synthetic"), null);
  assert.equal(benchmarkSettings(60, 300, "unknown"), null);
  assert.equal(
    benchmarkSettings(0, 10, "baseline-training").payloadElements,
    24388992,
  );
  assert.equal(benchmarkSettings(600, 3600, "synthetic").measureSeconds, 3600);
});
test("model workloads require matching initial fingerprints", () => {
  const f = "a".repeat(64),
    g = "b".repeat(64);
  assert.equal(
    workloadCompatible({ workload: "baseline-training" }, [
      { device: { workloadFingerprint: f } },
      { device: { workloadFingerprint: f } },
    ]),
    true,
  );
  assert.equal(
    workloadCompatible({ workload: "baseline-training" }, [
      { device: { workloadFingerprint: f } },
      { device: { workloadFingerprint: g } },
    ]),
    false,
  );
  assert.equal(
    workloadCompatible({ workload: "baseline-training" }, [{ device: {} }]),
    false,
  );
});
test("optimized profiles freeze precision and microbatch compatibility", () => {
  const settings = benchmarkSettings(60, 300, "baseline-training", 32768, "optimized-bf16", 32);
  assert.equal(settings.trainingProfile, "optimized-bf16");
  assert.equal(settings.microbatch, 32);
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 32768, "reference-fp32", 4), null);
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 32768, "optimized-fp32", 3), null);
  assert.equal(benchmarkSettings(60, 300, "synthetic", 32768, "optimized-bf16", 4), null);
  const device = {workloadFingerprint: "a".repeat(64), trainingProfile: "optimized-bf16", microbatch: 32};
  assert.equal(workloadCompatible(settings, [{device}]), true);
  assert.equal(workloadCompatible(settings, [{device: {...device, trainingProfile: "reference-fp32"}}]), false);
  assert.equal(workloadCompatible(settings, [{device: {...device, microbatch: 16}}]), false);
  const localStepSettings = benchmarkSettings(60, 300, "baseline-training", 32768, "optimized-bf16", 32, 8);
  assert.equal(localStepSettings.localSteps, 8);
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 32768, "reference-fp32", 1, 8), null);
  assert.equal(benchmarkSettings(60, 300, "baseline-training", 32768, "optimized-bf16", 32, 3), null);
  assert.equal(benchmarkSettings(60, 300, "synthetic", 32768, "reference-fp32", 1, 4), null);
  assert.equal(workloadCompatible(localStepSettings, [{device: {...device, localSteps: 8}}]), true);
  assert.equal(workloadCompatible(localStepSettings, [{device: {...device, localSteps: 1}}]), false);
});
test("compiled BF16 requires a single compiled NVIDIA participant", () => {
  const settings = benchmarkSettings(60, 300, "baseline-training", 32768, "compiled-bf16", 16);
  const device = {kind: "cuda:0", workloadFingerprint: "a".repeat(64), trainingProfile: "compiled-bf16", microbatch: 16, compiledModel: true};
  assert.equal(workloadCompatible(settings, [{device}]), true);
  assert.equal(workloadCompatible(settings, [{device}, {device}]), false);
  assert.equal(workloadCompatible(settings, [{device: {...device, compiledModel: false}}]), false);
  assert.equal(workloadCompatible(settings, [{device: {...device, kind: "rocm:0"}}]), false);
});
test("the exact warmup and measurement boundaries", () => {
  const s = { startsAt: 100000, cancelledAt: null, config: BENCHMARK_CONFIG };
  assert.equal(benchmarkPhase(s, 99999), "countdown");
  assert.equal(benchmarkPhase(s, 100000), "warmup");
  assert.equal(benchmarkPhase(s, 159999), "warmup");
  assert.equal(benchmarkPhase(s, 160000), "measuring");
  assert.equal(benchmarkPhase(s, 459999), "measuring");
  assert.equal(benchmarkPhase(s, 460000), "finished");
});
test("an offline or non-ready agent blocks Start", () => {
  const s = {
    startsAt: null,
    cancelledAt: null,
    config: BENCHMARK_CONFIG,
    coordinatorSeenAt: 100000,
    expected: ["a", "b"],
  };
  const a = [
    { userId: "a", ready: true, requestedReady: true, lastSeenAt: 100000 },
    { userId: "b", ready: true, requestedReady: true, lastSeenAt: 100000 },
  ];
  assert.equal(canStart(s, a, 100001), true);
  assert.equal(canStart(s, a.slice(1), 100001), false);
  assert.equal(canStart(s, [a[0], { ...a[1], ready: false }], 100001), false);
  assert.equal(canStart(s, a, 120001), false);
});
test("reject malformed and nonfinite agent reports", () => {
  assert.equal(
    validResult({ protocol: 1, complete: true, measuredRounds: Infinity }),
    false,
  );
  assert.equal(validResult(null), false);
});
