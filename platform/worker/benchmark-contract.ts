export const BENCHMARK_CONFIG = {
  protocol: 3,
  payloadElements: 25_000_000,
  matrixSize: 1024,
  warmupSeconds: 60,
  measureSeconds: 300,
  workload: "synthetic" as "synthetic" | "baseline-training",
  trainingTokens: 500_000_000,
  batchTokens: 32768,
  trainingProfile: "reference-fp32" as "reference-fp32" | "optimized-fp32" | "optimized-fp16" | "optimized-bf16" | "compiled-bf16",
  microbatch: 1,
  localSteps: 1,
  model: {
    vocab_size: 8192,
    width: 384,
    layers: 12,
    heads: 6,
    expansion: 4,
    context: 512,
    tied: true,
  },
};
export function benchmarkSettings(
  warmup: unknown,
  measure: unknown,
  workload: unknown,
  batchTokens: unknown = 32768,
  trainingProfile: unknown = "reference-fp32",
  microbatch: unknown = 1,
  localSteps: unknown = 1,
) {
  if (
    typeof batchTokens !== "number" ||
    !Number.isInteger(batchTokens) ||
    batchTokens < 512 ||
    batchTokens > 262144 ||
    batchTokens % 512 !== 0 ||
    typeof warmup !== "number" ||
    !Number.isInteger(warmup) ||
    warmup < 0 ||
    warmup > 600 ||
    typeof measure !== "number" ||
    !Number.isInteger(measure) ||
    measure < 10 ||
    measure > 3600 ||
    !["synthetic", "baseline-training"].includes(String(workload)) ||
    !["reference-fp32", "optimized-fp32", "optimized-fp16", "optimized-bf16", "compiled-bf16"].includes(String(trainingProfile)) ||
    ![1, 2, 4, 8, 16, 32, 64].includes(microbatch as number) ||
    ![1, 2, 4, 8, 16, 32].includes(localSteps as number) ||
    (trainingProfile === "reference-fp32" && microbatch !== 1) ||
    (trainingProfile === "reference-fp32" && localSteps !== 1) ||
    (trainingProfile === "optimized-fp16" && localSteps !== 1) ||
    (workload === "synthetic" && trainingProfile !== "reference-fp32") ||
    (workload === "synthetic" && localSteps !== 1)
  )
    return null;
  return {
    ...BENCHMARK_CONFIG,
    warmupSeconds: warmup,
    measureSeconds: measure,
    workload: workload as "synthetic" | "baseline-training",
    batchTokens: batchTokens as number,
    trainingProfile: trainingProfile as typeof BENCHMARK_CONFIG.trainingProfile,
    microbatch: microbatch as number,
    localSteps: localSteps as number,
    payloadElements: workload === "baseline-training" ? 24_388_992 : 25_000_000,
  };
}
export const ONLINE_MS = 20_000;
export function workloadCompatible(
  config: { workload?: string; trainingProfile?: string; microbatch?: number; localSteps?: number },
  agents: { device: { workloadFingerprint?: unknown; trainingProfile?: string; microbatch?: number; localSteps?: number; compiledModel?: boolean; kind?: string } }[],
) {
  if (config.workload !== "baseline-training") return true;
  if (config.trainingProfile === "compiled-bf16" && (
    agents.length !== 1 || agents[0].device.compiledModel !== true || !agents[0].device.kind?.startsWith("cuda:")
  )) return false;
  const fingerprints = agents.map((a) => a.device.workloadFingerprint);
  return (
    fingerprints.length > 0 &&
    fingerprints.every(
      (f) =>
        typeof f === "string" &&
        /^[a-f0-9]{64}$/.test(f) &&
        f === fingerprints[0],
    ) && agents.every((a) =>
      (a.device.trainingProfile ?? "reference-fp32") === (config.trainingProfile ?? "reference-fp32") &&
      (a.device.microbatch ?? 1) === (config.microbatch ?? 1) &&
      (a.device.localSteps ?? 1) === (config.localSteps ?? 1)
    )
  );
}
export type Timing = {
  startsAt: number | null;
  cancelledAt: number | null;
  config: { warmupSeconds: number; measureSeconds: number };
};
export function benchmarkPhase(session: Timing, now: number) {
  if (session.cancelledAt !== null) return "cancelled";
  if (session.startsAt === null) return "lobby";
  if (now < session.startsAt) return "countdown";
  if (now < session.startsAt + session.config.warmupSeconds * 1000)
    return "warmup";
  if (
    now <
    session.startsAt +
      (session.config.warmupSeconds + session.config.measureSeconds) * 1000
  )
    return "measuring";
  return "finished";
}
export function canStart(
  session: Timing & { coordinatorSeenAt: number; expected: string[] },
  agents: {
    userId: string;
    ready: boolean;
    lastSeenAt: number;
    requestedReady: boolean;
  }[],
  now: number,
) {
  return (
    benchmarkPhase(session, now) === "lobby" &&
    session.coordinatorSeenAt > now - ONLINE_MS &&
    session.expected.length >= 1 &&
    session.expected.length <= 10 &&
    session.expected.every((id) =>
      agents.some(
        (a) =>
          a.userId === id &&
          a.ready &&
          a.requestedReady &&
          a.lastSeenAt > now - ONLINE_MS,
      ),
    )
  );
}
export function validResult(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const r = value as Record<string, unknown>;
  const finite = (x: unknown) =>
    typeof x === "number" && Number.isFinite(x) && x >= 0;
  if (
    r.workload === "baseline-training" &&
    (![
      "processedTokens",
      "tokensPerSecond",
      "localBatchTokens",
      "parameterCount",
      "optimizerSteps",
      "meanLoss",
    ].every((k) => finite(r[k])) ||
      !Number.isInteger(r.processedTokens) ||
      !Number.isInteger(r.localBatchTokens))
  )
    return false;
  return (
    r.protocol === 3 &&
    typeof r.complete === "boolean" &&
    [
      "measuredRounds",
      "warmupRounds",
      "failedRounds",
      "boundaryRounds",
      "measuredSeconds",
      "latencyP50Ms",
      "latencyP95Ms",
      "gemmGflops",
      "clockUncertaintyMs",
    ].every((k) => finite(r[k])) &&
    Number.isInteger(r.measuredRounds) &&
    Number.isInteger(r.failedRounds) &&
    finite(r.meanRoundMs) &&
    typeof r.note === "string" &&
    r.note.length <= 1000
  );
}
