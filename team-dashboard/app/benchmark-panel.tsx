"use client";
import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  Check,
  Clock,
  Download,
  Play,
  RefreshCw,
  Users,
} from "lucide-react";
import { apiFetch, teamKey } from "../lib/client-api";
type Result = {
  complete: boolean;
  measuredRounds: number;
  warmupRounds: number;
  failedRounds: number;
  boundaryRounds: number;
  measuredSeconds: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  meanRoundMs: number;
  gemmGflops: number;
  clockUncertaintyMs: number;
  note: string;
  workload?: string;
  tokensPerSecond?: number;
  processedTokens?: number;
  localBatchTokens?: number;
  workShare?: number;
  warmupTokensPerSecond?: number;
  activeParticipants?: number;
  droppedParticipants?: string[];
  parameterCount?: number;
  trainingProfile?: string;
  precision?: string;
  microbatch?: number;
  localSteps?: number;
  compiledModel?: boolean;
  compileSetupSeconds?: number;
};
type Agent = {
  userId: string;
  instanceId: string;
  device: {
    name: string;
    kind: string;
    torch: string;
    platform: string;
    hostname?: string;
    trainingProfile?: string;
    microbatch?: number;
    localSteps?: number;
  };
  lastSeenAt: number;
  requestedReady: boolean;
  ready: boolean;
  result: Result | null;
};
type Session = {
  id: string;
  phase: string;
  canStart: boolean;
  coordinatorSeenAt: number;
  expected: string[];
  startsAt: number | null;
  config: {
    warmupSeconds: number;
    measureSeconds: number;
    payloadElements: number;
    matrixSize: number;
    workload?: "synthetic" | "baseline-training";
    trainingTokens?: number;
    batchTokens?: number;
    trainingProfile?: string;
    microbatch?: number;
    localSteps?: number;
  };
};
type State = {
  serverTime: number;
  user: { id: string; role: string };
  session: Session | null;
  agents: Agent[];
};
const fmt = (n: number) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(n);
const duration = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.ceil(seconds % 60)).padStart(2, "0")}`;
export default function BenchmarkPanel() {
  const [draftWarmup, setDraftWarmup] = useState(60),
    [draftMeasure, setDraftMeasure] = useState(300);
  const [draftWorkload, setDraftWorkload] = useState("synthetic");
  const [draftProfile, setDraftProfile] = useState("reference-fp32");
  const [draftMicrobatch, setDraftMicrobatch] = useState(1);
  const [draftLocalSteps, setDraftLocalSteps] = useState(1);
  const [draftBatchTokens, setDraftBatchTokens] = useState(32768);
  const [data, setData] = useState<State | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [offset, setOffset] = useState(0),
    [now, setNow] = useState(Date.now());
  const load = useCallback(async () => {
    if (!teamKey()) {
      setData(null);
      return;
    }
    try {
      const sent = Date.now();
      const response = await apiFetch("/api/benchmark");
      const body = (await response.json()) as State & { error?: string };
      if (!response.ok)
        throw new Error(body.error ?? "Could not load benchmark");
      setOffset(body.serverTime - (sent + Date.now()) / 2);
      setData(body);
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    void load();
    const interval = setInterval(() => void load(), 2000),
      clock = setInterval(() => setNow(Date.now()), 250);
    window.addEventListener("glm-auth", load);
    return () => {
      clearInterval(interval);
      clearInterval(clock);
      window.removeEventListener("glm-auth", load);
    };
  }, [load]);
  const session = data?.session,
    serverNow = now + offset,
    started = session?.startsAt ?? 0,
    warmEnd = started + (session?.config.warmupSeconds ?? 60) * 1000,
    finish = warmEnd + (session?.config.measureSeconds ?? 300) * 1000;
  useEffect(() => {
    setDraftWarmup(session?.config.warmupSeconds ?? 60);
    setDraftMeasure(session?.config.measureSeconds ?? 300);
    setDraftWorkload(session?.config.workload ?? "synthetic");
    setDraftProfile(session?.config.trainingProfile ?? "reference-fp32");
    setDraftMicrobatch(session?.config.microbatch ?? 1);
    setDraftLocalSteps(session?.config.localSteps ?? 1);
    setDraftBatchTokens(session?.config.batchTokens ?? 32768);
  }, [
    session?.id,
    session?.config.warmupSeconds,
    session?.config.measureSeconds,
    session?.config.workload,
    session?.config.trainingProfile,
    session?.config.microbatch,
    session?.config.localSteps,
    session?.config.batchTokens,
  ]);
  const warmup = session?.config.warmupSeconds ?? 60,
    measurement = session?.config.measureSeconds ?? 300;
  const training = session?.config.workload === "baseline-training";
  const compiled = session?.config.trainingProfile === "compiled-bf16";
  const precision = session?.config.trainingProfile?.endsWith("bf16") ? "BF16 mixed precision" : "FP32";
  const settingsDirty =
    draftWarmup !== warmup ||
    draftMeasure !== measurement ||
    draftWorkload !== (session?.config.workload ?? "synthetic") ||
    draftProfile !== (session?.config.trainingProfile ?? "reference-fp32") ||
    draftMicrobatch !== (session?.config.microbatch ?? 1) ||
    draftLocalSteps !== (session?.config.localSteps ?? 1) ||
    draftBatchTokens !== (session?.config.batchTokens ?? 32768);
  const phase = session?.phase ?? "no lobby",
    coordinator = data?.user.role === "coordinator";
  const remaining =
    phase === "countdown"
      ? Math.max(0, (started - serverNow) / 1000)
      : phase === "warmup"
        ? Math.max(0, (warmEnd - serverNow) / 1000)
        : phase === "measuring"
          ? Math.max(0, (finish - serverNow) / 1000)
          : 0;
  const progress =
    phase === "warmup"
      ? Math.min(
          100,
          Math.max(0, (serverNow - started) / Math.max(1, warmup * 10)),
        )
      : phase === "measuring"
        ? Math.min(100, Math.max(0, (serverNow - warmEnd) / (measurement * 10)))
        : phase === "finished"
          ? 100
          : 0;
  async function action(action: string, extra: object = {}) {
    setBusy(true);
    try {
      const response = await apiFetch("/api/benchmark", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, sessionId: session?.id, ...extra }),
      });
      const body = (await response.json()) as { error?: string };
      if (!response.ok) throw new Error(body.error ?? "Action failed");
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `glm-benchmark-${session?.id ?? "lobby"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  const results =
      data?.agents.filter(
        (a) => a.result && session?.expected.includes(a.userId),
      ) ?? [],
    complete =
      results.length > 0 &&
      results.length === (session?.expected.length ?? 0) &&
      results.every((a) => a.result?.complete);
  const minRounds = results.length
    ? Math.min(...results.map((a) => a.result!.measuredRounds))
    : 0;
  const maxMeasuredSec = results.length
    ? Math.max(...results.map((a) => a.result!.measuredSeconds))
    : 0;
  const groupTps =
    complete && training && minRounds > 0 && maxMeasuredSec > 0
      ? results.reduce((sum, a) => {
          const r = a.result!;
          const perRound =
            r.measuredRounds > 0
              ? (r.processedTokens ?? 0) / r.measuredRounds
              : (r.localBatchTokens ?? 0);
          return sum + minRounds * perRound;
        }, 0) / maxMeasuredSec
      : 0;
  const estimatedHours =
    groupTps > 0
      ? (session?.config.trainingTokens ?? 500000000) / groupTps / 3600
      : 0;
  return (
    <section className="benchmark-panel">
      <div className="lab-banner">
        <span>
          <Activity size={18} />
          <b>One team. One synchronized test.</b> {warmup} seconds warmup +{" "}
          {measurement} seconds measured.
        </span>
        <span className="badge">
          {training
            ? `Real model training · ${compiled ? "compiled CUDA · " : ""}${precision}`
            : "Synthetic compute + Hivemind averaging"}
        </span>
      </div>
      {error && (
        <div role="alert" className="notice error">
          {error}
        </div>
      )}
      <p className="caption">
        Every online computer must be ready. Start freezes the participant list;
        offline computers are excluded. You can run multiple nodes under one
        member key. One computer measures compute only, without peer averaging.
      </p>
      {coordinator && (
        <section className="card benchmark-settings">
          <h2>Benchmark settings</h2>
          <p className="caption">
            Save before Start. Choose the workload before connecting agents.
            Timing can change while computers wait in the lobby.
          </p>
          <div className="benchmark-setting-fields">
            <label>
              Warmup (seconds)
              <input
                type="number"
                min={0}
                max={600}
                step={1}
                value={draftWarmup}
                disabled={busy || (!!session && phase !== "lobby")}
                onChange={(e) => setDraftWarmup(Number(e.target.value))}
              />
            </label>
            <label>
              Measurement (seconds)
              <input
                type="number"
                min={10}
                max={3600}
                step={1}
                value={draftMeasure}
                disabled={busy || (!!session && phase !== "lobby")}
                onChange={(e) => setDraftMeasure(Number(e.target.value))}
              />
            </label>
            <label>
              Workload
              <select
                value={draftWorkload}
                disabled={
                  busy ||
                  (!!session && phase !== "lobby") ||
                  !!data?.agents.length
                }
                onChange={(e) => {
                  setDraftWorkload(e.target.value);
                  if (e.target.value === "synthetic") {
                    setDraftProfile("reference-fp32");
                    setDraftMicrobatch(1);
                    setDraftLocalSteps(1);
                  }
                }}
              >
                <option value="synthetic">Synthetic connectivity / GEMM</option>
                <option value="baseline-training">
                  24.39M model · real training
                </option>
              </select>
            </label>
            {draftWorkload === "baseline-training" && <>
              <label>
                Global tokens per exchange
                <select value={draftBatchTokens}
                  disabled={busy || (!!session && phase !== "lobby") || !!data?.agents.length}
                  onChange={(e) => setDraftBatchTokens(Number(e.target.value))}>
                  {[32768, 65536, 98304, 131072, 262144].map((n) => <option value={n} key={n}>{n.toLocaleString()}</option>)}
                </select>
              </label>
              <label>
                Training profile
                <select value={draftProfile}
                  disabled={busy || (!!session && phase !== "lobby") || !!data?.agents.length}
                  onChange={(e) => {
                    setDraftProfile(e.target.value);
                    setDraftMicrobatch(e.target.value === "reference-fp32" ? 1 : e.target.value === "optimized-bf16" ? 32 : e.target.value === "compiled-bf16" ? 16 : 4);
                    if (e.target.value === "reference-fp32") {
                      setDraftLocalSteps(1);
                    }
                  }}>
                  <option value="reference-fp32">Reference FP32</option>
                  <option value="optimized-fp32">Optimized FP32</option>
                  <option value="optimized-bf16">Optimized BF16</option>
                  <option value="compiled-bf16">Compiled BF16</option>
                </select>
              </label>
              <label>
                Sequences per microbatch
                <select value={draftMicrobatch}
                  disabled={busy || draftProfile === "reference-fp32" || (!!session && phase !== "lobby") || !!data?.agents.length}
                  onChange={(e) => setDraftMicrobatch(Number(e.target.value))}>
                  {[1, 2, 4, 8, 16, 32, 64].map((n) => <option value={n} key={n}>{n}</option>)}
                </select>
              </label>
              <label>
                Local steps per sync
                <select value={draftLocalSteps}
                  disabled={busy || draftProfile === "reference-fp32" || (!!session && phase !== "lobby") || !!data?.agents.length}
                  onChange={(e) => setDraftLocalSteps(Number(e.target.value))}>
                  {[1, 2, 4, 8, 16, 32].map((n) => <option value={n} key={n}>{n}</option>)}
                </select>
              </label>
            </>}
            <button
              className="secondary"
              disabled={busy || phase !== "lobby" || !settingsDirty}
              onClick={() =>
                void action("configure", {
                  warmupSeconds: draftWarmup,
                  measureSeconds: draftMeasure,
                  workload: draftWorkload,
                  trainingProfile: draftProfile,
                  microbatch: draftMicrobatch,
                  localSteps: draftLocalSteps,
                  batchTokens: draftBatchTokens,
                })
              }
            >
              Save settings
            </button>
          </div>
          {draftProfile === "compiled-bf16" && <p className="caption">
            NVIDIA only, one training computer. Compilation happens before Ready and may take several minutes on first use. Choose Optimized BF16 for AMD or multi-computer runs.
          </p>}
          {!session && (
            <p className="caption">
              Start the coordinator command to create a lobby first.
            </p>
          )}
          {settingsDirty && session && (
            <p className="caption">
              Unsaved changes. Save settings before Start.
            </p>
          )}
        </section>
      )}
      <div className="benchmark-grid">
        <section className="card">
          <div className="section-title">
            <div>
              <span className="eyebrow">CONNECTION LOBBY</span>
              <h2>{compiled ? "Connect one NVIDIA computer" : "Connect 1–10 computers"}</h2>
            </div>
            <Users size={20} />
          </div>
          <p className="muted benchmark-intro">
            Each person launches an agent locally, then marks it ready here.
            Starting this page never launches software on a teammate’s computer.
          </p>
          {!data && (
            <p className="caption">
              Connect with your team access key above to view the private lobby.
            </p>
          )}
          {!session && (
            <div className="empty-state">
              <h3>Waiting for the rendezvous</h3>
              <p>
                The coordinator starts the lightweight local service. It creates
                the lobby and publishes the private-network peer address.
              </p>
            </div>
          )}
          {(session?.startsAt
            ? session.expected
            : data?.agents.map((a) => a.userId)
          )?.map((id) => {
            const a = data?.agents.find((a) => a.userId === id),
              online = !!a && serverNow - a.lastSeenAt < 20000;
            return (
              <div className="bench-agent" key={id}>
                <div className={`presence ${online ? "online" : ""}`} />
                <div className="bench-agent-copy">
                  <b>
                    {id.split(":")[0]} ·{" "}
                    {a?.device.hostname || a?.instanceId.slice(0, 8)}
                  </b>
                  <small>{a?.device.name ?? "Agent not connected"}</small>
                </div>
                <span className={`badge ${online && a?.ready ? "green" : ""}`}>
                  {a?.result
                    ? "Result saved"
                    : !online
                      ? "Offline"
                      : a?.ready
                        ? "Ready"
                        : a?.requestedReady
                          ? "Preparing"
                          : "Connected"}
                </span>
                {(coordinator || id.startsWith((data?.user.id ?? "") + ":")) &&
                  online &&
                  phase === "lobby" && (
                    <button
                      className="secondary small"
                      disabled={busy}
                      onClick={() =>
                        void action("request_ready", {
                          ready: !a?.requestedReady,
                          agentId: id,
                        })
                      }
                    >
                      {a?.requestedReady ? "Not ready" : "Mark ready"}
                    </button>
                  )}
                {(coordinator || id.startsWith((data?.user.id ?? "") + ":")) &&
                  phase === "lobby" && (
                    <button
                      className="text-button"
                      disabled={busy}
                      onClick={() =>
                        void action("remove_agent", { agentId: id })
                      }
                    >
                      Remove
                    </button>
                  )}
              </div>
            );
          })}
          <div className="bench-online">
            <i
              className={
                session && serverNow - session.coordinatorSeenAt < 20000
                  ? "online"
                  : ""
              }
            />
            {session && serverNow - session.coordinatorSeenAt < 20000
              ? "Rendezvous online"
              : "Rendezvous offline / not started"}
          </div>
        </section>
        <section className="card benchmark-clock">
          <div className="section-title">
            <div>
              <span className="eyebrow">CONTROL ROOM</span>
              <h2>
                {phase === "no lobby"
                  ? "Waiting for a lobby"
                  : phase === "lobby"
                    ? "Ready when the team is."
                    : phase === "countdown"
                      ? "Starting together…"
                      : phase === "warmup"
                        ? "Warming up."
                        : phase === "measuring"
                          ? "Measurement in progress."
                          : phase === "cancelled"
                            ? "Benchmark cancelled."
                            : "Measurement finished."}
              </h2>
            </div>
            <span className="badge">{phase}</span>
          </div>
          <div className="benchmark-time">
            {remaining > 0
              ? duration(remaining)
              : phase === "finished"
                ? duration(measurement)
                : `${warmup}s + ${measurement}s`}
            <small>
              {phase === "warmup"
                ? "warmup remaining · excluded from results"
                : phase === "measuring"
                  ? "measurement time remaining"
                  : phase === "finished"
                    ? "minutes measured after warmup"
                    : "warmup + measurement"}
            </small>
          </div>
          <div className="journey-progress">
            <div style={{ width: `${progress}%` }} />
          </div>
          <div className="benchmark-steps">
            <span>
              <Clock size={14} />
              15s shared countdown
            </span>
            <span>01 · Warmup {warmup}s</span>
            <span>02 · Measure {measurement}s</span>
          </div>
          {coordinator && phase === "lobby" && (
            <button
              className="primary full"
              disabled={busy || !session?.canStart || settingsDirty}
              onClick={() => void action("start")}
            >
              <Play size={15} />
              Start benchmark
            </button>
          )}
          {phase === "lobby" && (
            <p className="caption">
              {coordinator
                ? "Start unlocks when the rendezvous and all online computers are online and ready."
                : "The coordinator starts once all online computers are online and ready."}
            </p>
          )}
          {coordinator &&
            session &&
            !["finished", "cancelled"].includes(phase) && (
              <button
                className="text-button bench-cancel"
                disabled={busy}
                onClick={() => void action("cancel")}
              >
                Cancel benchmark / close lobby
              </button>
            )}
          {["finished", "cancelled"].includes(phase) && (
            <p className="caption">
              {results.length} / {session?.expected.length ?? 0} results
              received.{" "}
              {complete
                ? "All participants reported complete measurement windows."
                : "Awaiting reports, or one or more agents reported an incomplete test."}{" "}
              Start the coordinator again to create a new lobby.
            </p>
          )}
        </section>
      </div>
      {training && (
        <section className="card training-benchmark-summary">
          <span className="eyebrow">ACTUAL MODEL WORKLOAD</span>
          <h2>
            {groupTps > 0
              ? `${fmt(groupTps)} group tokens / second`
              : "Training throughput appears after complete reports"}
          </h2>
          <p>
            24,388,992 parameters · context 512 · {precision} · FP32 weights and AdamW · microbatch {session?.config.microbatch ?? 1}{(session?.config.localSteps ?? 1) > 1 ? ` · ${session?.config.localSteps} local steps/sync` : ""} · approximately{" "}
            {fmt(session?.config.batchTokens ?? 32768)} tokens per optimizer step.
          </p>
          {groupTps > 0 && (
            <p>
              <strong>
                {fmt(estimatedHours)} hours ({fmt(estimatedHours / 24)} days)
              </strong>{" "}
              estimated for{" "}
              {fmt((session?.config.trainingTokens ?? 500000000) / 1000000)}M
              tokens at this measured group rate. With 20% contingency:{" "}
              {fmt(estimatedHours * 1.2)} hours.
            </p>
          )}
          <p className="caption">
            Generated token batches; real forward/backward and optimizer updates.
            Multiple nodes include gradient transfer and Hivemind averaging;
            optimized solo runs keep gradients on the GPU. No corpus I/O,
            evaluation or checkpoint costs. Estimate requires every
            participant's complete report and uses the smallest shared round
            count.
          </p>
        </section>
      )}
      <section className="section-title below">
        <div>
          <span className="eyebrow">ACTUAL AGENT REPORTS</span>
          <h2>Benchmark results</h2>
        </div>
        <button
          className="secondary"
          disabled={!results.length}
          onClick={download}
        >
          <Download size={15} />
          Download results
        </button>
      </section>
      {results.length ? (
        <div className="card run-table">
          <table>
            <thead>
              <tr>
                <th>Participant / device</th>
                <th>Valid measured rounds</th>
                <th>Round latency p50 / p95</th>
                <th>
                  {training ? "Training tokens / second" : "Synthetic GEMM"}
                </th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {results.map((a) => (
                <tr key={a.userId}>
                  <td>
                    <strong>{a.userId}</strong>
                    <p>{a.device.name}</p>
                  </td>
                  <td>
                    {a.result!.measuredRounds}
                    <small>
                      {a.result!.warmupRounds} warmup · {a.result!.failedRounds}{" "}
                      failed · {a.result!.boundaryRounds} boundary
                    </small>
                  </td>
                  <td>
                    {fmt(a.result!.latencyP50Ms)} /{" "}
                    {fmt(a.result!.latencyP95Ms)} ms
                  </td>
                  <td>
                    {training ? (
                      <>
                        {`${fmt(a.result!.tokensPerSecond ?? 0)} tokens/s`}
                        {a.result!.localBatchTokens ? (
                          <small>
                            {fmt(a.result!.localBatchTokens)} {(a.result!.localSteps ?? 1) > 1 ? `tok/step × ${a.result!.localSteps}` : "tok/round"}
                            {a.result!.workShare !== undefined
                              ? ` (${fmt(a.result!.workShare * 100)}%)`
                              : ""}
                          </small>
                        ) : null}
                      </>
                    ) : (
                      `${fmt(a.result!.gemmGflops)} GFLOP/s`
                    )}
                  </td>
                  <td>
                    {a.result!.complete ? "Complete" : "Incomplete"}
                    <p>{a.result!.note}</p>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="card empty-state">
          <Activity size={26} />
          <h3>Results appear after the measurement window.</h3>
          <p>
            Warmup work is excluded. Failed or partial-group rounds never count
            as valid measured rounds.
          </p>
        </div>
      )}
      <section className="card assumptions">
        <Activity size={20} />
        <div>
          <h3>What this test measures</h3>
          <p>
            {training ? (
              `The language model runs real ${precision} training steps on generated token batches, with FP32 weights and AdamW state. Multiple computers average gradients through Hivemind; one computer keeps gradients on its GPU in optimized profiles. This is a disposable throughput run, not a corpus training session. BF16 changes numerical precision and needs validation on real training data.`
            ) : (
              <>
                A 25-million-element FP32 tensor is averaged through Hivemind
                using FP16 communication compression. Agents also measure a
                fixed 1024×1024 synthetic matrix-multiplication workload. This
                tests group connectivity, averaging latency, and a simple local
                compute workload. It does not train the theoretical model or
                produce tokens/second for an LLM.
              </>
            )}
          </p>
          <p>
            The configured measurement window is common to all participants. A
            round counts only if it starts and finishes inside that window and
            returns all expected participants. Slow boundary rounds may finish
            later but are excluded. All agents run until the deadline; dropouts
            and missing results are shown explicitly.
          </p>
        </div>
      </section>
      <section className="card benchmark-setup">
        <h3>Connect your machine</h3>
        <p>
          Install the{" "}
          <a
            href="https://github.com/michaelkrymski/distributed-compute/tree/main/training/BENCHMARK.md"
            target="_blank"
            rel="noreferrer"
          >
            Hivemind benchmark tools
          </a>
          , join the team’s private LAN/VPN, then start an agent. Supply your
          access key through the local environment or hidden terminal prompt;
          never put it in Git or a URL.
        </p>
        <pre>
          python -m glmbench agent --api
          https://your-team.example --host
          YOUR_PRIVATE_IP
        </pre>
        <p>
          The coordinator uses{" "}
          <code>
            python -m glmbench coordinator --api … --host YOUR_PRIVATE_IP
          </code>{" "}
          before everyone joins.
        </p>
      </section>
    </section>
  );
}
