import { getDb } from "../../../db";
import {
  benchmarkSessions as sessions,
  benchmarkAgents as agents,
} from "../../../db/schema";
import { getTeamUser, apiJson as json, apiOptions } from "../../../lib/access";
import {
  BENCHMARK_CONFIG,
  ONLINE_MS,
  benchmarkPhase,
  canStart,
  validResult,
  benchmarkSettings,
  workloadCompatible,
} from "../../../lib/benchmark";
import { eq, and, desc, isNull, sql } from "drizzle-orm";
export const dynamic = "force-dynamic";
export const OPTIONS = apiOptions;
async function current() {
  return (
    (
      await getDb()
        .select()
        .from(sessions)
        .where(eq(sessions.activeSlot, "active"))
    )[0] ?? null
  );
}
export async function GET(request: Request) {
  const user = await getTeamUser(request);
  if (!user) return json({ error: "Enter a valid team access key." }, 401);
  try {
    const session = await current(),
      now = Date.now();
    const workers = session
      ? await getDb()
          .select()
          .from(agents)
          .where(eq(agents.sessionId, session.id))
      : [];
    return json({
      serverTime: now,
      user: { id: user.userId, name: user.displayName, role: user.role },
      session: session
        ? {
            ...session,
            phase: benchmarkPhase(session, now),
            expected:
              session.startsAt === null
                ? workers.map((a) => a.userId)
                : session.expected,
            canStart:
              workloadCompatible(
                session.config,
                workers.filter((a) => a.lastSeenAt > now - ONLINE_MS),
              ) &&
              canStart(
                {
                  ...session,
                  expected: workers
                    .filter((a) => a.lastSeenAt > now - ONLINE_MS)
                    .map((a) => a.userId),
                },
                workers,
                now,
              ),
          }
        : null,
      agents: workers,
    });
  } catch (e) {
    console.error("Benchmark read failed", e);
    return json({ error: "Benchmark database unavailable." }, 503);
  }
}
export async function POST(request: Request) {
  const user = await getTeamUser(request);
  if (!user) return json({ error: "Enter a valid team access key." }, 401);
  if (!request.headers.get("content-type")?.includes("application/json"))
    return json({ error: "JSON required." }, 415);
  const text = await request.text();
  if (text.length > 16000) return json({ error: "Request too large." }, 413);
  let b;
  try {
    b = JSON.parse(text);
  } catch {
    return json({ error: "Invalid JSON." }, 400);
  }
  if (!b || typeof b !== "object")
    return json({ error: "Invalid request." }, 400);
  const db = getDb(),
    now = Date.now();
  try {
    const session = await current();
    if (b.action === "create") {
      if (user.role !== "coordinator")
        return json({ error: "Only the coordinator can create a lobby." }, 403);
      if (
        !Array.isArray(b.peers) ||
        b.peers.length < 1 ||
        b.peers.length > 3 ||
        b.peers.some(
          (p: unknown) =>
            typeof p !== "string" ||
            p.length > 300 ||
            !/^\/ip[46]\/.+\/tcp\/\d+\/p2p\/[A-Za-z0-9]+$/.test(p),
        )
      )
        return json(
          { error: "Provide reachable Hivemind bootstrap addresses." },
          400,
        );
      if (
        session &&
        !["finished", "cancelled"].includes(benchmarkPhase(session, now))
      )
        return json(
          {
            error:
              "A lobby already exists. Cancel it before replacing the coordinator.",
          },
          409,
        );
      const id = crypto.randomUUID();
      await db.batch([
        db
          .update(sessions)
          .set({ activeSlot: null })
          .where(eq(sessions.activeSlot, "active")),
        db.insert(sessions).values({
          id,
          activeSlot: "active",
          createdAt: now,
          createdBy: user.userId,
          peers: b.peers,
          expected: [],
          config: BENCHMARK_CONFIG,
          coordinatorSeenAt: now,
          startsAt: null,
          cancelledAt: null,
        }),
      ]);
      return json({ ok: true, id, serverTime: now });
    }
    if (!session || b.sessionId !== session.id)
      return json(
        { error: "The active benchmark lobby changed. Refresh and reconnect." },
        409,
      );
    const machineId = `${user.userId}:${b.instanceId}`;
    const owner = and(
      eq(agents.sessionId, session.id),
      eq(agents.userId, machineId),
    );
    if (b.action === "configure") {
      if (user.role !== "coordinator")
        return json({ error: "Admin access required." }, 403);
      const config = benchmarkSettings(
        b.warmupSeconds,
        b.measureSeconds,
        b.workload,
        b.batchTokens ?? session.config.batchTokens ?? 32768,
        b.trainingProfile ?? session.config.trainingProfile ?? "reference-fp32",
        b.microbatch ?? session.config.microbatch ?? 1,
        b.localSteps ?? session.config.localSteps ?? 1,
      );
      if (!config)
        return json(
          {
            error:
              "Use 0–600 warmup seconds and 10–3600 measurement seconds, and select a valid workload.",
          },
          400,
        );
      const changed = await db
        .update(sessions)
        .set({ config })
        .where(
          and(
            eq(sessions.id, session.id),
            isNull(sessions.startsAt),
            isNull(sessions.cancelledAt),
            sql`((coalesce(json_extract(config,'$.workload'),'synthetic') = ${config.workload} AND coalesce(json_extract(config,'$.batchTokens'),32768) = ${config.batchTokens} AND coalesce(json_extract(config,'$.trainingProfile'),'reference-fp32') = ${config.trainingProfile} AND coalesce(json_extract(config,'$.microbatch'),1) = ${config.microbatch} AND coalesce(json_extract(config,'$.localSteps'),1) = ${config.localSteps}) OR NOT EXISTS (SELECT 1 FROM benchmark_agents WHERE session_id=${session.id}))`,
          ),
        )
        .returning({ id: sessions.id });
      if (!changed.length)
        return json(
          {
            error:
              "Timing is editable only before Start. Remove all agents before changing the workload.",
          },
          409,
        );
    } else if (b.action === "coordinator_heartbeat") {
      if (user.role !== "coordinator" || user.userId !== session.createdBy)
        return json({ error: "Coordinator key required." }, 403);
      await db
        .update(sessions)
        .set({ coordinatorSeenAt: now })
        .where(eq(sessions.id, session.id));
    } else if (b.action === "register") {
      if (benchmarkPhase(session, now) !== "lobby")
        return json(
          { error: "Agents can join only before the benchmark starts." },
          409,
        );
      if (
        typeof b.instanceId !== "string" ||
        !/^[-a-zA-Z0-9]{10,80}$/.test(b.instanceId) ||
        !b.device ||
        ["name", "kind", "torch", "platform"].some(
          (k) => typeof b.device[k] !== "string" || b.device[k].length > 180,
        )
      )
        return json({ error: "Invalid agent registration." }, 400);
      if (
        session.config.workload === "baseline-training" &&
        !/^[a-f0-9]{64}$/.test(b.device.workloadFingerprint ?? "")
      )
        return json(
          { error: "Training agent must report a model fingerprint." },
          400,
        );
      // SQLite serializes this INSERT with Start. A registration cannot sneak
      // into a started session or exceed the ten-machine limit concurrently.
      if (session.config.trainingProfile === "compiled-bf16" && (
        b.device.compiledModel !== true || !b.device.kind.startsWith("cuda:")
      )) return json({error: "Compiled BF16 requires one NVIDIA CUDA node after compilation succeeds."}, 400);
      if (
        (b.device.workload ?? "synthetic") !==
        (session.config.workload ?? "synthetic") ||
        (session.config.workload === "baseline-training" && (
          (b.device.trainingProfile ?? "reference-fp32") !== (session.config.trainingProfile ?? "reference-fp32") ||
          (b.device.microbatch ?? 1) !== (session.config.microbatch ?? 1) ||
          (b.device.localSteps ?? 1) !== (session.config.localSteps ?? 1)
        ))
      )
        return json({ error: "Workload changed. Restart the agent." }, 409);
      const changed = await db.run(sql`INSERT INTO benchmark_agents
        (session_id,user_id,instance_id,device,last_seen_at,requested_ready,ready,result)
        SELECT ${session.id},${machineId},${b.instanceId},${JSON.stringify(b.device)},${now},0,0,NULL
        WHERE EXISTS (SELECT 1 FROM benchmark_sessions WHERE id=${session.id} AND starts_at IS NULL AND cancelled_at IS NULL AND coalesce(json_extract(config,'$.workload'),'synthetic') = ${b.device.workload ?? "synthetic"} AND coalesce(json_extract(config,'$.batchTokens'),32768) = ${b.device.batchTokens ?? 32768} AND coalesce(json_extract(config,'$.trainingProfile'),'reference-fp32') = ${b.device.trainingProfile ?? "reference-fp32"} AND coalesce(json_extract(config,'$.microbatch'),1) = ${b.device.microbatch ?? 1} AND coalesce(json_extract(config,'$.localSteps'),1) = ${b.device.localSteps ?? 1})
        AND (SELECT COUNT(*) FROM benchmark_agents WHERE session_id=${session.id}) < ${session.config.trainingProfile === "compiled-bf16" ? 1 : 10}
        ON CONFLICT(session_id,user_id) DO NOTHING`);
      if (!changed.meta.changes)
        return json(
          {
            error:
              session.config.trainingProfile === "compiled-bf16"
                ? "Compiled BF16 accepts one NVIDIA computer; the lobby is occupied or already started."
                : "Lobby full (10 computers), already registered, or already started.",
          },
          409,
        );
    } else if (b.action === "request_ready") {
      if (
        benchmarkPhase(session, now) !== "lobby" ||
        typeof b.ready !== "boolean"
      )
        return json({ error: "Readiness can change only in the lobby." }, 409);
      const target = typeof b.agentId === "string" ? b.agentId : machineId;
      if (user.role !== "coordinator" && !target.startsWith(user.userId + ":"))
        return json({ error: "This is another member's computer." }, 403);
      await db
        .update(agents)
        .set({ requestedReady: b.ready, ready: false })
        .where(
          and(eq(agents.sessionId, session.id), eq(agents.userId, target)),
        );
    } else if (b.action === "remove_agent") {
      if (
        benchmarkPhase(session, now) !== "lobby" ||
        typeof b.agentId !== "string"
      )
        return json(
          { error: "Computers can be removed only in the lobby." },
          409,
        );
      if (
        user.role !== "coordinator" &&
        !b.agentId.startsWith(user.userId + ":")
      )
        return json({ error: "This is another member's computer." }, 403);
      await db
        .delete(agents)
        .where(
          and(eq(agents.sessionId, session.id), eq(agents.userId, b.agentId)),
        );
    } else if (b.action === "heartbeat" || b.action === "result") {
      const agent = (await db.select().from(agents).where(owner))[0];
      if (!agent || agent.instanceId !== b.instanceId)
        return json({ error: "Register this agent first." }, 409);
      if (b.action === "result") {
        if (!["finished", "cancelled"].includes(benchmarkPhase(session, now)))
          return json(
            {
              error:
                "Results may be submitted only after the measurement window.",
            },
            409,
          );
        if (
          !validResult(b.result) ||
          (b.result.workload ?? "synthetic") !==
            (session.config.workload ?? "synthetic")
        )
          return json({ error: "Invalid benchmark results." }, 400);
        if (session.config.workload === "baseline-training") {
          const context = session.config.model?.context ?? 512;
          const stepBatch = session.config.batchTokens ?? 32768;
          const localSteps = session.config.localSteps ?? 1;
          const maxProcessedBatch = stepBatch * localSteps;
          const validLocalBatch =
            Number.isInteger(b.result.localBatchTokens) &&
            b.result.localBatchTokens >= context &&
            b.result.localBatchTokens <= stepBatch &&
            b.result.localBatchTokens % context === 0;
          const validProcessed =
            Number.isInteger(b.result.processedTokens) &&
            b.result.processedTokens >= b.result.measuredRounds * context &&
            b.result.processedTokens <= b.result.measuredRounds * maxProcessedBatch &&
            b.result.processedTokens % context === 0;
          if (
            b.result.parameterCount !== session.config.payloadElements ||
            !validLocalBatch ||
            !validProcessed ||
            b.result.modelFingerprint !== agent.device.workloadFingerprint ||
            (b.result.trainingProfile ?? "reference-fp32") !== (session.config.trainingProfile ?? "reference-fp32") ||
            (b.result.microbatch ?? 1) !== (session.config.microbatch ?? 1) ||
            (b.result.localSteps ?? 1) !== localSteps ||
            (b.result.precision ?? "fp32") !== ((session.config.trainingProfile ?? "reference-fp32").endsWith("bf16") ? "bf16" : "fp32") ||
            (b.result.compiledModel ?? false) !== (session.config.trainingProfile === "compiled-bf16") ||
            (session.config.trainingProfile === "compiled-bf16" && (!Number.isFinite(b.result.compileSetupSeconds) || b.result.compileSetupSeconds < 0)) ||
            b.result.measuredSeconds > session.config.measureSeconds
          )
            return json(
              {
                error:
                  "Training report does not match the frozen model/batch contract.",
              },
              400,
            );
        }
        await db
          .update(agents)
          .set({ result: b.result, lastSeenAt: now, ready: false })
          .where(owner);
      } else {
        if (typeof b.ready !== "boolean")
          return json({ error: "Invalid heartbeat." }, 400);
        await db
          .update(agents)
          .set({ lastSeenAt: now, ready: agent.requestedReady && b.ready })
          .where(owner);
      }
    } else if (b.action === "start") {
      if (user.role !== "coordinator")
        return json({ error: "Coordinator key required to start." }, 403);
      const workers = await db
        .select()
        .from(agents)
        .where(eq(agents.sessionId, session.id));
      if (
        !workloadCompatible(
          session.config,
          workers.filter((a) => a.lastSeenAt > now - ONLINE_MS),
        )
      )
        return json(
          {
            error:
              "Online nodes must use the same model code and initial weights. Update both computers to the same theoretical branch commit.",
          },
          409,
        );
      if (
        !canStart(
          {
            ...session,
            expected: workers
              .filter((a) => a.lastSeenAt > now - ONLINE_MS)
              .map((a) => a.userId),
          },
          workers,
          now,
        )
      )
        return json(
          {
            error:
              "Connect 1–10 computers and mark every online computer ready. The coordinator must also be online.",
          },
          409,
        );
      // Conditional update makes repeated/concurrent Start clicks idempotent.
      const changed = await db
        .update(sessions)
        .set({
          startsAt: now + 15_000,
          expected: sql`(SELECT json_group_array(user_id) FROM benchmark_agents WHERE session_id=${session.id} AND last_seen_at > ${now - ONLINE_MS})`,
        })
        .where(
          and(
            eq(sessions.id, session.id),
            isNull(sessions.startsAt),
            isNull(sessions.cancelledAt),
            sql`(${session.config.workload ?? "synthetic"} != 'baseline-training' OR (SELECT COUNT(DISTINCT json_extract(device,'$.workloadFingerprint')) FROM benchmark_agents WHERE session_id=${session.id} AND last_seen_at > ${now - ONLINE_MS})=1)`,
            sql`${sessions.coordinatorSeenAt} > ${now - ONLINE_MS}`,
            sql`(SELECT COUNT(*) FROM benchmark_agents WHERE session_id=${session.id} AND last_seen_at > ${now - ONLINE_MS}) BETWEEN 1 AND 10`,
            sql`NOT EXISTS (SELECT 1 FROM benchmark_agents WHERE session_id=${session.id} AND last_seen_at > ${now - ONLINE_MS} AND (ready=0 OR requested_ready=0))`,
          ),
        )
        .returning({ id: sessions.id });
      if (!changed.length)
        return json({ error: "This benchmark has already started." }, 409);
    } else if (b.action === "cancel") {
      if (user.role !== "coordinator")
        return json({ error: "Coordinator key required to cancel." }, 403);
      await db
        .update(sessions)
        .set({ cancelledAt: now })
        .where(eq(sessions.id, session.id));
    } else return json({ error: "Unknown action." }, 400);
    return json({ ok: true, serverTime: now });
  } catch (e) {
    console.error("Benchmark write failed", e);
    return json({ error: "Could not save benchmark state." }, 503);
  }
}
