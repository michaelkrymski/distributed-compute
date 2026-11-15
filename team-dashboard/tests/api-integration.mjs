// Real HTTP requests against the actual Vinext API + local Cloudflare D1.
// Test credentials are generated locally and never embedded in source.
import assert from "node:assert/strict";
import fs from "node:fs";
const base = process.env.GLM_API_BASE ?? "http://127.0.0.1:5173";
if (!["127.0.0.1", "localhost"].includes(new URL(base).hostname))
  throw new Error("This test may run only against a local test database");
const keys = JSON.parse(
  fs.readFileSync(
    process.env.GLM_TEST_KEYS_FILE ?? "work/team-access-keys.json",
    "utf8",
  ),
);
const ids = Object.keys(keys),
  owner = "michaelkrymski",
  member = ids.find((id) => id !== owner);
async function request(path, id, body, status = 200) {
  const r = await fetch(base + path, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer " + (keys[id] ?? ""),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json();
  assert.equal(r.status, status, JSON.stringify(data));
  return data;
}
await request("/api/team", "anonymous", null, 401);
await request("/api/benchmark", "anonymous", null, 401);
let state = await request("/api/benchmark", owner);
if (state.session && !["finished", "cancelled"].includes(state.session.phase))
  await request("/api/benchmark", owner, {
    action: "cancel",
    sessionId: state.session.id,
  });
await request(
  "/api/benchmark",
  member,
  { action: "create", peers: ["/ip4/127.0.0.1/tcp/40100/p2p/QmTestPeer"] },
  403,
);
const created = await request("/api/benchmark", owner, {
  action: "create",
  peers: ["/ip4/127.0.0.1/tcp/40100/p2p/QmTestPeer"],
});
const sessionId = created.id;
await request(
  "/api/benchmark",
  member,
  {
    action: "configure",
    sessionId,
    warmupSeconds: 0,
    measureSeconds: 10,
    workload: "synthetic",
  },
  403,
);
await request(
  "/api/benchmark",
  owner,
  {
    action: "configure",
    sessionId,
    warmupSeconds: -1,
    measureSeconds: 10,
    workload: "synthetic",
  },
  400,
);
await request("/api/benchmark", owner, {
  action: "configure",
  sessionId,
  warmupSeconds: 17,
  measureSeconds: 42,
  workload: "baseline-training",
});
state = await request("/api/benchmark", owner);
assert.equal(state.session.config.warmupSeconds, 17);
assert.equal(state.session.config.measureSeconds, 42);
assert.equal(state.session.config.payloadElements, 24388992);
await request("/api/benchmark", owner, {
  action: "configure", sessionId, warmupSeconds: 17, measureSeconds: 42,
  workload: "baseline-training", trainingProfile: "optimized-bf16", microbatch: 32,
});
state = await request("/api/benchmark", owner);
assert.equal(state.session.config.trainingProfile, "optimized-bf16");
assert.equal(state.session.config.microbatch, 32);
await request("/api/benchmark", owner, {
  action: "configure", sessionId, warmupSeconds: 17, measureSeconds: 42,
  workload: "baseline-training", trainingProfile: "optimized-bf16", microbatch: 32,
  batchTokens: 98304,
});
assert.equal((await request("/api/benchmark", owner)).session.config.batchTokens, 98304);
await request("/api/benchmark", owner, {
  action: "configure", sessionId, warmupSeconds: 17, measureSeconds: 42,
  workload: "baseline-training", trainingProfile: "optimized-bf16", microbatch: 32,
  batchTokens: 32768,
});
// An older FP32 client cannot silently join a BF16 lobby.
await request("/api/benchmark", owner, {
  action: "register", sessionId, instanceId: "outdated-agent-0001",
  device: {name: "Local API fixture", kind: "cpu", torch: "fixture", platform: "Linux",
    workload: "baseline-training", batchTokens: 32768, workloadFingerprint: "a".repeat(64)},
}, 409);
await request("/api/benchmark", owner, {
  action: "configure",
  sessionId,
  warmupSeconds: 60,
  measureSeconds: 300,
  workload: "synthetic",
  trainingProfile: "reference-fp32",
  microbatch: 1,
});
await request("/api/benchmark", owner, { action: "start", sessionId }, 409);
await request("/api/benchmark", member, { action: "start", sessionId }, 403);
// Ten machines, including multiple machines authenticated by the same member.
const machineOwners = Array.from({ length: 10 }, (_, i) =>
  i === 1 ? member : owner,
);
for (let i = 0; i < 10; i++) {
  const id = machineOwners[i],
    instanceId = "integration-" + i;
  await request("/api/benchmark", id, {
    action: "register",
    sessionId,
    instanceId,
    device: {
      name: "Real API contract fixture",
      kind: "cpu",
      torch: "test",
      platform: "test",
    },
  });
  await request("/api/benchmark", owner, { action: "start", sessionId }, 409);
}
await request(
  "/api/benchmark",
  owner,
  {
    action: "register",
    sessionId,
    instanceId: "integration-eleven",
    device: {
      name: "Over capacity",
      kind: "cpu",
      torch: "test",
      platform: "test",
    },
  },
  409,
);
await request(
  "/api/benchmark",
  member,
  {
    action: "request_ready",
    sessionId,
    agentId: owner + ":integration-0",
    ready: true,
  },
  403,
);
for (let i = 0; i < 10; i++) {
  const id = machineOwners[i],
    instanceId = "integration-" + i;
  await request("/api/benchmark", id, {
    action: "request_ready",
    sessionId,
    instanceId,
    ready: true,
  });
  await request("/api/benchmark", id, {
    action: "heartbeat",
    sessionId,
    instanceId,
    ready: true,
  });
  if (i < 9)
    await request("/api/benchmark", owner, { action: "start", sessionId }, 409);
}
await request(
  "/api/benchmark",
  owner,
  {
    action: "configure",
    sessionId,
    warmupSeconds: 60,
    measureSeconds: 300,
    workload: "baseline-training",
  },
  409,
);
state = await request("/api/benchmark", owner);
assert.equal(state.session.canStart, true);
assert.equal(state.session.config.warmupSeconds, 60);
assert.equal(state.session.config.measureSeconds, 300);
await request(
  "/api/benchmark",
  member,
  {
    action: "heartbeat",
    sessionId,
    instanceId: "someone-elses-agent",
    ready: true,
  },
  409,
);
await request("/api/benchmark", owner, { action: "start", sessionId });
state = await request("/api/benchmark", owner);
assert.equal(state.session.expected.length, 10);
await request(
  "/api/benchmark",
  owner,
  {
    action: "configure",
    sessionId,
    warmupSeconds: 0,
    measureSeconds: 10,
    workload: "synthetic",
  },
  409,
);
await request(
  "/api/benchmark",
  owner,
  {
    action: "register",
    sessionId,
    instanceId: "integration-too-late",
    device: { name: "Late", kind: "cpu", torch: "test", platform: "test" },
  },
  409,
);

await request("/api/benchmark", owner, { action: "start", sessionId }, 409);
await request(
  "/api/benchmark",
  member,
  {
    action: "result",
    sessionId,
    instanceId: "integration-1",
    result: {},
  },
  409,
);
await request("/api/benchmark", owner, { action: "cancel", sessionId });
await request("/api/team", owner, {
  action: "resolve",
  pollId: "vocab",
  choice: null,
});
await request("/api/team", owner, {
  action: "vote",
  pollId: "vocab",
  choice: 1,
});
await request("/api/team", owner, {
  action: "vote",
  pollId: "vocab",
  choice: 2,
});
state = await request("/api/team", member);
assert.equal(
  state.votes.filter((v) => v.pollId === "vocab" && v.userId === owner).length,
  1,
);
assert.equal(
  state.votes.find((v) => v.pollId === "vocab" && v.userId === owner).choice,
  2,
);
await request("/api/team", owner, {
  action: "resolve",
  pollId: "vocab",
  choice: 2,
});
await request(
  "/api/team",
  member,
  { action: "vote", pollId: "vocab", choice: 1 },
  409,
);
await request("/api/team", owner, {
  action: "resolve",
  pollId: "vocab",
  choice: null,
});
await request("/api/team", owner, {
  action: "vote",
  pollId: "vocab",
  choice: null,
});
console.log(
  "PASS: real API/D1 authentication, role restrictions, ten-machine capacity, shared-member identities, readiness gate, fixed timing, duplicate Start rejection, agent ownership, vote replacement and finalization",
);

const compiledSession = (await request("/api/benchmark", owner, {
  action: "create", peers: ["/ip4/127.0.0.1/tcp/40100/p2p/QmTestPeer"],
})).id;
await request("/api/benchmark", owner, {
  action: "configure", sessionId: compiledSession, warmupSeconds: 0,
  measureSeconds: 10, workload: "baseline-training",
  trainingProfile: "compiled-bf16", microbatch: 16,
});
const compiledDevice = {
  name: "Compiled API contract fixture", kind: "cuda:0", torch: "fixture", platform: "Linux",
  workload: "baseline-training", batchTokens: 32768, workloadFingerprint: "b".repeat(64),
  trainingProfile: "compiled-bf16", microbatch: 16, compiledModel: true,
};
for (const device of [
  {...compiledDevice, kind: "rocm:0"},
  {...compiledDevice, compiledModel: false},
]) await request("/api/benchmark", owner, {
  action: "register", sessionId: compiledSession, instanceId: "compiled-rejected", device,
}, 400);
await request("/api/benchmark", owner, {
  action: "register", sessionId: compiledSession, instanceId: "compiled-accepted", device: compiledDevice,
});
await request("/api/benchmark", owner, {
  action: "register", sessionId: compiledSession, instanceId: "compiled-second-node", device: compiledDevice,
}, 409);
await request("/api/benchmark", owner, {
  action: "configure", sessionId: compiledSession, warmupSeconds: 0,
  measureSeconds: 10, workload: "baseline-training", trainingProfile: "optimized-bf16", microbatch: 16,
}, 409);
await request("/api/benchmark", owner, {action: "cancel", sessionId: compiledSession});
console.log("PASS: compiled profile rejects AMD/uncompiled agents, enforces one NVIDIA node and freezes the profile after registration");
