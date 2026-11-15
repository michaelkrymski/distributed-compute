import { body, fail } from "./security.js";
import { benchmarkSettings, benchmarkPhase, canStart, workloadCompatible, validResult } from "./benchmark-contract.ts";

const empty = kind => kind === "benchmark" ? {session:null,agents:[],history:[]} : {room:null,members:[],history:[]};
const fresh = (agent, now) => agent.lastSeenAt > now - 20000;
const peer = value => typeof value === "string" && value.length <= 300 && /^\/ip[46]\/[^\s]+\/tcp\/\d+\/p2p\/[A-Za-z0-9]+$/.test(value);
const integer = (n,min,max) => Number.isSafeInteger(n) && n >= min && n <= max;
const admin = actor => { if (actor.role !== "coordinator") fail("Team administrator access required.",403); };
export async function stateRead(db, projectId, kind) {
  const row = await db.prepare("SELECT payload FROM coordination_states WHERE project_id=? AND kind=?").bind(projectId,kind).first();
  const value = row ? JSON.parse(row.payload) : empty(kind);
  if (kind === "benchmark" && value.session) {
    const now = Date.now(), online = value.agents.filter(a => fresh(a,now));
    value.session.phase = benchmarkPhase(value.session,now);
    value.session.canStart = canStart({...value.session,expected:online.map(a=>a.userId)},value.agents,now) && workloadCompatible(value.session.config,online);
    if (value.session.startsAt === null) value.session.expected = value.agents.map(a=>a.userId);
  }
  return value;
}
function settings(b, project, previous = {}) {
  const context = project.spec.model.context;
  const batch = b.batchTokens ?? previous.batchTokens ?? Math.ceil(98304/context)*context;
  if (!integer(batch,context,262144) || batch % context) fail("Tokens per exchange must be a positive multiple of this project's context length, up to 262144.");
  const config = benchmarkSettings(b.warmupSeconds ?? previous.warmupSeconds ?? 60,b.measureSeconds ?? previous.measureSeconds ?? 300,b.workload ?? previous.workload ?? "baseline-training",512,b.trainingProfile ?? previous.trainingProfile ?? "optimized-fp32",b.microbatch ?? previous.microbatch ?? 4,b.localSteps ?? previous.localSteps ?? 1);
  if (!config) fail("Invalid benchmark settings. Warmup: 0–600s; measurement: 10–3600s. Select compatible workload/profile settings.");
  config.batchTokens = batch;
  config.model = project.spec.model;
  config.payloadElements = config.workload === "baseline-training" ? project.spec.parameters : Math.min(project.spec.parameters,25000000);
  config.trainingTokens = project.spec.targetTokens;
  config.projectId = project.id;
  config.projectRevision = project.revision;
  config.runtime = project.spec.runtime;
  config.adapterConfig = project.spec.adapterConfig;
  if (config.workload === "baseline-training" && config.batchTokens % config.model.context !== 0) fail("Tokens per exchange must be divisible by the model context length.");
  return config;
}
function benchmarkMutation(state,b,actor,project,now) {
  const action = b.action, machineId = `${actor.userId}:${b.instanceId ?? ""}`;
  let session = state.session;
  if (action === "create") {
    admin(actor);
    if (session && !["finished","cancelled"].includes(benchmarkPhase(session,now))) fail("A benchmark lobby already exists. Cancel it first.",409);
    if (!Array.isArray(b.peers) || b.peers.length < 1 || b.peers.length > 3 || !b.peers.every(peer)) fail("Start a local coordinator and provide its reachable Hivemind peer address.");
    if (session) state.history.unshift({session,agents:state.agents});
    state.history = state.history.slice(0,20);
    state.session = {id:crypto.randomUUID(),createdAt:now,createdBy:actor.userId,peers:b.peers,expected:[],config:settings(b,project),coordinatorSeenAt:now,startsAt:null,cancelledAt:null};
    state.agents = [];return;
  }
  if (!session || b.sessionId && b.sessionId !== session.id) fail("No matching benchmark session.",409);
  const phase = benchmarkPhase(session,now);
  const agent = state.agents.find(a=>a.userId===machineId && a.instanceId===b.instanceId);
  const target = typeof b.agentId === "string" ? b.agentId : machineId;
  if (action === "coordinator_heartbeat") { admin(actor); if (actor.userId !== session.createdBy) fail("Only this coordinator can send its heartbeat.",403); session.coordinatorSeenAt=now;return; }
  if (action === "configure") {
    admin(actor);if (phase !== "lobby") fail("Settings are editable only in the lobby.",409);
    const config = settings(b,project,session.config);
    if (state.agents.length && ["workload","batchTokens","trainingProfile","microbatch","localSteps"].some(k=>config[k]!==session.config[k])) fail("Remove the agents before changing the workload.",409);
    session.config=config;return;
  }
  if (action === "register") {
    if (phase !== "lobby") fail("Agents can join only before Start.",409);
    const limit = session.config.trainingProfile === "compiled-bf16" ? 1 : 10;
    if (state.agents.length >= limit || state.agents.some(a=>a.userId===machineId)) fail("Lobby full or computer already joined.",409);
    if (typeof b.instanceId !== "string" || !/^[-a-zA-Z0-9]{10,80}$/.test(b.instanceId) || !b.device || ["name","kind","torch","platform"].some(k=>typeof b.device[k]!=="string" || b.device[k].length>180)) fail("Invalid computer registration.");
    if ((b.device.workload ?? "synthetic") !== session.config.workload || (b.device.batchTokens ?? 32768) !== session.config.batchTokens || ["trainingProfile","microbatch","localSteps"].some(k=>(b.device[k] ?? ({trainingProfile:"reference-fp32",microbatch:1,localSteps:1}[k]))!==session.config[k])) fail("Workload changed. Restart the agent.",409);
    if (session.config.workload === "baseline-training" && !/^[a-f0-9]{64}$/.test(b.device.workloadFingerprint ?? "")) fail("A model fingerprint is required.");
    if (session.config.trainingProfile === "compiled-bf16" && (!b.device.compiledModel || !b.device.kind.startsWith("cuda:"))) fail("Compiled BF16 requires a compiled NVIDIA worker.");
    state.agents.push({sessionId:session.id,userId:machineId,instanceId:b.instanceId,device:b.device,lastSeenAt:now,requestedReady:false,ready:false,result:null});return;
  }
  if (action === "request_ready" || action === "remove_agent") {
    if (phase !== "lobby") fail("Readiness and membership are frozen after Start.",409);
    if (actor.role !== "coordinator" && !target.startsWith(actor.userId+":")) fail("This is another member's computer.",403);
    const row = state.agents.find(a=>a.userId===target);if (!row) fail("Computer not found.",404);
    if (action === "remove_agent") state.agents=state.agents.filter(a=>a.userId!==target);
    else {if (typeof b.ready!=="boolean") fail("Invalid ready state.");row.requestedReady=b.ready;row.ready=false;}
    return;
  }
  if (action === "heartbeat" || action === "result") {
    if (!agent) fail("Register this computer first.",409);
    if (action === "heartbeat") {agent.lastSeenAt=now;agent.ready=!!b.ready && agent.requestedReady;return;}
    const r=b.result,c=session.config;
    if (!["finished","cancelled"].includes(phase)) fail("Results are accepted only after measurement.",409);
    if (!validResult(r) || (r.workload ?? "synthetic")!==c.workload || r.measuredSeconds>c.measureSeconds) fail("Invalid benchmark report.");
    if (r.complete && (r.measuredRounds<1 || r.failedRounds>0 || r.measuredSeconds<c.measureSeconds-1 || r.clockUncertaintyMs>1000)) fail("An incomplete or degraded benchmark cannot be marked complete.");
    if (c.workload === "baseline-training" && (r.parameterCount!==c.payloadElements || r.modelFingerprint!==agent.device.workloadFingerprint || !integer(r.processedTokens,0,Number.MAX_SAFE_INTEGER) || r.processedTokens % c.model.context || !integer(r.localBatchTokens,c.model.context,c.batchTokens) || r.localBatchTokens % c.model.context || r.processedTokens<r.measuredRounds*c.model.context || r.processedTokens>r.measuredRounds*c.batchTokens*c.localSteps || (r.trainingProfile ?? "reference-fp32")!==c.trainingProfile || (r.microbatch ?? 1)!==c.microbatch || (r.localSteps ?? 1)!==c.localSteps || (r.precision ?? "fp32")!==(c.trainingProfile.endsWith("bf16") ? "bf16" : "fp32"))) fail("Report does not match this project's workload.");
    agent.result=r;agent.lastSeenAt=now;return;
  }
  if (action === "start") {
    admin(actor);const online=state.agents.filter(a=>fresh(a,now));
    if (!canStart({...session,expected:online.map(a=>a.userId)},state.agents,now) || !workloadCompatible(session.config,online)) fail("Coordinator must be online and every online computer must be ready with a matching workload.",409);
    session.expected=online.map(a=>a.userId);session.startsAt=now+15000;return;
  }
  if (action === "cancel") {admin(actor);session.cancelledAt=now;return;}
  fail("Unknown benchmark action.");
}
function roomsMutation(state,b,actor,project,now) {
  let room=state.room;
  if (b.action === "create") {
    admin(actor);if (room && ["lobby","running"].includes(room.phase)) fail("A training room already exists.",409);
    if (!peer(b.peer) || !/^[a-f0-9]{64}$/.test(b.fingerprint ?? "") || !integer(b.shards,1,1000) || typeof b.name!=="string" || !b.name.trim() || b.name.length>100) fail("Verify the local run bundle, peer address and shard count.");
    if (room) state.history.unshift({room,members:state.members});state.history=state.history.slice(0,20);
    state.room={id:crypto.randomUUID(),name:b.name,fingerprint:b.fingerprint,peer:b.peer,target_tokens:project.spec.targetTokens,shards:b.shards,phase:"lobby",created_by:actor.userId,created_at:now,expected:"[]"};state.members=[];return;
  }
  if (!room || b.roomId!==room.id) fail("Training room not found.",404);
  if (b.action === "cancel") {admin(actor);room.phase="cancelled";return;}
  if (b.action === "start") {
    admin(actor);const online=state.members.filter(m=>m.last_seen>now-20000);
    if (room.phase!=="lobby" || !online.length || online.some(m=>!m.ready)) fail("Every online computer must be ready.",409);
    const expected=online.map(m=>m.instance_id).sort();room.expected=JSON.stringify(expected);room.phase="running";
    for (const m of online) {m.shard=expected.indexOf(m.instance_id);m.leader=m.shard===0 ? 1 : 0;}return;
  }
  if (typeof b.instanceId!=="string" || !/^[a-f0-9]{32}$/.test(b.instanceId)) fail("Invalid computer ID.");
  let m=state.members.find(m=>m.instance_id===b.instanceId);
  if (b.action === "join") {
    if (room.phase!=="lobby" || m || state.members.length>=room.shards) fail("Room started, full or computer already joined.",409);
    if (b.fingerprint!==room.fingerprint || typeof b.name!=="string" || b.name.length>100 || typeof b.device!=="string" || b.device.length>200) fail("Run bundle mismatch or invalid device.");
    state.members.push({room_id:room.id,instance_id:b.instanceId,user_id:actor.userId,name:b.name,device:b.device,fingerprint:b.fingerprint,last_seen:now,ready:0,shard:null,leader:0,progress:null});return;
  }
  if (!m || m.user_id!==actor.userId) fail("Computer not owned by this account.",403);
  if (b.action === "ready" || b.action === "leave") {
    if (room.phase!=="lobby") fail("Room membership is frozen after Start.",409);
    if (b.action === "leave") state.members=state.members.filter(v=>v!==m);
    else {if (typeof b.ready!=="boolean") fail("Invalid ready value.");m.ready=b.ready ? 1 : 0;}return;
  }
  if (b.action === "heartbeat") {
    m.last_seen=now;if (b.progress && typeof b.progress==="object") {const raw=JSON.stringify(b.progress);if (raw.length>4000) fail("Progress report too large.");m.progress=raw;}
    const expected=JSON.parse(room.expected);
    if (room.phase==="running" && expected.length && expected.every(id=>state.members.some(v=>v.instance_id===id && v.progress && JSON.parse(v.progress).event==="worker_finished"))) room.phase="finished";
    return;
  }
  fail("Unknown room action.");
}
export async function coordinate(request,db,project,actor,kind) {
  if (request.method === "GET") return {...await stateRead(db,project.id,kind),serverTime:Date.now(),user:{id:actor.userId,name:actor.userId,role:actor.role},project:{id:project.id,...project.spec}};
  if (request.method !== "POST") fail("Method not allowed.",405);
  const b=await body(request);
  await db.prepare("INSERT INTO coordination_states(project_id,kind,revision,payload) VALUES(?,?,0,?) ON CONFLICT DO NOTHING").bind(project.id,kind,JSON.stringify(empty(kind))).run();
  for (let attempt=0;attempt<8;attempt++) {
    const row=await db.prepare("SELECT revision,payload FROM coordination_states WHERE project_id=? AND kind=?").bind(project.id,kind).first();
    const state=JSON.parse(row.payload);
    if (kind==="benchmark") benchmarkMutation(state,b,actor,project,Date.now());else roomsMutation(state,b,actor,project,Date.now());
    const payload=JSON.stringify(state);if (payload.length>256000) fail("Project session history is full. Create a new project.",409);
    const r=await db.prepare("UPDATE coordination_states SET payload=?,revision=revision+1 WHERE project_id=? AND kind=? AND revision=? AND EXISTS(SELECT 1 FROM compute_projects WHERE id=? AND revision=?)").bind(payload,project.id,kind,row.revision,project.id,project.revision).run();
    if (r.meta.changes) return {ok:true,id:state.session?.id ?? state.room?.id,serverTime:Date.now()};
  }
  fail("Session changed concurrently. Refresh and retry.",409);
}
