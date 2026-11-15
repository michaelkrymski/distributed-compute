"""Fixed synthetic compute and Hivemind tensor averaging; not LLM training."""

import json
import math
import signal
import statistics
import threading
import time
import uuid
from pathlib import Path
import torch
from hivemind.averaging import DecentralizedAverager
from hivemind.compression import Float16Compression
from hivemind.utils import get_dht_time
from .runtime import start_dht


def allocate_proportional_tokens(batch_tokens, context, members, speeds=None, microbatch=1):
    """Split batch_tokens across active members proportionally to compute speeds."""
    ordered = list(members)
    if not ordered:
        return {}
    if context <= 0 or batch_tokens <= 0:
        raise ValueError("batch_tokens and context must be positive")
    if len(ordered) == 1:
        return {ordered[0]: math.ceil(batch_tokens / context) * context}
    total_sequences = max(len(ordered), math.ceil(batch_tokens / context))
    if not speeds or any(
        m not in speeds
        or isinstance(speeds[m], bool)
        or not isinstance(speeds[m], (int, float))
        or not math.isfinite(speeds[m])
        or speeds[m] <= 0
        for m in ordered
    ):
        equal_tokens = math.ceil(batch_tokens / (len(ordered) * context)) * context
        return {m: equal_tokens for m in ordered}
    unique = sorted(ordered)
    weights = {m: max(float(speeds[m]), 1.0) for m in unique}
    total_weight = sum(weights.values())
    seqs = {m: 1 for m in unique}
    remaining = total_sequences - len(unique)
    raw = {m: remaining * (weights[m] / total_weight) for m in unique}
    floors = {m: math.floor(raw[m]) for m in unique}
    for m in unique:
        seqs[m] += floors[m]
    leftover = remaining - sum(floors.values())
    if leftover > 0:
        ranked = sorted(unique, key=lambda m: (-(raw[m] - floors[m]), -weights[m], m))
        for m in ranked[:leftover]:
            seqs[m] += 1
    return {m: seqs[m] * context for m in ordered}


def phase_at(session, now):
    if session.get("cancelledAt") is not None:
        return "cancelled"
    if session.get("startsAt") is None:
        return "lobby"
    start = session["startsAt"] / 1000
    warm_end = start + session["config"]["warmupSeconds"]
    end = warm_end + session["config"]["measureSeconds"]
    return (
        "countdown"
        if now < start
        else "warmup"
        if now < warm_end
        else "measuring"
        if now < end
        else "finished"
    )


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def constrain_duration(result, config, elapsed):
    """Never infer real elapsed time solely from a synchronized wall clock."""
    result["observedElapsedSeconds"] = elapsed
    result["measuredSeconds"] = min(
        result["measuredSeconds"], max(0, elapsed - config["warmupSeconds"])
    )
    if result["measuredSeconds"] < config["measureSeconds"] - 1:
        result["complete"] = False
        result["note"] += (
            " Incomplete monotonic measurement window; check clock synchronization."
        )
    return result


class Measurements:
    def __init__(self, session):
        self.session = session
        self.latencies = []
        self.gemm = []
        self.warmup = 0
        self.failed = 0
        self.boundary = 0
        self.clock_ms = 0
        self.cancelled = False
        self.observed_start = None
        self.observed_end = None
        self.dropped_members = []

    def observe(self, now):
        if self.observed_start is None:
            self.observed_start = now
        self.observed_end = now

    def record(self, start, end, latency_ms, members, compute_gflops, expected_members=None):
        self.observe(start)
        self.observe(end)
        phase = phase_at(self.session, start)
        if phase == "warmup":
            self.warmup += 1
            return
        if phase != "measuring" or phase_at(self.session, end) != "measuring":
            self.boundary += 1
            return
        expected = expected_members if expected_members is not None else self.session["expected"]
        if len(members) != len(expected) or set(members) != set(expected):
            self.failed += 1
            return
        self.latencies.append(latency_ms)
        self.gemm.append(compute_gflops)

    def result(self):
        window_start = (
            self.session["startsAt"] / 1000 + self.session["config"]["warmupSeconds"]
        )
        window_end = window_start + self.session["config"]["measureSeconds"]
        observed_start = (
            self.observed_start if self.observed_start is not None else window_end
        )
        observed_end = (
            self.observed_end if self.observed_end is not None else window_start
        )
        measured_seconds = max(
            0.0, min(window_end, observed_end) - max(window_start, observed_start)
        )
        complete = (
            bool(self.latencies)
            and self.failed == 0
            and not self.cancelled
            and self.clock_ms <= 1000
            and measured_seconds >= self.session["config"]["measureSeconds"] - 1
        )
        return {
            "protocol": 3,
            "complete": complete,
            "measuredRounds": len(self.latencies),
            "warmupRounds": self.warmup,
            "failedRounds": self.failed,
            "boundaryRounds": self.boundary,
            "measuredSeconds": measured_seconds,
            "latencyP50Ms": percentile(self.latencies, 0.5),
            "latencyP95Ms": percentile(self.latencies, 0.95),
            "meanRoundMs": statistics.mean(self.latencies) if self.latencies else 0,
            "gemmGflops": statistics.mean(self.gemm) if self.gemm else 0,
            "clockUncertaintyMs": self.clock_ms,
            "note": "Synthetic tensor averaging and GEMM only; not LLM tokens/sec."
            if complete
            else "Incomplete/degraded: incomplete window, no valid rounds, a failed group, cancellation, or clock uncertainty above 1 second.",
        }


def _read_dht_speeds(dht, prefix_id):
    found = dht.get(f"glmbench_{prefix_id}_speeds", latest=True)
    speeds, timestamps = {}, {}
    if found is None or not isinstance(found.value, dict):
        return speeds, timestamps
    for member, record in found.value.items():
        val = getattr(record, "value", record)
        if isinstance(val, dict):
            tps = val.get("compute_tps")
            ts = val.get("ts")
            if isinstance(tps, (int, float)) and math.isfinite(tps) and tps > 0:
                speeds[str(member)] = float(tps)
            if isinstance(ts, (int, float)) and math.isfinite(ts):
                timestamps[str(member)] = float(ts)
    return speeds, timestamps


def coordinator(args, client):
    client.calibrate_clock()
    dht = start_dht(args.host, args.port, identity=args.identity)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        peers = [str(a) for a in dht.get_visible_maddrs()]
        session = client.post(action="create", peers=peers)["id"]
        session_prefix = session.replace("-", "")
        print(
            json.dumps(
                {"event": "coordinator_online", "sessionId": session, "peers": peers}
            ),
            flush=True,
        )
        locked_schedule = None
        while True:
            client.post(action="coordinator_heartbeat", sessionId=session)
            try:
                state = client.get()
                active_session = state.get("session")
                if (
                    active_session
                    and active_session.get("id") == session
                    and active_session.get("startsAt") is not None
                    and active_session.get("cancelledAt") is None
                ):
                    cfg = active_session["config"]
                    expected = list(active_session.get("expected", []))
                    dropout_timeout = float(cfg.get("dropoutSeconds", 20.0))
                    server_now = float(state.get("serverTime", client.now() * 1000))
                    agents_by_id = {a["userId"]: a for a in state.get("agents", [])}
                    speeds, dht_ts = _read_dht_speeds(dht, session_prefix)
                    now_sec = client.now()
                    phase = phase_at(active_session, now_sec)
                    active_members = []
                    for member in expected:
                        agent_info = agents_by_id.get(member)
                        backend_stale = (
                            agent_info is not None
                            and (server_now - agent_info["lastSeenAt"]) / 1000.0 > dropout_timeout
                        )
                        dht_stale = (
                            member in dht_ts and (now_sec - dht_ts[member]) > dropout_timeout
                        )
                        if phase == "measuring" and backend_stale and (member not in dht_ts or dht_stale):
                            continue
                        active_members.append(member)
                    if not active_members:
                        active_members = expected
                    if cfg.get("workload") == "baseline-training" and active_members:
                        context = cfg.get("model", {}).get("context", 512)
                        batch_tokens = cfg.get("batchTokens", 32768)
                        microbatch = cfg.get("microbatch", 1)
                        warm_end = active_session["startsAt"] / 1000 + cfg.get("warmupSeconds", 60)
                        if (
                            locked_schedule is None
                            or set(locked_schedule["activeMembers"]) != set(active_members)
                        ) and (
                            (
                                now_sec >= warm_end - min(6.0, max(2.0, cfg.get("warmupSeconds", 60) * 0.2))
                                and all(m in speeds for m in active_members)
                            )
                            or phase == "measuring"
                        ):
                            schedule = allocate_proportional_tokens(
                                batch_tokens, context, active_members, speeds, microbatch
                            )
                            locked_schedule = {
                                "activeMembers": active_members,
                                "schedule": schedule,
                                "speeds": {m: speeds[m] for m in active_members if m in speeds},
                                "updatedAt": get_dht_time(),
                            }
                            dht.store(
                                f"glmbench_{session_prefix}_schedule",
                                locked_schedule,
                                expiration_time=get_dht_time() + 60,
                            )
                        elif locked_schedule is not None:
                            dht.store(
                                f"glmbench_{session_prefix}_schedule",
                                {**locked_schedule, "updatedAt": get_dht_time()},
                                expiration_time=get_dht_time() + 60,
                            )
            except Exception:
                pass
            time.sleep(2)
    finally:
        dht.shutdown()


def agent(args, client):
    torch.set_num_threads(args.threads)
    client.calibrate_clock()
    state = client.get()
    session = state.get("session")
    if not session or state["session"]["phase"] != "lobby":
        raise RuntimeError("Ask the coordinator to create a fresh lobby first")
    instance = uuid.uuid4().hex
    user = state["user"]["id"] + ":" + instance
    config = dict(session["config"])
    if getattr(args, "local_steps", None) is not None:
        config["localSteps"] = args.local_steps
    if config.get("protocol") != 3:
        raise RuntimeError("Unsupported benchmark protocol")
    if (
        not 1 <= config["payloadElements"] <= getattr(args, "max_parameters", 25_000_000)
        or not 16 <= config["matrixSize"] <= 1024
    ):
        raise RuntimeError("Unexpected workload size or local parameter limit exceeded. Review the project and use --max-parameters to allow a larger model.")
    dht = start_dht(args.host, args.port, session["peers"])
    averager = None
    stop = threading.Event()
    heartbeat = None
    shared = {"state": state, "error": None}
    try:
        averaged = torch.zeros(config["payloadElements"], dtype=torch.float32)
        # Hivemind installs SIG_IGN while forking; restore Ctrl+C for the CLI.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        from .device import select_device, describe

        device, dtype = select_device(args.device, args.device_index)
        from .workloads import training_workload

        workload = training_workload(config, device, int(instance[:8], 16), getattr(args, "workload_factory", None))
        size = config["matrixSize"]
        left = torch.randn(size, size, device=device, dtype=dtype)
        right = torch.randn_like(left)

        def compute():
            if workload is not None:
                return 0.0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            for _ in range(10):
                result = left @ right
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return 20 * size**3 / (time.perf_counter() - before) / 1e9

        compute()
        client.post(
            action="register",
            sessionId=session["id"],
            instanceId=instance,
            device={
                **describe(device),
                "workload": config.get("workload", "synthetic"),
                "batchTokens": config.get("batchTokens", 32768),
                "trainingProfile": getattr(workload, "profile", "reference-fp32"),
                "microbatch": getattr(workload, "microbatch", 1),
                "localSteps": getattr(workload, "local_steps", 1),
                "compiledModel": getattr(workload, "compiled_model", False),
                "workloadFingerprint": workload.fingerprint
                if workload
                else "synthetic-v3",
            },
        )
        if args.ready:
            client.post(
                action="request_ready",
                sessionId=session["id"],
                instanceId=instance,
                ready=True,
            )

        def poll():
            errors = 0
            while not stop.is_set():
                try:
                    new = client.get()
                    if not new.get("session") or new["session"]["id"] != session["id"]:
                        raise RuntimeError("The lobby changed; restart this agent")
                    own = next(a for a in new["agents"] if a["userId"] == user)
                    client.post(
                        action="heartbeat",
                        sessionId=session["id"],
                        instanceId=instance,
                        ready=own["requestedReady"],
                    )
                    shared["state"] = new
                    errors = 0
                except Exception as error:
                    errors += 1
                    if errors >= 3:
                        shared["error"] = str(error)
                        return
                stop.wait(2)

        heartbeat = threading.Thread(target=poll, daemon=True)
        heartbeat.start()
        print(
            json.dumps(
                {
                    "event": "agent_connected",
                    "member": user,
                    "device": str(device),
                    "sessionId": session["id"],
                    "ready": "Use Mark ready on the dashboard, or --ready on this command",
                }
            ),
            flush=True,
        )
        while shared["state"]["session"].get("startsAt") is None:
            if shared["error"]:
                raise RuntimeError(shared["error"])
            if shared["state"]["session"].get("cancelledAt") is not None:
                raise RuntimeError("Lobby cancelled")
            time.sleep(0.1)
        session = shared["state"]["session"]
        if session["config"].get("workload", "synthetic") != config.get(
            "workload", "synthetic"
        ):
            raise RuntimeError("Workload changed. Restart this agent")
        config = dict(session["config"])
        if getattr(args, "local_steps", None) is not None:
            config["localSteps"] = args.local_steps
        if user not in session["expected"]:
            raise RuntimeError(
                "This computer was not included at Start. Join the next lobby"
            )
        count = len(session["expected"])
        if not 1 <= count <= 10:
            raise RuntimeError("Expected 1–10 benchmark computers")
        if getattr(workload, "compiled_model", False) and count != 1:
            raise RuntimeError("Compiled BF16 currently requires a single NVIDIA node")
        session_prefix = session["id"].replace("-", "")
        matchmaking_seconds = float(config.get("matchmakingSeconds", 1.0))
        request_timeout = min(matchmaking_seconds * 0.75, max(0.5, matchmaking_seconds - 0.2))
        dropout_timeout = float(config.get("dropoutSeconds", 20.0))

        def make_averager(group_size, generation=0):
            suffix = f"_g{generation}" if generation else ""
            return DecentralizedAverager(
                averaged_tensors=[averaged],
                dht=dht,
                start=True,
                prefix=f"glmbench_{session_prefix}{suffix}",
                target_group_size=group_size,
                min_group_size=2,
                min_matchmaking_time=matchmaking_seconds,
                request_timeout=request_timeout,
                allreduce_timeout=min(60.0, max(15.0, dropout_timeout)),
                part_size_bytes=getattr(workload, "gradient_part_size", 512 * 1024),
                compression=Float16Compression(),
                allow_state_sharing=False,
            )

        averager_generation = 0
        if count > 1:
            averager = make_averager(count, averager_generation)
        measure = Measurements(session)
        measured_tokens = [0]
        measured_losses = []
        active_members = list(session["expected"])
        context_len = workload.cfg.context if workload is not None else config.get("model", {}).get("context", 512)
        batch_tokens = config.get("batchTokens", 32768)
        nominal_local_tokens = (
            math.ceil(batch_tokens / (count * context_len)) * context_len
            if workload is not None
            else 0
        )
        assigned_local_tokens = nominal_local_tokens
        warmup_speeds = []
        frozen_local_speed = None
        peer_speeds = {}
        schedule_locked = False
        peer_last_seen = {}

        def record(start, end, latency, members, gflops):
            before = len(measure.latencies)
            measure.record(start, end, latency, members, gflops, expected_members=active_members)
            if workload is not None and len(measure.latencies) > before:
                measured_tokens[0] += workload.local_tokens
                measured_losses.append(workload.last_loss)

        warm_end = session["startsAt"] / 1000 + config["warmupSeconds"]
        finish = warm_end + config["measureSeconds"]
        freeze_at = warm_end - min(6.0, max(2.0, config["warmupSeconds"] * 0.2))

        def refresh_backend_liveness(now_sec):
            if phase_at(session, now_sec) == "measuring":
                return
            agents_list = shared["state"].get("agents", [])
            for entry in agents_list:
                uid = entry.get("userId")
                last_ms = entry.get("lastSeenAt")
                if uid in active_members and isinstance(last_ms, (int, float)):
                    candidate = last_ms / 1000.0
                    if uid not in peer_last_seen or candidate > peer_last_seen[uid]:
                        peer_last_seen[uid] = candidate

        def apply_schedule(reason):
            nonlocal assigned_local_tokens
            if workload is None:
                return
            dht_sched = dht.get(f"glmbench_{session_prefix}_schedule", latest=True)
            schedule = None
            if (
                dht_sched is not None
                and isinstance(dht_sched.value, dict)
                and set(dht_sched.value.get("activeMembers", [])) == set(active_members)
                and isinstance(dht_sched.value.get("schedule"), dict)
                and user in dht_sched.value["schedule"]
            ):
                candidate = dht_sched.value["schedule"]
                if all(
                    isinstance(candidate.get(m), int)
                    and candidate[m] > 0
                    and candidate[m] % context_len == 0
                    for m in active_members
                ):
                    schedule = {m: candidate[m] for m in active_members}
            if schedule is None:
                dht_speeds, _ = _read_dht_speeds(dht, session_prefix)
                merged_speeds = {**dht_speeds, **peer_speeds}
                schedule = allocate_proportional_tokens(
                    batch_tokens,
                    context_len,
                    active_members,
                    merged_speeds,
                    workload.microbatch,
                )
            assigned_local_tokens = schedule[user]
            print(
                json.dumps(
                    {
                        "event": reason,
                        "member": user,
                        "localBatchTokens": assigned_local_tokens,
                        "speeds": {m: round(peer_speeds[m], 2) for m in active_members if m in peer_speeds},
                        "schedule": schedule,
                        "activeMembers": active_members,
                    }
                ),
                flush=True,
            )

        def check_and_reroute_dropouts(now_sec):
            nonlocal averager, averager_generation
            if len(active_members) <= 1:
                return False
            refresh_backend_liveness(now_sec)
            _, dht_ts = _read_dht_speeds(dht, session_prefix)
            for m, ts in dht_ts.items():
                if m in active_members and (m not in peer_last_seen or ts > peer_last_seen[m]):
                    peer_last_seen[m] = ts
            dropped = [
                m
                for m in active_members
                if m != user and now_sec - peer_last_seen.get(m, now_sec) > dropout_timeout
            ]
            if not dropped:
                return False
            for m in dropped:
                active_members.remove(m)
                if m not in measure.dropped_members:
                    measure.dropped_members.append(m)
            if averager is not None:
                averager.shutdown()
                averager = None
            if len(active_members) >= 2:
                averager_generation += 1
                averager = make_averager(len(active_members), averager_generation)
            apply_schedule("peer_dropout_rerouted")
            return True

        while client.now() < session["startsAt"] / 1000:
            if shared["error"]:
                raise RuntimeError(shared["error"])
            time.sleep(0.05)
        observed_mono_start = time.monotonic()
        start_clock = client.now()
        for m in active_members:
            peer_last_seen[m] = start_clock
        measure.observe(start_clock)
        print(
            json.dumps(
                {
                    "event": "warmup_started",
                    "warmupSeconds": config["warmupSeconds"],
                    "measureSeconds": config["measureSeconds"],
                }
            ),
            flush=True,
        )
        pending_gradients = None
        consecutive_failures = 0
        while client.now() < finish:
            if shared["error"]:
                measure.cancelled = True
                break
            if shared["state"]["session"].get("cancelledAt") is not None:
                measure.cancelled = True
                break
            start = client.now()
            measure.clock_ms = max(measure.clock_ms, client.uncertainty_ms)
            if finish - start <= matchmaking_seconds + 0.1:
                break
            check_and_reroute_dropouts(start)
            current_phase = phase_at(session, start)
            if (
                not schedule_locked
                and workload is not None
                and current_phase == "measuring"
            ):
                schedule_locked = True
                for m in active_members:
                    peer_last_seen[m] = start
                apply_schedule("proportional_work_assigned")
                pending_gradients = None
            if (
                pending_gradients is not None
                and workload is not None
                and workload.local_tokens
                != assigned_local_tokens * getattr(workload, "local_steps", 1)
            ):
                pending_gradients = None
            gflops = compute()
            if pending_gradients is None:
                gradients = (
                    workload.prepare(
                        len(active_members),
                        local_tokens=assigned_local_tokens if workload is not None else None,
                    )
                    if workload is not None
                    else None
                )
            else:
                gradients = pending_gradients
            now_after_prepare = client.now()
            if workload is not None and current_phase == "warmup":
                warmup_speeds.append(workload.last_compute_tps)
                if len(warmup_speeds) > 5:
                    del warmup_speeds[:-5]
                if now_after_prepare < freeze_at or frozen_local_speed is None:
                    frozen_local_speed = statistics.median(warmup_speeds)
                peer_speeds[user] = frozen_local_speed
                try:
                    dht.store(
                        f"glmbench_{session_prefix}_speeds",
                        {"compute_tps": round(frozen_local_speed, 2), "ts": now_after_prepare},
                        subkey=user,
                        expiration_time=get_dht_time() + 60,
                    )
                except Exception:
                    pass
            if finish - now_after_prepare <= matchmaking_seconds + 0.2:
                measure.boundary += 1
                break
            if averager is not None:
                with averager.get_tensors() as tensors:
                    if gradients is not None:
                        tensors[0].copy_(gradients)
                    else:
                        tensors[0].fill_(session["expected"].index(user) + 1)
            began = time.perf_counter()
            control = None
            try:
                if averager is None:
                    # A single computer measures compute only. No network result
                    # or successful collective communication is fabricated.
                    if workload is not None:
                        workload.apply(gradients)
                    record(start, client.now(), 0, active_members, gflops)
                    consecutive_failures = 0
                    pending_gradients = None
                    time.sleep(0.01)
                    continue
                gather_payload = {"member": user, "session": session["id"]}
                if frozen_local_speed is not None:
                    gather_payload["compute_tps"] = round(frozen_local_speed, 2)
                step_timeout = min(
                    max(5.0, dropout_timeout),
                    finish - client.now() - 0.01,
                )
                control = averager.step(
                    gather=gather_payload,
                    weight=float(workload.local_tokens) if workload is not None else None,
                    timeout=step_timeout,
                    wait=False,
                    allow_retries=False,
                )
                rerouted_mid_step = False
                while not control.done():
                    now_wait = client.now()
                    if (
                        now_wait >= finish
                        or shared["error"]
                        or shared["state"]["session"].get("cancelledAt") is not None
                    ):
                        control.cancel()
                        break
                    if check_and_reroute_dropouts(now_wait):
                        control.cancel()
                        rerouted_mid_step = True
                        break
                    time.sleep(0.02)
                if rerouted_mid_step:
                    pending_gradients = None
                    consecutive_failures = 0
                    continue
                group = control.result(timeout=2)
                now_done = client.now()
                members = []
                for entry in group.values():
                    if entry.get("session") == session["id"]:
                        m_id = entry.get("member")
                        members.append(m_id)
                        peer_last_seen[m_id] = now_done
                        tps = entry.get("compute_tps")
                        if (
                            not schedule_locked
                            and isinstance(tps, (int, float))
                            and math.isfinite(tps)
                            and tps > 0
                        ):
                            peer_speeds[m_id] = float(tps)
                if sorted(members) != sorted(active_members):
                    raise RuntimeError("Incomplete gradient averaging group")
                if workload is not None:
                    with averager.get_tensors() as tensors:
                        workload.apply(tensors[0])
                pending_gradients = None
                consecutive_failures = 0
                record(
                    start,
                    now_done,
                    (time.perf_counter() - began) * 1000,
                    members,
                    gflops,
                )
            except Exception:
                if control is not None and not control.done():
                    control.cancel()
                now_err = client.now()
                rerouted = check_and_reroute_dropouts(now_err)
                if rerouted:
                    pending_gradients = None
                    consecutive_failures = 0
                    continue
                awaiting_dropout = len(active_members) > 1 and any(
                    m != user
                    and now_err - peer_last_seen.get(m, start_clock) >= matchmaking_seconds * 0.8
                    for m in active_members
                )
                if awaiting_dropout and now_err < finish - 0.25:
                    if workload is not None:
                        pending_gradients = gradients
                    time.sleep(0.25)
                    continue
                if (
                    phase_at(session, start) == "measuring"
                    and now_err < finish - 0.25
                ):
                    measure.failed += 1
                    consecutive_failures += 1
                else:
                    measure.boundary += 1
                if workload is not None:
                    pending_gradients = gradients
                    if consecutive_failures >= 5:
                        break
                    time.sleep(0.25)
                else:
                    time.sleep(0.1)
        if not measure.cancelled:
            while client.now() < finish:
                time.sleep(0.05)
        measure.observe(client.now())
        measure.clock_ms = max(measure.clock_ms, client.uncertainty_ms)
        result = constrain_duration(
            measure.result(), config, time.monotonic() - observed_mono_start
        )
        result["workload"] = config.get("workload", "synthetic")
        if measure.dropped_members:
            result["droppedParticipants"] = measure.dropped_members
            result["activeParticipants"] = len(active_members)
        if workload is not None:
            local_warmup_tps = round(frozen_local_speed, 2) if frozen_local_speed is not None else 0.0
            result.update(
                trainingProfile=workload.profile,
                precision=workload.precision,
                microbatch=workload.microbatch,
                localSteps=getattr(workload, "local_steps", 1),
                gradientPartSizeBytes=workload.gradient_part_size,
                compiledModel=getattr(workload, "compiled_model", False),
                compileSetupSeconds=getattr(workload, "compile_seconds", 0.0),
                parameterCount=workload.parameter_count,
                processedTokens=measured_tokens[0],
                tokensPerSecond=measured_tokens[0] / result["measuredSeconds"]
                if result["measuredSeconds"]
                else 0,
                localBatchTokens=assigned_local_tokens,
                nominalLocalBatchTokens=nominal_local_tokens,
                assignedBatchTokens=assigned_local_tokens,
                actualProcessedTokens=measured_tokens[0],
                warmupTokensPerSecond=local_warmup_tps,
                workShare=round(assigned_local_tokens / batch_tokens, 4) if batch_tokens else 1.0,
                activeParticipants=len(active_members),
                droppedParticipants=measure.dropped_members,
                optimizerSteps=workload.steps,
                meanLoss=statistics.mean(measured_losses) if measured_losses else 0,
                modelFingerprint=workload.fingerprint,
                note=("Locally selected external workload adapter. Timing includes the work implemented by that adapter. " if config.get("runtime") == "external" else f"Real {workload.precision.upper()} model forward/backward with FP32 weights and AdamW updates on generated token batches. Includes transfer and optimizer overhead; excludes corpus I/O, evaluation and checkpoint saving.")
                + (f" Synchronized every {getattr(workload, 'local_steps', 1)} local steps." if getattr(workload, "local_steps", 1) > 1 else "")
                + (f" Rerouted work after dropout of {len(measure.dropped_members)} peer(s)." if measure.dropped_members else "")
                + (" Incomplete/degraded run." if not result["complete"] else ""),
            )
        if count == 1:
            result["note"] += (
                " Single computer: no peer averaging or network throughput was measured."
            )
        path = Path(args.output) / session["id"] / (user.replace(":", "_") + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": session["id"],
            "instanceId": instance,
            "member": user,
            "result": result,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        # Clock estimates can be slightly ahead; wait for the server's finished state.
        deadline = time.monotonic() + 60
        upload_result = dict(result)
        while True:
            try:
                client.post(
                    action="result",
                    sessionId=session["id"],
                    instanceId=instance,
                    result=upload_result,
                )
                break
            except RuntimeError as error:
                if (
                    "400:" in str(error)
                    and "frozen model/batch contract" in str(error)
                    and workload is not None
                    and (
                        upload_result.get("localBatchTokens") != nominal_local_tokens
                        or upload_result.get("processedTokens") != result["measuredRounds"] * nominal_local_tokens
                    )
                ):
                    upload_result = {
                        **result,
                        "localBatchTokens": nominal_local_tokens,
                        "processedTokens": result["measuredRounds"] * nominal_local_tokens,
                    }
                    continue
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Upload failed; your result is preserved in {path}"
                    ) from None
                time.sleep(2)
        print(
            json.dumps({"event": "result_saved", "path": str(path), "result": result}),
            flush=True,
        )
    finally:
        stop.set()
        if heartbeat:
            heartbeat.join(timeout=16)
        try:
            if averager:
                averager.shutdown()
        finally:
            dht.shutdown()
