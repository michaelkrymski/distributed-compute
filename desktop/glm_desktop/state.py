"""Presentation logic for existing API responses and local worker events."""
import math


def finite(value, fallback=0):
    return value if type(value) in (int, float) and math.isfinite(value) else fallback


def benchmark_view(state, now_ms):
    session = state.get("session") or {}
    phase = session.get("phase", "no session")
    config = session.get("config") or {}
    start = finite(session.get("startsAt"))
    warm = finite(config.get("warmupSeconds"), 60)
    measured = finite(config.get("measureSeconds"), 300)
    elapsed = max(0, (now_ms - start) / 1000) if start else 0
    fraction = min(1, max(0, (elapsed - warm) / max(1, measured))) if phase in ("measuring", "finished") else 0
    remaining = max(0, (start - now_ms) / 1000) if phase == "countdown" else max(0, warm - elapsed) if phase == "warmup" else max(0, warm + measured - elapsed) if phase == "measuring" else 0
    agents = state.get("agents") or []
    participants = session.get("expected") or []
    reports = [a for a in agents if a.get("userId") in participants and a.get("result")]
    complete = bool(participants) and len(reports) == len(participants) and all(a["result"].get("complete") is True for a in reports)
    rate = 0
    if complete:
        results = [a["result"] for a in reports]
        seconds = max(finite(r.get("measuredSeconds")) for r in results)
        if seconds > 0:
            rate = min(finite(r.get("measuredRounds")) for r in results) * sum((finite(r.get("processedTokens")) / finite(r.get("measuredRounds"))) if finite(r.get("measuredRounds")) > 0 and "processedTokens" in r else finite(r.get("localBatchTokens")) * finite(r.get("localSteps"), 1) for r in results) / seconds
    rows = []
    for agent in agents:
        online = now_ms - finite(agent.get("lastSeenAt")) < 20000
        device = agent.get("device") or {}
        status = "Offline" if not online else "Ready" if agent.get("ready") else "Getting ready" if agent.get("requestedReady") else "Connected"
        rows.append(dict(name=device.get("hostname") or agent.get("userId", "Computer").split(":")[0], device=device.get("name", "Unknown device"), status=status, ready=bool(agent.get("ready")) and online))
    return dict(rate=rate, phase=phase, progress=fraction, remaining=math.ceil(remaining), rows=rows,
        summary=f"{len(agents)} computers · {sum(r['ready'] for r in rows)} ready",
        result=f"{rate:,.0f} tokens/sec combined" if rate else "Incomplete reports — estimate unavailable" if phase == "finished" and reports else "Results appear after measurement",
        workload=f"{config.get('batchTokens', 32768):,} tokens per exchange · {config.get('trainingProfile', 'reference-fp32')}" if session else "Waiting for an organizer to open a session")


def training_view(row, target_tokens, target_epochs):
    tokens = finite(row.get("nominal_global_tokens", row.get("nominalGlobalTokens")))
    epoch = finite(row.get("epoch"))
    fraction = tokens / target_tokens if target_tokens else epoch / target_epochs if target_epochs else 0
    loss = row.get("loss")
    return dict(progress=max(0, min(1, fraction)), tokens=tokens, epoch=epoch,
        summary=f"Epoch {epoch:,.0f} · {tokens:,.0f} nominal global tokens",
        loss=f"{loss:.4f}" if type(loss) in (float, int) and math.isfinite(loss) else "—")
