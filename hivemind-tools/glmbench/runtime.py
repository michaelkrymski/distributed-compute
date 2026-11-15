"""Hivemind networking primitives, independent of any particular model."""

import ipaddress
import platform
from pathlib import Path
import hivemind


def start_dht(host, port, peers=(), identity=None):
    if platform.system() == "Windows":
        raise RuntimeError("Run in Linux or WSL2, not native Windows")
    address = ipaddress.ip_address(host)
    if address.is_unspecified:
        raise ValueError("Bind to your specific LAN/VPN IP, not 0.0.0.0")
    if not 0 <= port <= 65535:
        raise ValueError("Invalid port")
    if identity:
        Path(identity).parent.mkdir(parents=True, exist_ok=True)
    return hivemind.DHT(
        start=True,
        startup_timeout=60,
        # Bootstrap precedes the Python control connection. The default 30s
        # daemon idle timer can otherwise kill a peer that is still starting.
        idle_timeout=120,
        initial_peers=list(peers),
        host_maddrs=[f"/ip{address.version}/{address}/tcp/{port}"],
        identity_path=identity,
        nat_port_map=False,
        auto_nat=False,
        use_relay=False,
    )


def collaborative_optimizer(
    dht,
    run_id,
    parameters,
    optimizer_factory,
    *,
    local_batch_tokens,
    target_batch_tokens,
    matchmaking_time=10,
):
    """Optional adapter for a separately supplied model/training loop.

    Call zero_grad(), loss.backward(), then step() for each local microbatch.
    Local loss must be a mean over local_batch_tokens. No model is constructed.
    """
    from hivemind.compression import Float16Compression

    if min(local_batch_tokens, target_batch_tokens) <= 0:
        raise ValueError("Batch token counts must be positive")
    return hivemind.Optimizer(
        dht=dht,
        run_id=run_id,
        params=parameters,
        optimizer=optimizer_factory,
        batch_size_per_step=local_batch_tokens,
        target_batch_size=target_batch_tokens,
        offload_optimizer=False,
        reuse_grad_buffers=False,
        use_local_updates=False,
        delay_optimizer_step=False,
        delay_grad_averaging=False,
        delay_state_averaging=False,
        grad_compression=Float16Compression(),
        state_averaging_compression=Float16Compression(),
        matchmaking_time=matchmaking_time,
        averaging_timeout=120,
        averager_opts={"request_timeout": min(3, matchmaking_time / 2)},
    )
