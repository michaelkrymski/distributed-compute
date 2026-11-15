"""Direct peer networking, intended for a trusted LAN or private mesh VPN."""

import ipaddress
import json
import math
import platform
import threading
import time
from pathlib import Path
import hivemind
from hivemind.utils import get_dht_time
from .clock import NetworkClock, install_hivemind_clock


def calibrate_clock():
    """Install a fork-safe monotonic clock before starting Hivemind children."""
    clock = NetworkClock(time.time(), time.monotonic())
    install_hivemind_clock(clock)
    return clock


def start_dht(host, port, peers=(), identity=None):
    if platform.system() == "Windows":
        raise RuntimeError(
            "Run the training process in Linux/WSL2; native Windows is not supported by this starter"
        )
    address = ipaddress.ip_address(host)
    if address.is_unspecified:
        raise ValueError(
            "Bind to a specific LAN/VPN IP, or 127.0.0.1 for local tests; do not use 0.0.0.0"
        )
    if not 0 <= port <= 65535:
        raise ValueError("port must be in [0,65535]")
    if identity:
        Path(identity).parent.mkdir(parents=True, exist_ok=True)
    return hivemind.DHT(
        start=True,
        initial_peers=list(peers),
        host_maddrs=[f"/ip{address.version}/{address}/tcp/{port}"],
        identity_path=str(identity) if identity else None,
        nat_port_map=False,
        auto_nat=False,
        use_relay=False,
    )


def key(manifest, suffix):
    return f"gatorllm.{manifest['run_id']}.{suffix}"


def workers(dht, manifest):
    found = dht.get(key(manifest, "workers"), latest=True)
    if found is None or not isinstance(found.value, dict):
        return {}
    return {
        str(name): record.value
        for name, record in found.value.items()
        if isinstance(record.value, dict)
        and record.value.get("fingerprint") == manifest["fingerprint"]
    }


def check_contract(dht, manifest, timeout=30, clock=None):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        contract = dht.get(key(manifest, "contract"), latest=True)
        if contract:
            if contract.value != manifest["fingerprint"]:
                raise ValueError("Rendezvous and worker have different run bundles")
            if clock is not None:
                begin = time.monotonic()
                remote = dht.get(key(manifest, "clock"), latest=True)
                end = time.monotonic()
                if (
                    remote is not None
                    and isinstance(remote.value, (int, float))
                    and math.isfinite(remote.value)
                ):
                    corrected = float(remote.value) + (end - begin) / 2
                    if abs(corrected - clock()) > 0.5:
                        clock.update(corrected, end)
            return clock
        time.sleep(0.5)
    raise RuntimeError(
        "No matching rendezvous contract found. Check --peer, the run bundle, and VPN connectivity"
    )


class Heartbeat:
    def __init__(self, callback, interval=10):
        self.callback, self.interval = callback, interval
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.callback()
            except Exception as error:
                self.error = error
                return
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def check(self):
        if self.error:
            raise RuntimeError("DHT heartbeat failed") from self.error

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=15)


def bootstrap(args, manifest):
    clock = calibrate_clock()
    dht = start_dht(args.host, args.port, identity=args.identity)
    try:
        peers = [str(a) for a in dht.get_visible_maddrs()]
        if args.peers_file:
            path = Path(args.peers_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(peers) + "\n")
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "initial_peers": peers,
                    "role": "rendezvous only; no training or checkpoints",
                }
            ),
            flush=True,
        )

        def announce():
            now = clock()
            if not dht.store(
                key(manifest, "contract"),
                manifest["fingerprint"],
                expiration_time=now + 120,
            ):
                raise RuntimeError("Could not publish run contract")
            dht.store(
                key(manifest, "clock"),
                float(now) + 0.25,
                expiration_time=now + 120,
            )

        announce()
        with Heartbeat(announce) as heartbeat:
            while True:
                heartbeat.check()
                now = clock()
                dht.store(
                    key(manifest, "clock"),
                    float(now) + 0.25,
                    expiration_time=now + 120,
                )
                time.sleep(0.5)
    finally:
        dht.shutdown()
