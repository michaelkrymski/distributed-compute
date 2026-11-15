import json
import math
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse


class Client:
    def __init__(self, base_url, key):
        url = urlparse(base_url)
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError(
                "Use HTTPS for the backend; HTTP is allowed only on loopback for tests"
            )
        if not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError(
                "Provide a plain backend origin without credentials or query parameters"
            )
        self.url = base_url.rstrip("/") + "/api/benchmark"
        self.key = key
        self._anchor = (time.time(), time.monotonic())
        self._synced = False
        self._rtt_ms = 0.0
        self.clock_jump_ms = 0.0
        self.uncertainty_ms = float("inf")
        self._clock_samples = []
        self._clock_lock = threading.Lock()
        self._run_clock = None
        self._run_uncertainty_ms = 0.0

    def calibrate_clock(self):
        """Choose a precise sample before any peer workers are forked."""
        from .clock import NetworkClock, install_hivemind_clock
        for _ in range(5):
            self.get()
        if self.uncertainty_ms > 1000:
            raise RuntimeError("Backend clock calibration is too uncertain; retry before joining")
        with self._clock_lock:
            self._run_clock = NetworkClock(*self._anchor)
            self._run_uncertainty_ms = self.uncertainty_ms
        install_hivemind_clock(self._run_clock)

    def now(self):
        if self._run_clock is not None:
            return self._run_clock()
        server, mono = self._anchor
        return server + (time.monotonic() - mono)

    def synchronize(self, server_time_ms, begin, end):
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (server_time_ms, begin, end)) or end < begin:
            raise RuntimeError("Invalid backend clock sample")
        corrected = server_time_ms / 1000 + (end - begin) / 2
        rtt_ms = (end - begin) * 500
        with self._clock_lock:
            if self._synced:
                server, mono = self._anchor
                correction_ms = abs(corrected - (server + end - mono)) * 1000
                if correction_ms > 1000 + self._rtt_ms + rtt_ms:
                    self.clock_jump_ms = max(self.clock_jump_ms, correction_ms)
                    self._clock_samples.clear()
            self._clock_samples.append((corrected, end, rtt_ms))
            latest = max(sample[1] for sample in self._clock_samples)
            self._clock_samples = [s for s in self._clock_samples if latest - s[1] <= 30]
            # A slow HTTP response is a worse clock sample, not evidence that
            # the clock actually changed. Prefer a recent low-latency sample,
            # allowing a conservative 1ms/second aging penalty and a 30s TTL.
            def uncertainty(sample):
                return sample[2] + latest - sample[1]
            best = min(self._clock_samples, key=lambda s: (uncertainty(s), -s[1]))
            # Keep the pair atomic for the worker reading now() concurrently.
            self._anchor = best[:2]
            self._synced = True
            self._rtt_ms = best[2]
            self.uncertainty_ms = max(uncertainty(best), self.clock_jump_ms)
            if self._run_clock is not None:
                # Never move an active run's deadlines. Continue checking its
                # agreement with the server, rejecting drift or server jumps.
                deviation = abs(self._run_clock.epoch + latest - self._run_clock.monotonic - (best[0] + latest - best[1])) * 1000
                self.uncertainty_ms = max(self.uncertainty_ms, deviation + uncertainty(best), self._run_uncertainty_ms)

    def request(self, body=None):
        # Reads and heartbeats can be repeated safely. Never replay lobby creation
        # or another state-changing action after an ambiguous network failure.
        safe = body is None or body.get("action") in ("heartbeat", "coordinator_heartbeat")
        for attempt in range(3 if safe else 1):
            try:
                return self._request_once(body)
            except (TimeoutError, urllib.error.URLError, ConnectionError):
                if not safe or attempt == 2:
                    raise RuntimeError(
                        "Backend connection timed out or failed; "
                        "safe requests were retried when applicable"
                    ) from None
                print(json.dumps({"event": "backend_retry", "attempt": attempt + 1}), flush=True)
                time.sleep(attempt + 1)

    def _request_once(self, body=None):
        headers = {
            "User-Agent": "glmbench/0.4.0",
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        begin = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            try:
                message = json.load(error).get("error", "Request rejected")
            except Exception:
                message = (
                    "Non-JSON response from the hosting layer; "
                    "the request may have been blocked before reaching the API"
                )
            finally:
                error.close()
            raise RuntimeError(f"Backend returned {error.code}: {message}") from None
        end = time.monotonic()
        if "serverTime" in result:
            self.synchronize(result["serverTime"], begin, end)
        return result

    def get(self):
        return self.request()

    def post(self, **body):
        return self.request(body)
