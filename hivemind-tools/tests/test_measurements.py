from glmbench.benchmark import (
    Measurements,
    allocate_proportional_tokens,
    constrain_duration,
    phase_at,
)
from glmbench.client import Client
import pytest


def session():
    return {
        "startsAt": 100000,
        "cancelledAt": None,
        "expected": ["a", "b"],
        "config": {"warmupSeconds": 60, "measureSeconds": 300},
    }


def test_warmup_and_boundary_rounds_excluded():
    m = Measurements(session())
    m.record(120, 121, 1000, ["a", "b"], 10)
    m.record(159, 161, 2000, ["a", "b"], 10)
    m.record(161, 162, 1000, ["a", "b"], 10)
    m.record(459, 461, 2000, ["a", "b"], 10)
    r = m.result()
    assert r["warmupRounds"] == 2
    assert r["measuredRounds"] == 1
    assert r["boundaryRounds"] == 1
    assert r["latencyP50Ms"] == 1000


def test_partial_or_duplicate_groups_are_failures():
    m = Measurements(session())
    m.record(161, 162, 1000, ["a"], 10)
    m.record(163, 164, 1000, ["a", "a"], 10)
    r = m.result()
    assert r["measuredRounds"] == 0
    assert r["failedRounds"] == 2
    assert not r["complete"]


def test_no_credentials_in_urls_and_no_plaintext_remote_http():
    with pytest.raises(ValueError):
        Client("http://example.com", "not-a-real-key")
    with pytest.raises(ValueError):
        Client("https://key@example.com", "not-a-real-key")


def test_partial_window_does_not_claim_five_measured_minutes():
    m = Measurements(session())
    m.observe(100)
    m.record(161, 162, 1000, ["a", "b"], 10)
    m.observe(180)
    m.cancelled = True
    result = m.result()
    assert result["measuredSeconds"] == 20
    assert not result["complete"]


def test_server_clock_jump_is_retained_after_later_good_samples():
    c = Client("http://127.0.0.1", "unit-test-placeholder")
    c.synchronize(100000, 10, 10)
    c.synchronize(105000, 15, 15)
    assert c.uncertainty_ms == 0
    c.synchronize(112000, 20, 20)  # Server jumps forward two seconds.
    assert c.clock_jump_ms == 2000
    c.synchronize(117000, 25, 25)
    assert c.uncertainty_ms == 2000


def test_slow_response_does_not_replace_a_recent_precise_clock(monkeypatch):
    c = Client("http://127.0.0.1", "unit-test-placeholder")
    c.synchronize(100000, 10, 10.02)
    c.synchronize(102000, 12, 14.2)  # 2.2-second response delay, no clock jump.
    monkeypatch.setattr("glmbench.client.time.monotonic", lambda: 14.2)
    assert abs(c.now() - 104.2) < 0.02
    assert c.uncertainty_ms < 20
    assert c.clock_jump_ms == 0


def test_old_clock_samples_expire_instead_of_hiding_persistent_latency():
    c = Client("http://127.0.0.1", "unit-test-placeholder")
    c.synchronize(100000, 10, 10.02)
    c.synchronize(140000, 50, 52.4)
    assert c.uncertainty_ms > 1000


def test_invalid_clock_sample_is_rejected():
    c = Client("http://127.0.0.1", "unit-test-placeholder")
    for sample in ((float("nan"), 10, 11), (100000, 11, 10), ("invalid", 10, 11)):
        with pytest.raises(RuntimeError, match="Invalid backend clock"):
            c.synchronize(*sample)


def test_wall_clock_cannot_fabricate_a_full_measured_window():
    report = {"complete": True, "measuredSeconds": 300, "note": "Synthetic test."}
    result = constrain_duration(report, session()["config"], 350)
    assert result["measuredSeconds"] == 290
    assert result["observedElapsedSeconds"] == 350
    assert not result["complete"]


def test_measurements_support_rerouted_active_members():
    m = Measurements(session())
    m.record(161, 162, 500, ["a"], 10, expected_members=["a"])
    r = m.result()
    assert r["measuredRounds"] == 1
    assert r["failedRounds"] == 0


def test_glmbench_proportional_allocation():
    alloc = allocate_proportional_tokens(98304, 512, ["gpu0", "gpu1"], {"gpu0": 90000.0, "gpu1": 45000.0})
    assert sum(alloc.values()) == 98304
    assert alloc["gpu0"] > alloc["gpu1"]
