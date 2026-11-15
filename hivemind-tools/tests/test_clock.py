import multiprocessing
import time
from unittest.mock import patch

import pytest
from glmbench.clock import NetworkClock, install_hivemind_clock
from glmbench.client import Client


def child_clock(connection):
    from hivemind.dht.node import get_dht_time
    connection.send(get_dht_time())
    connection.close()


def test_backend_clock_reaches_real_child_process_despite_wrong_os_clock():
    epoch=1700000000.0
    clock=NetworkClock(epoch, time.monotonic())
    with patch('time.time', return_value=epoch+3600):
        install_hivemind_clock(clock)
        parent, child=multiprocessing.Pipe()
        worker=multiprocessing.Process(target=child_clock,args=(child,))
        worker.start()
        assert parent.poll(10)
        assert abs(parent.recv()-epoch) < 10
        worker.join(10)
        assert worker.exitcode == 0
        parent.close();child.close()


def test_active_deadlines_do_not_move_when_new_samples_arrive(monkeypatch):
    client=Client('http://127.0.0.1','fixture')
    monkeypatch.setattr(time,'monotonic',lambda:10)
    client.synchronize(100000,10,10)
    client._run_clock=NetworkClock(*client._anchor)
    monkeypatch.setattr(time,'monotonic',lambda:20)
    client.synchronize(110500,20,20)
    assert client.now() == 110
    assert client.uncertainty_ms >= 500
    client.synchronize(115000,20,20)
    assert client.now() == 110
    assert client.uncertainty_ms > 1000


def test_shared_memory_clock_update_reaches_forked_child():
    epoch = 1700000000.0
    mono = time.monotonic()
    clock = NetworkClock(epoch, mono)
    install_hivemind_clock(clock)
    clock.update(epoch + 500.0, mono)
    parent, child = multiprocessing.Pipe()
    worker = multiprocessing.Process(target=child_clock, args=(child,))
    worker.start()
    assert parent.poll(10)
    assert abs(parent.recv() - (epoch + 500.0)) < 10
    worker.join(10)
    assert worker.exitcode == 0
    parent.close()
    child.close()
