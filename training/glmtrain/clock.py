"""Application clock: rendezvous epoch plus elapsed monotonic time, never OS wall time."""
import importlib
import multiprocessing
import sys
import time


class NetworkClock:
    def __init__(self, epoch, monotonic):
        self._shared = multiprocessing.RawArray("d", 3)
        self.update(epoch, monotonic)

    @property
    def epoch(self):
        return self._shared[0]

    @epoch.setter
    def epoch(self, value):
        val = float(value)
        self._shared[0] = val
        self._shared[2] = val - self._shared[1]

    @property
    def monotonic(self):
        return self._shared[1]

    @monotonic.setter
    def monotonic(self, value):
        val = float(value)
        self._shared[1] = val
        self._shared[2] = self._shared[0] - val

    def update(self, epoch, monotonic):
        e = float(epoch)
        m = float(monotonic)
        self._shared[0] = e
        self._shared[1] = m
        self._shared[2] = e - m

    def __call__(self):
        return self._shared[2] + time.monotonic()


def install_hivemind_clock(clock):
    # Hivemind imports its clock by value in multiple modules. Pre-import known
    # Hivemind submodules and replace references before starting DHT/averager
    # children. Linux fork preserves the shared-memory clock in every child;
    # unsupported process modes fail closed.
    if multiprocessing.get_start_method() != "fork":
        raise RuntimeError("The synchronized Hivemind clock requires Linux fork workers")
    for mod_name in (
        "hivemind.utils",
        "hivemind.utils.timed_storage",
        "hivemind.dht",
        "hivemind.dht.node",
        "hivemind.averaging",
        "hivemind.optim",
    ):
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass
    import hivemind.utils

    original = hivemind.utils.get_dht_time
    for name, module in list(sys.modules.items()):
        if name == "hivemind" or name.startswith("hivemind."):
            for attr, value in list(vars(module).items()):
                if attr == "get_dht_time" and (
                    value is original
                    or isinstance(value, NetworkClock)
                    or getattr(value, "__name__", None) == "get_dht_time"
                ):
                    setattr(module, attr, clock)
