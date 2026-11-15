"""Example local factory for testing the external-workload extension point.

The operator selects this function explicitly with --workload-factory; the
coordination service cannot choose Python imports or execute remote source.
"""
from .benchmark import TrainingWorkload


def decoder(config, device, seed):
    return TrainingWorkload({**config, "model": config["adapterConfig"]["model"]}, device, seed)
