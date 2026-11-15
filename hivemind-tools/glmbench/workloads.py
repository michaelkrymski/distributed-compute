"""Load a supplied training workload without bundling a model in this package."""
import importlib


def training_workload(config, device, seed):
    if config.get("workload", "synthetic") != "baseline-training":
        return None
    try:
        module = importlib.import_module("glmtrain.benchmark")
    except ImportError as error:
        raise RuntimeError(
            "Model benchmark requires the theoretical branch. Install both packages: "
            "python -m pip install -e ./training -e ./hivemind-tools"
        ) from error
    workload = module.TrainingWorkload(config, device, seed)
    if workload.parameter_count != config["payloadElements"]:
        raise RuntimeError("Model parameter count does not match the benchmark contract")
    return workload
