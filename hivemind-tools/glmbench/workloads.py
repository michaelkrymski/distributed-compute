"""Load a supplied training workload without bundling a model in this package."""
import importlib
import re


def training_workload(config, device, seed, factory=None):
    if config.get("workload", "synthetic") != "baseline-training":
        return None
    if config.get("runtime") == "external":
        if not factory or not re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", factory):
            raise RuntimeError("This project needs your local workload adapter. Pass --workload-factory module:function explicitly after reviewing its code.")
        module_name, attribute = factory.split(":")
        workload = getattr(importlib.import_module(module_name), attribute)(config, device, seed)
        if workload.parameter_count != config["payloadElements"] or workload.cfg.context != config["model"]["context"] or not re.fullmatch(r"[a-f0-9]{64}", workload.fingerprint):
            raise RuntimeError("External workload does not match the project's parameter count, context or fingerprint contract")
        return workload
    try:
        if config.get("trainingProfile") == "optimized-fp16":
            module = importlib.import_module("glmtrain.fp16_benchmark")
            workload_type = module.FP16TrainingWorkload
        else:
            module = importlib.import_module("glmtrain.benchmark")
            workload_type = module.TrainingWorkload
    except ImportError as error:
        raise RuntimeError(
            "Model benchmark requires the training package. Install both packages: "
            "python -m pip install -e ./training -e ./hivemind-tools"
        ) from error
    workload = workload_type(config, device, seed)
    if workload.parameter_count != config["payloadElements"]:
        raise RuntimeError("Model parameter count does not match the benchmark contract")
    return workload
