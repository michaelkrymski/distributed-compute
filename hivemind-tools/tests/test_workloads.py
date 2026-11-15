import pytest
from glmbench.workloads import training_workload


def test_server_cannot_choose_a_python_import_for_external_workload():
    config = {"workload": "baseline-training", "runtime": "external", "workloadFactory": "unreviewed.module:run"}
    with pytest.raises(RuntimeError, match="explicitly"):
        training_workload(config, None, 42)


def test_external_factory_must_be_an_explicit_import_reference():
    config = {"workload": "baseline-training", "runtime": "external"}
    with pytest.raises(RuntimeError, match="explicitly"):
        training_workload(config, None, 42, "https://unreviewed.invalid/remote.py")
