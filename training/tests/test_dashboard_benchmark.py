import torch
import pytest
from glmtrain.benchmark import TrainingWorkload, allocate_proportional_tokens


def config():
    return {"model": {"vocab_size": 300, "width": 32, "layers": 1, "heads": 4,
                      "expansion": 4, "context": 16, "tied": True}, "batchTokens": 64}


def test_real_training_gradients_and_optimizer_updates():
    torch.set_num_threads(1)
    a = TrainingWorkload(config(), torch.device("cpu"), 123)
    b = TrainingWorkload(config(), torch.device("cpu"), 456)
    assert a.fingerprint == b.fingerprint
    before = a.model.embedding.weight.detach().clone()
    ga, gb = a.prepare(2), b.prepare(2)
    assert ga.numel() == a.parameter_count
    assert a.local_tokens == b.local_tokens == 32
    assert not torch.equal(ga, gb)
    gradients = (ga + gb) / 2
    a.apply(gradients)
    b.apply(gradients)
    assert a.steps == b.steps == 1
    assert not torch.equal(a.model.embedding.weight, before)
    for pa, pb in zip(a.parameters, b.parameters):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    assert a.optimizer.state and a.last_loss > 0


def test_optimized_microbatches_preserve_gradients_with_a_short_tail():
    torch.set_num_threads(1)
    reference_config = {**config(), "batchTokens": 80}
    fast_config = {**reference_config, "trainingProfile": "optimized-fp32", "microbatch": 4}
    reference = TrainingWorkload(reference_config, torch.device("cpu"), 123)
    fast = TrainingWorkload(fast_config, torch.device("cpu"), 123)
    reference_gradients = reference.prepare(1)
    assert fast.prepare(1) is None  # Solo training never stages gradients on CPU.
    assert reference.local_tokens == fast.local_tokens == 80
    torch.testing.assert_close(fast.flat_grad, reference_gradients, rtol=3e-4, atol=2e-6)
    assert abs(fast.last_loss - reference.last_loss) < 1e-5
    reference.apply(reference_gradients)
    fast.apply(None)
    for a, b in zip(reference.parameters, fast.parameters):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-6)
    assert reference.fingerprint != fast.fingerprint
    assert fast.steps == 1


def test_optimized_collective_keeps_effective_tokens_and_updates_identical_replicas():
    settings = {**config(), "batchTokens": 80, "trainingProfile": "optimized-fp32", "microbatch": 4}
    a = TrainingWorkload(settings, torch.device("cpu"), 123)
    b = TrainingWorkload(settings, torch.device("cpu"), 456)
    ga, gb = a.prepare(3).clone(), b.prepare(3).clone()
    assert a.local_tokens == b.local_tokens == 32
    averaged = (ga + gb) / 2
    with pytest.raises(RuntimeError, match="peer averaging"):
        a.apply(None)
    a.apply(averaged)
    b.apply(averaged)
    for pa, pb in zip(a.parameters, b.parameters):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
    assert a.prepare(3).numel() == a.parameter_count
    for p in a.parameters:
        assert p.grad is not None


def test_training_profiles_validate_capabilities_and_settings():
    for extra in ({"trainingProfile": "unknown"}, {"microbatch": 3}, {"microbatch": 4}, {"localSteps": 3}, {"localSteps": 4}, {"trainingProfile": "optimized-bf16"}, {"trainingProfile": "compiled-bf16"}):
        with pytest.raises((ValueError, RuntimeError)):
            TrainingWorkload({**config(), **extra}, torch.device("cpu"), 123)


def test_allocate_proportional_tokens_conserves_batch_and_scales_by_speed():
    equal = allocate_proportional_tokens(98304, 512, ["fast", "slow"], speeds=None)
    assert equal == {"fast": 49152, "slow": 49152}

    prop = allocate_proportional_tokens(
        98304, 512, ["fast", "slow"], speeds={"fast": 97000.0, "slow": 54700.0}
    )
    assert sum(prop.values()) == 98304
    assert prop["fast"] > prop["slow"]
    assert prop["fast"] % 512 == 0 and prop["slow"] % 512 == 0

    solo = allocate_proportional_tokens(98304, 512, ["fast"], speeds={"fast": 97000.0})
    assert solo == {"fast": 98304}

    tiny = allocate_proportional_tokens(1024, 512, ["fast", "slow"], speeds={"fast": 999999.0, "slow": 1.0})
    assert tiny == {"fast": 512, "slow": 512}


def test_prepare_with_explicit_local_tokens_measures_speed():
    torch.set_num_threads(1)
    settings = {**config(), "batchTokens": 96, "trainingProfile": "optimized-fp32", "microbatch": 2}
    w = TrainingWorkload(settings, torch.device("cpu"), 123)
    grad = w.prepare(2, local_tokens=64)
    assert grad.numel() == w.parameter_count
    assert w.local_tokens == 64
    assert w.last_compute_seconds > 0 and w.last_compute_tps > 0
    with pytest.raises(ValueError, match="positive multiple"):
        w.prepare(2, local_tokens=20)


def test_local_steps_runs_multiple_updates_per_sync():
    torch.set_num_threads(1)
    single_cfg = {**config(), "batchTokens": 64, "trainingProfile": "optimized-fp32", "microbatch": 2, "localSteps": 1}
    multi_cfg = {**config(), "batchTokens": 64, "trainingProfile": "optimized-fp32", "microbatch": 2, "localSteps": 4}
    single = TrainingWorkload(single_cfg, torch.device("cpu"), 123)
    multi = TrainingWorkload(multi_cfg, torch.device("cpu"), 123)
    assert single.fingerprint != multi.fingerprint
    before = multi.model.embedding.weight.detach().clone()
    grad = multi.prepare(2, local_tokens=32)
    assert grad.numel() == multi.parameter_count
    assert multi.local_tokens == 32 * 4
    assert multi.steps == 3
    assert not torch.equal(multi.model.embedding.weight, before)
    multi.apply(grad)
    assert multi.steps == 4

    solo = TrainingWorkload(multi_cfg, torch.device("cpu"), 123)
    assert solo.prepare(1, local_tokens=32) is None
    assert solo.local_tokens == 32 * 4
    assert solo.steps == 3
    solo.apply(None)
    assert solo.steps == 4
