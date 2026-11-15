# Using another model or training project

Each project has its own model contract, training token goal, worker keys,
benchmark lobby, training room and history. A new project does not inherit the
original team's model, keys or training data. Existing self-hosted teams remain
separate until their owners intentionally move workers to a new workspace.

## Included decoder driver

Create a project with its vocabulary, width, layers, heads, expansion, context,
weight tying and token goal. Parameter count is calculated on the server using
the same formula as `glmtrain.model.ModelConfig`. Download its training config,
prepare a tokenizer/corpus bundle using the existing training CLI, and share the
complete verified bundle with workers. Benchmarking generated token batches is
separate from corpus training and does not establish final training quality.

## Other architectures: explicit local adapters

Choose **My own local workload adapter** when creating the project. Record the
actual parameter count, context, token goal and small non-secret JSON settings.
On every worker, install your reviewed local Python code and explicitly select:

```sh
python -m glmbench agent --api YOUR_SHARED_SERVICE_ORIGIN \
  --host YOUR_TAILSCALE_IP --device auto --ready \
  --max-parameters 1000000000 --workload-factory your_package:make_workload
```

The service cannot choose this import, download a repository, or execute a shell
command. Without an explicit factory, an external workload is rejected locally.
The local parameter limit defaults to 25M; increase it only for a model your
computer can accommodate. A model-count ceiling of 2B is a contract bound, not
a claim that your device can train a 2B model.

The factory signature is `(config, device, seed) -> workload`. It receives the
project settings in `config['adapterConfig']`; return a workload implementing
the interface used by `glmtrain.benchmark.TrainingWorkload`:

- `parameter_count`, `cfg.context`, and a 64-hex-character `fingerprint` covering
  architecture/code/initial weights. All peers must match.
- `prepare(world_size, local_tokens=...)` returns a flat CPU FP32 gradient tensor.
- `apply(averaged_gradients)` performs the corresponding local update.
- Runtime fields: `profile`, `precision`, `microbatch`, `gradient_part_size`,
  `local_tokens`, `last_loss`, `last_compute_tps`, `steps`; optional `local_steps`,
  `compiled_model` and `compile_seconds` follow the included driver.

Make `prepare/apply` agree on token weighting and optimizer semantics. Copying
the interface does not make unrelated optimizers or parameter layouts compatible.
`glmtrain.adapters:decoder` is a runnable example using the included model with
its configuration under `adapterConfig.model`.

Corpus training with a different architecture needs a matching local trainer;
the included corpus trainer is the configurable decoder implementation. Reuse
`glmbench.runtime.collaborative_optimizer` or your own Hivemind integration and
the project's room protocol for registration, readiness and progress. The web
service coordinates and reports progress; it does not implement arbitrary model
architectures or turn a benchmark adapter into a corpus-training pipeline.

## Coordination protocol

Keys are project-scoped, so `/api/benchmark` and `/api/rooms` identify the right
project without putting credentials in URLs. The desktop app can use that key
and the shared service origin directly. Web team accounts manage roles, project
configuration and sessions; desktop workers use reusable keys saved locally.

The room protocol accepts a verified bundle fingerprint and reachable peer, then
freezes ready workers and shard assignments on Start. Workers report their own
loss/epoch/nominal token progress. No dataset or checkpoint upload is required.
