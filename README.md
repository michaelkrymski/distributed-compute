# Distributed Compute

A Windows/Linux desktop client and coordination platform for teams contributing GPUs to shared PyTorch training and benchmarks. Supports NVIDIA CUDA, AMD ROCm on compatible Linux systems, and CPU smoke tests. Tailscale provides peer connectivity; Hivemind handles discovery and gradient averaging.

## Start here

- [Desktop app and installation](desktop/README.md)
- [Guided machine setup](setup/QUICKSTART.md)
- [Administrator/backend setup](desktop/PROJECT-SETUP.md)
- [Real model benchmark](training/BENCHMARK.md)
- [Training and reference model](training/README.md)
- [Coordination tools](hivemind-tools/README.md)
- [Migration and service configuration](MIGRATION.md)

```sh
git clone https://github.com/michaelkrymski/distributed-compute.git
cd distributed-compute
python -m pip install -e ./desktop
glm-desktop
```

This repository is private: cloning requires your GitHub account to have access. GPU runtime installation is guided separately from GUI installation. Select your team's server in the app and use your reusable member key or administrator account. No keys, datasets or private results are included.

This source move does not relocate the existing database or running service. The new checkout has no default production server; add your existing team's address manually, or configure a new deployment as documented in [MIGRATION.md](MIGRATION.md).

## Licensing and scope

This is a private development repository, not an open-source release. [Source provenance](PROVENANCE.md) records original team source and rights that still need agreement before a public release. [Third-party notices](THIRD_PARTY_NOTICES.md) cover Hivemind, DiLoCo implementations, Slint and distribution.

The included model is a reference workload. The optimizer uses periodic Hivemind gradient averaging; it is not a complete DiLoCo implementation. Python module names remain compatible with existing run bundles.
