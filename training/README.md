# Shared Hivemind training

> Personal repository migration: see [MIGRATION.md](../MIGRATION.md). Hosting URLs below are examples; configure your own server or enter your existing team server manually. No backend was moved by this source extraction.

## Automated computer setup — start here

Run from a complete `codex/contributor-app` source checkout in a real terminal.

Windows / NVIDIA (PowerShell):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup\windows.ps1
```

Native Ubuntu 24.04 / AMD:

```bash
bash setup/linux.sh --device rocm
```

Native Ubuntu 24.04 / NVIDIA: use `--device cuda`. CPU-only: use `--device cpu`, or `-Device cpu` on Windows. Add `--plan` / `-Plan` to preview without changing anything. Only x86_64 is automated. Windows targets WSL2 `Ubuntu-24.04`; AMD uses native Ubuntu in this project.

No source checkout yet? Install Git, then run (GitHub may request your sign-in):

```text
git clone --branch main https://github.com/michaelkrymski/distributed-compute.git
cd distributed-compute
```

The Windows script continues into the Linux script. Six stages stream real installation output: prerequisites, driver checks, Tailscale, a separate Python 3.12 environment, the project's pinned PyTorch/training packages, and a small local GPU check. The environment is `~/.local/share/glm/training-venv`. Existing training environments are left alone; reruns reuse/repair this dedicated environment.

**USER TAKEOVER means you control the terminal/browser.** Create your Ubuntu username/password, enter sudo passwords, follow the Tailscale sign-in link/QR, and complete MFA/device approval yourself. Join the organizer's **same tailnet**, not a new personal one. Ask for a tailnet invitation if needed. The app's team key is separate and is not requested by setup. Setup does not save passwords or request a Tailscale auth key.

After any requested restart or full Ubuntu sign-out, rerun the **same command**. Linux exit code 20 means a manual action is required. WSL installation may need Administrator PowerShell; subsequent steps use your normal account. The execution-policy override above applies only to that PowerShell process. Setup never restarts the machine, changes firewall rules, purges drivers, or applies the experimental MTU workaround.

Driver handling:

- Windows NVIDIA: check the Windows driver; if missing, open NVIDIA's official installer page for you. Never install a Linux NVIDIA driver inside WSL. Existing drivers are checked, not automatically replaced.
- Native Ubuntu NVIDIA: if no working driver is detected, request Ubuntu's recommended driver install, then stop for reboot/Secure Boot enrollment.
- Native Ubuntu AMD: use the existing `amdgpu`/KFD driver, install the ROCm PyTorch wheel, and add missing `render`/`video` membership. If `/dev/kfd` is missing, stop for the official OS/GPU-specific instructions rather than guess a DKMS package. Group changes require signing out/in.
- If WSL does not run systemd, stop for Microsoft's systemd setup instructions; the script does not shut down existing WSL workloads.

On success, copy the printed Python path, Tailscale IPv4 and distro name into the desktop app, then click **Check GPU & Tailscale**. Setup does not join a room or start training.

Official references: [WSL commands](https://learn.microsoft.com/windows/wsl/reference), [NVIDIA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/), [Tailscale Linux installation](https://tailscale.com/docs/install/linux), [PyTorch 2.8 wheels](https://pytorch.org/get-started/previous-versions/#v280), [AMD support matrix](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html).

---

A basic working process for the team's **25M-parameter / 500M-token experiment**. One organizer prepares the run bundle. A small CPU rendezvous process keeps discovery online. Each teammate runs one full model replica and contributes gradients from an assigned data shard.

This uses `hivemind.Optimizer` with gradient accumulation and periodic gradient averaging. **It is not DiLoCo**: there is no DiLoCo outer optimizer or hundreds of independent AdamW updates between synchronizations. Start here to establish a correct distributed baseline before experimenting with local-update algorithms.

## 1. Install on each machine

Use **Linux, or Ubuntu inside WSL2 on Windows**, with Python **3.11 or 3.12**. NVIDIA CUDA training and CPU testing are implemented; AMD ROCm can be checked with the local GPU smoke test below. Apple MPS is not implemented. Distributed GPU training still requires hardware validation. A GPU is optional for the rendezvous node. CPU workers can join for tests but may contribute slowly.

From the repository's `training/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Pick ONE PyTorch build before installing the package:
# NVIDIA: choose the CUDA 2.8.0 wheel appropriate for your driver from
# https://pytorch.org/get-started/previous-versions/#v280
# For example, if CUDA 12.8 is supported by your NVIDIA driver:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# CPU-only alternative:
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e '.[test]'
python -m glmtrain --help
```

The package pins the directly used library versions; install the same versions on every worker. The CPU workflow was tested on Ubuntu/WSL with Python 3.12 and Torch 2.8.0+cpu. Use the opt-in GPU smoke test below to verify model execution on your own device; CPU CI does not establish GPU compatibility.

If this change is still in a pull request, first fetch and check out `michaelkrymski/theoretical-model-implementation`. After merge, use `main`.

## 2. Prepare one shared run bundle (organizer only)

First choose and clean your corpus. Put UTF-8 training documents in `data/train.txt` and held-out documents in `data/validation.txt`, **one document per line**. Split documents before tokenizer training and remove duplicates/overlap yourself. This starter rejects identical train/validation files but does not perform corpus-wide deduplication or license checking.

```bash
python -m glmtrain prepare \
  --train data/train.txt \
  --validation data/validation.txt \
  --config configs/baseline.json \
  --output runs/baseline-001

python -m glmtrain inspect --bundle runs/baseline-001
```

The bundle contains `run.json`, `tokenizer.json`, `train.bin`, and `validation.bin`. Share this **same complete folder** with the team through an appropriate file-sharing channel; do not train a different tokenizer per machine. Large corpus binaries and checkpoints are ignored by Git. Both token files use little-endian uint16 token IDs and explicit end-of-document tokens.

Preparation trains byte-level BPE on training text only. The requested vocabulary is 8,192; a tiny corpus may produce fewer tokens, in which case the recorded model vocabulary and parameter count reflect the actual tokenizer. The full baseline with 8,192 tokens has **24,388,992 trainable parameters**: 12 layers, width 384, 6 heads, 4× GELU feed-forward, tied embeddings, RMSNorm, and RoPE, with no linear biases.

`run.json` records the architecture, optimizer settings, corpus/tokenizer checksums, a unique run ID, and a fingerprint. Workers verify all of these before connecting. Changing a file after preparation invalidates the bundle. To change shared settings, create a **new bundle and run ID** from a new configuration; do not silently edit an active run.

## 3. Put the training machines on a private network

Use the same trusted LAN, or a private mesh VPN such as [Tailscale](https://tailscale.com/docs/how-to/quickstart). Admit the team's machines to that network and verify they can reach each other. Install/run the VPN **inside the Linux environment doing the training**, or explicitly configure equivalent routing into WSL2. A VPN IP available only on the Windows host is not automatically an address Linux can bind to.

Every node uses `--host` to bind to its **own** LAN/VPN IP. The rendezvous listens on TCP 40100; the examples assign each worker TCP 40101. Those ports must be reachable by teammates on the private network, including worker-to-worker connections. Multiple processes on the same machine need different ports, or port 0 for automatic allocation in local tests.

Loopback is the default, so the commands do not expose a service until you explicitly choose another local interface. The implementation disables automatic router port mapping and libp2p relay discovery; the VPN handles connectivity across home networks.

**Group membership is enforced by your private network, not GitHub or the dashboard.** Run IDs and bundle fingerprints are compatibility checks, not passwords. Only trusted teammates should reach these processes because they exchange model/optimizer state and can affect the training result. The starter does not provide an open, untrusted volunteer network or defend against malicious gradients.

## 4. Start a rendezvous node (one organizer)

Replace `100.64.0.10` with this Linux machine's actual private-network IP:

```bash
python -m glmtrain bootstrap \
  --bundle runs/baseline-001 \
  --host 100.64.0.10 --port 40100
```

Keep it running. It prints a peer address shaped like:

```text
/ip4/100.64.0.10/tcp/40100/p2p/ACTUAL_PEER_ID
```

Copy the **actual complete printed address** to each teammate. It is also saved in `output/peers.json`. The private bootstrap key stays in `output/bootstrap.key`; do not distribute that key. Keeping it preserves the peer ID when restarting the rendezvous on the same host/port.

The rendezvous performs discovery only. It does not hold the authoritative model, execute training, or preserve a checkpoint if all workers leave. Keep at least one training worker online for mid-run joining.

## 5. Start the first training worker

Each teammate gets a different shard index. The baseline configuration reserves five shards; this suggested mapping follows the GitHub team snapshot:

| Shard | Teammate |
|---|---|
| 0 | onlyspxctre |
| 1 | michaelkrymski |
| 2 | jsaluja09 |
| 3 | jacklowry08 |
| 4 | carlimarquilage-ctrl |

Any teammate may start the first worker; the mapping does not define a leader. Example for the organizer assigned shard 1 (replace both addresses):

```bash
python -m glmtrain train \
  --bundle runs/baseline-001 \
  --peer /ip4/100.64.0.10/tcp/40100/p2p/ACTUAL_PEER_ID \
  --host 100.64.0.11 --port 40101 \
  --name michaelkrymski --shard 1 \
  --output output/michaelkrymski \
  --microbatch 4 --device cuda --start-new
```

Use `--start-new` **once, for an intentionally fresh run only**. Do not use it to recover a run that has lost all live workers. To require two workers before starting the first batches, append `--min-peers 2`; this is a startup barrier, not a permanent quorum.

## 6. Teammates join

On another machine, with the same bundle and their own interface/shard/output:

```bash
python -m glmtrain train \
  --bundle runs/baseline-001 \
  --peer /ip4/100.64.0.10/tcp/40100/p2p/ACTUAL_PEER_ID \
  --host 100.64.0.12 --port 40101 \
  --name jacklowry08 --shard 3 \
  --output output/jacklowry08 \
  --microbatch 2 --device cuda
```

The joining worker downloads current model/optimizer state from a live worker before training. A smaller GPU can use `--microbatch 1`; capable NVIDIA GPUs can optionally use `--precision bf16`. CPU testing uses `--device cpu --microbatch 1`. Model weights stay FP32; BF16 controls CUDA autocast. Optimizer states and the complete model still need to fit each worker's memory.

Batch sizes may differ, while the model, tokenizer, data bundle, AdamW settings, and target batch remain shared. Each worker's contribution is weighted by the number of processed tokens. Training samples are drawn with replacement from disjoint contiguous shards, and targets never cross shard boundaries. Different workers may not claim the same shard simultaneously. Duplicate shard detection is best effort using DHT heartbeats, not a transactional lease; coordinate assignments with the team. A crashed peer can take about 45 seconds to expire.

You do not need all five people online at once. Fewer online shards means less corpus coverage until the other workers participate. Do not renumber the shards during the run. More than five simultaneous workers requires preparing a new bundle with a larger `data_shards` setting.

## 7. Observe, stop, and resume

```bash
python -m glmtrain status \
  --bundle runs/baseline-001 \
  --host 100.64.0.11 \
  --peer /ip4/100.64.0.10/tcp/40100/p2p/ACTUAL_PEER_ID
```

Status lists currently visible workers, shards, Hivemind epochs, and local token counters. Records expire; this is not a durable global contribution ledger. Every worker writes local `metrics.jsonl`. Every 100 completed global epochs (configurable with `--checkpoint-every`) and at the nominal end, it evaluates four held-out windows and atomically saves `latest.pt`. This is a small validation sample, not a comprehensive evaluation.

Press Ctrl+C to leave. Resume with the same command, output directory, and shard, adding `--resume` and omitting `--start-new`. Recovery restores model weights, optimizer, global epoch, local token counter, and sampling RNG from the last checkpoint. Partial accumulation after that checkpoint is discarded. If other workers remain online, the resumed worker then catches up with them. If everyone stopped, recover one worker from a saved checkpoint first, then let others join it. Back up a completed checkpoint before changing the run.

`--max-microsteps N` bounds a test process's local batch count; it is not a global stopping rule. Ending mid-accumulation does not generate a resumable checkpoint. The rendezvous identity and files on an idle worker do not replace a model checkpoint.

The shared target is **500M processed tokens across the collaboration**, not 500M per person. The starter stops at `ceil(target_tokens / target_batch_tokens)` Hivemind epochs. Hivemind can overshoot a batch because peers contribute asynchronously, so the logged `nominal_global_tokens` is an **estimate**, not an exact 500M-token ledger. Local token counters count processed batches, including repeated samples and work that may be discarded during synchronization. A Hivemind epoch is a global synchronization interval, not a complete pass over the corpus.

The learning rate is a shared constant in this baseline. Gradient clipping is applied per local microbatch before averaging. Communication uses Float16 compression; model and optimizer updates remain FP32. Hivemind can continue with local gradients after failed communication, so watch its logs for repeated fallback messages. This is not a strict all-worker barrier algorithm.

## Tests

```bash
python -m pytest -q
python tests/smoke_swarm.py
python tests/smoke_checkpoint.py
python tests/smoke_late_join.py
```

The swarm test starts a real rendezvous and two CPU workers on loopback, requires an observed two-peer gradient-averaging event, checks completed epochs and compatible model checkpoints, then verifies checkpoint resume. The checkpoint smoke stops a single worker below target, resumes from its real checkpoint, and checks a subsequent model update, epoch and local-token progress, and replacement checkpoint. Unit tests also cover sampler restoration, incompatible shard/fingerprint rejection, and replacement failure preserving the old file. The late-join test checks that a new worker receives a trained peer's model, optimizer state and epoch, and then participates in averaging. These tests use tiny synthetic text and temporary files; the smoke tests use child processes and clean them up. They do not establish distributed bitwise reproducibility or GPU recovery.

### Local GPU smoke test (NVIDIA or AMD ROCm)

This is an **opt-in hardware test**, not part of the default CPU CI suite. From
`training/`, with the package installed in your active Python 3.11/3.12 environment:

```bash
# AMD: install the ROCm build instead of the NVIDIA/CPU wheel shown above.
# Match the wheel to a supported GPU/driver environment; no architecture overrides
# are applied by the test.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
python -m pip install -e '.[test]'
python tests/smoke_gpu.py --backend rocm

# NVIDIA, with the CUDA PyTorch build already installed:
# python tests/smoke_gpu.py --backend cuda
# Detect either installed GPU backend:
# python tests/smoke_gpu.py
```

PyTorch exposes AMD ROCm through its `torch.cuda` API too. The test detects the
installed backend, verifies GPU access, and **fails rather than falling back to
CPU**. `--backend` additionally checks that the installed build is the one you
intended. Use `--device-index N` to select a visible GPU.

It constructs the full baseline model from `configs/baseline.json` (24,388,992
parameters, context 512), runs three FP32 AdamW updates on synthetic next-token
batches, checks finite gradients/weights and changed parameters, then saves and
reloads model/optimizer state. It checks exact restored weights, optimizer state,
and logits before performing another update. A temporary checkpoint is removed
on exit. Output includes the card/backend versions, losses, elapsed time, and peak
allocated GPU memory. `--steps N` (at least two) and `--microbatch N` adjust the
workload; the default microbatch is one.

A PASS establishes single-GPU model/optimizer execution and checkpoint resume on
that machine. It does **not** establish Hivemind GPU synchronization, BF16 support,
real-data convergence, or long-run stability. The regular pytest suite checks
backend-selection error handling with mocks; those checks are not GPU validation.

Local validation passed on an RX 6800 XT (`gfx1030`, 16 GB), using Python 3.12,
PyTorch `2.8.0+rocm6.4`, and a host ROCm 7.2.4 installation, without architecture
overrides. The default FP32 test completed all four updates (including resume),
with about 1.1 GiB peak allocated GPU memory. This is evidence for that specific
configuration, not a compatibility guarantee for other Radeon cards or ROCm builds.

## Scope

This is a basic experimental trainer, not an unattended distributed service. It does not automatically set up a VPN, invite GitHub members, provision a server/GPU, distribute datasets, or launch on teammates' computers. No long-running training service is started by installing it. Training telemetry is written locally and is not automatically posted to the website's database.

Hivemind API reference: [Optimizer source and configuration guide](https://github.com/learning-at-home/hivemind/blob/1.1.12/hivemind/optim/optimizer.py), [DHT implementation](https://github.com/learning-at-home/hivemind/blob/1.1.12/hivemind/dht/dht.py).


For the dashboard-controlled two-computer NVIDIA/AMD training throughput run, follow [BENCHMARK.md](BENCHMARK.md). This uses real model gradients and AdamW on generated token batches; it is separate from corpus training/checkpoint recovery.
