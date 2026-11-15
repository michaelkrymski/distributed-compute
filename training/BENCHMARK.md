# Two-computer model-training benchmark

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

Use this branch on **both computers**: `codex/contributor-app`.
Update to the same commit before starting. This branch includes the GPU and
checkpoint-recovery tests plus the dashboard agents. No pull request is required.

The benchmark trains a disposable copy of the **24,388,992-parameter model** using
real forward/backward passes, gradient accumulation, FP16-compressed Hivemind
gradient averaging and FP32 AdamW updates. Generated next-token batches avoid
requiring a shared corpus/tokenizer. This measures training throughput, not
learning quality; evaluation, checkpoint saving and corpus I/O are excluded.

## 1. Install on both machines

Use Python 3.11/3.12 on Linux or WSL2. The AMD 7800 XT starting point is native
Linux with a supported ROCm driver/OS combination. HIP SDK availability on Windows
does not establish support for this Hivemind setup. Follow the
[AMD compatibility matrix](https://rocm.docs.amd.com/en/docs-6.4.3/compatibility/compatibility-matrix.html)
for the 7800 XT and the [PyTorch 2.8 builds](https://pytorch.org/get-started/previous-versions/#v280).

```bash
git clone --branch main https://github.com/michaelkrymski/distributed-compute.git
cd distributed-compute
# For an existing checkout: git switch codex/contributor-app && git pull --ff-only
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On the **NVIDIA 5070 Ti computer**, with a CUDA-12.8-compatible driver:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

On the **AMD 7800 XT computer**, with the compatible ROCm driver already installed:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
```

Then on **both**:

```bash
python -m pip install -e './training[test]' -e './hivemind-tools[test]'
```

Confirm the hardware before connecting (these commands need no team key):

```bash
# NVIDIA
python -m glmbench doctor --device cuda
cd training && python tests/smoke_gpu.py --backend cuda && cd ..
# AMD: use these commands instead
# python -m glmbench doctor --device rocm
# cd training && python tests/smoke_gpu.py --backend rocm && cd ..
```

The device smoke test and model smoke test must pass on each actual card.
CPU CI does not validate these two GPUs or their cross-vendor driver setup.

## 2. Keep Tailscale reachable from the execution environment

Both machines must be on the same tailnet. In WSL, run the Linux Tailscale client
inside the environment running Python, or supply equivalent routing. The IP bound
below must be an IP of that Linux environment, not an unreachable Windows-only IP.
Check `tailscale status` and `tailscale ping OTHER_COMPUTER` in both directions.
The agent does not install a VPN, open router ports, or change firewall rules.

If small pings work but SSH handshakes or gradient transfers stall, test larger
transfers before increasing Hivemind timeouts. On our WSL-to-7800-XT USB setup,
1280-byte TCP packets were repeatedly lost; a per-peer IPv4 route MTU of 1200
restored SSH and a full 24,388,992-element gradient collective. This is a measured
network workaround, not a general AMD requirement. After confirming the same
symptom, install it on the affected Linux/WSL sender with:

```bash
sudo sh hivemind-tools/scripts/install-peer-mtu-workaround.sh PEER_TAILSCALE_IPV4 1200
```

It changes only that existing Tailscale peer route and uses a systemd timer to
restore it after reconnects/reboots. It does not change the default route or
firewall. The installer prints commands to disable it and restore the route.

## 3. Organizer starts the rendezvous

On the organizer's machine, in the repo root with the venv active:

```bash
python -m glmbench coordinator \
  --api https://your-team.example \
  --host "$(tailscale ip -4)" --port 40100
```

Enter **your coordinator key** at the hidden prompt. Keep this terminal open.
The coordinator consumes little compute and does not count as a training node.
If an old lobby exists, cancel it on the website before starting a fresh one.

## 4. Configure the dashboard before starting agents

Open [the dashboard](https://your-team.example/), sign in as admin,
and choose **Benchmark → Benchmark settings**:

- Workload: **24.39M model · real training · FP32**.
- Warmup: start with **60 seconds** (editable from 0 to 600).
- Measurement: start with **300 seconds** (editable from 10 to 3,600).
- Click **Save settings**.

The dashboard also freezes a **Training profile** and **Sequences per microbatch**
before agents join:

- **Reference FP32**, microbatch 1: the original comparison workload.
- **Optimized FP32**, microbatch 4: larger matrix operations, fused GPU AdamW,
  fewer CPU synchronizations, stable contiguous gradient buffers, and no CPU
  gradient round trip on a single GPU. Weights, optimizer state, and computation
  remain FP32.
- **Optimized BF16**, microbatch 32: BF16 autocast for supported GPU operations,
  with FP32 model weights, gradient accumulation, loss reduction, and AdamW state.
  This changes numerical precision; validate real-corpus loss and evaluation
  against FP32 before choosing it for a training run. It never silently falls
  back to another precision on unsupported devices.
- **Compiled BF16**, microbatch 16: optional `torch.compile` CUDA graphs for a
  **single NVIDIA computer**. Compilation completes before the agent reports
  Ready, may take several minutes on its first run, and is reported separately
  from warmup/measurement. Compilation does not update weights or advance the
  benchmark data generator. The backend rejects AMD or a second node for this
  profile; use Optimized BF16 for the two-computer test.

Optimized multi-node profiles also use 2 MiB transfer chunks instead of the
512 KiB reference chunks. Full-model collective tests on these two machines
reduced exchange time by about 9%, without changing compression or gradient
values. Shorter matchmaking waits did not improve performance and were rejected.

The effective global batch defaults to 32,768 tokens. Select **Global tokens per
exchange** before agents join: 32,768, 65,536, 98,304, 131,072 or 262,144 tokens.
98,304 exchanges gradients one third as often per token as the default. This is
gradient accumulation with one AdamW update per exchange, not multiple local
optimizer updates/DiLoCo. The microbatch controls activation memory independently.
Larger batches change the optimization schedule; validate corpus quality and
learning-rate/warmup choices rather than assuming identical learning. Gradient
accumulation weights a short final microbatch correctly for any 1–10-node team.
All agents must use the same saved profile and microbatch, and changing either
requires a new lobby. Update both packages on every node before using optimized
profiles. If memory is insufficient, recreate the lobby with a smaller microbatch.
The 7800 XT short sweep used approximately 8.1 GiB of allocated tensor memory at
BF16/microbatch 32, excluding driver/display allocations.

Compilation remains disabled on AMD: `torch.compile` produced a GPU memory-access
fault on the current Ubuntu USB/ROCm stack, while the uncompiled optimized
profiles passed. The NVIDIA profile uses stable gradient buffers and explicit
CUDA-graph step boundaries during accumulation. Full-model BF16 gradient/loss
agreement is checked against eager execution; this is not a corpus-quality test.
No driver, kernel, power limit, or model architecture change is required.
The corpus trainer separately already accepts `--precision bf16 --microbatch 32`;
its data pipeline, Hivemind optimizer, validation and checkpoints still need their
own throughput/quality validation rather than assuming benchmark parity.

Choose the workload before agents connect; changing it requires removing the
existing lobby agents. Timing can be edited while agents wait, but freezes at
Start. The 15-second shared countdown is additional to the configured times.
Upgrade the dashboard/backend and both agents together (protocol 3).
Agents use the most precise backend clock sample from the last 30 seconds,
with a conservative aging penalty. One delayed HTTP response cannot replace a
recent precise sample. Genuine server clock jumps remain fatal to a valid report,
and samples expire so persistent latency still suppresses unreliable estimates.

Before joining, each agent/coordinator samples the HTTPS backend clock five
times. The selected server epoch plus local monotonic elapsed time becomes its
application clock. Hivemind DHT expiration and matchmaking use this same clock
in the parent and Linux fork workers, independent of OS wall time. Active-run
deadlines are fixed rather than moved by later HTTP responses. Heartbeats still
check disagreement with the backend and invalidate results above one second;
sleep/resume, real monotonic drift or backend clock jumps are not hidden.
This adapter requires Linux's fork start method and fails closed otherwise.
No system time service, administrator access or manual NTP setup is required by
the benchmark. Update every node to use the application clock before testing.

## 5. Both computers contribute compute

The organizer opens a **second terminal**, activates the same venv and runs the
NVIDIA agent. The AMD owner runs their agent in their own terminal:

```bash
# NVIDIA 5070 Ti (including the organizer's own computer)
python -m glmbench agent \
  --api https://your-team.example \
  --host "$(tailscale ip -4)" --device cuda --ready

# AMD 7800 XT: run this on that computer instead
python -m glmbench agent \
  --api https://your-team.example \
  --host "$(tailscale ip -4)" --device rocm --ready
```

Each agent asks for its member's node key, not the admin password. The organizer
can use their own node key while signed into the browser as admin. Multiple
computers may use the same member key and receive separate instance identities.
Use `--device-index N` if selecting among multiple visible GPUs; agent ports are
allocated dynamically. Strict `cuda`/`rocm` selection fails on the wrong wheel or
missing device instead of silently falling back to CPU.

Verify that the lobby shows exactly the two intended computers, correct GPU
names, both **Ready**, and the rendezvous online. Click **Start benchmark**.
Matching model/code/initial-weight fingerprints are required. Every node holds a
full model replica. In the reference profile, each uses microbatch 1, context
512, and 32 accumulation passes per gradient exchange with two nodes (16,384
tokens each; 32,768 globally). Optimized profiles group these same sequences
into the selected microbatches.

## 6. Read the results

The dashboard shows local and conservative group tokens/second only for measured
successful optimizer steps; warmup/boundary work is excluded. It estimates time
for 500M tokens at that group rate, and separately shows a 20% contingency.
This is a projection at the measured configuration and connectivity, not a
guarantee for another model, context, precision or corpus pipeline. A slower node
or network bottleneck can make two computers slower than one.

Download results before opening a new lobby. Results are also saved under the
ignored `hivemind-tools/results/` directory when launched there, or `results/` in
the current directory. No model checkpoint or real dataset is published. Clock
jumps, incomplete groups, failures or short observed windows make a report
incomplete and suppress the group training estimate. Never commit keys, results,
datasets or checkpoints.
