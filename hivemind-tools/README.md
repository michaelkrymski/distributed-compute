# Hivemind team coordination and benchmark

> Personal repository migration: see [MIGRATION.md](../MIGRATION.md). Hosting URLs below are examples; configure your own server or enter your existing team server manually. No backend was moved by this source extraction.

This folder is **model-independent**. It contains a rendezvous, a locally launched agent, a synchronized benchmark, and a small optimizer adapter for a separately supplied training loop. There is no language-model architecture, tokenizer, corpus, or training checkpoint here. Those are preserved on the `michaelkrymski/theoretical-model-implementation` branch.

## What the benchmark measures

After everyone connects and explicitly marks ready, the coordinator presses **Start 6-minute benchmark** on the website. The backend schedules a shared 15-second countdown, then **60 seconds warmup + 300 seconds measurement**. Warmup rounds are excluded from measured results. The fixed payload is a 25-million-element FP32 tensor (100 MB uncompressed) with Float16 communication compression. Each agent also times ten 1024×1024 matrix multiplications per round and reports their aggregate GFLOP/s.

This is a synthetic compute-and-communication benchmark, **not LLM training**. Round latency includes Hivemind matchmaking and averaging. GEMM GFLOP/s describes that specific standalone workload, not total GPU peak throughput, training speed, or tokens/second. No 500M-token completion-time prediction is inferred from these numbers. With one participant the test measures compute only: there is no tensor averaging or peer-network measurement.

A measured round counts only if it starts and finishes inside the five-minute window and includes exactly all frozen participant identities, with no duplicate identities. Partial groups and failed rounds are excluded and recorded. Rounds that cross the measurement boundary are excluded. Each agent uses the backend clock with a round-trip correction; reports with estimated clock uncertainty over one second are marked incomplete. Cancellation, failed groups, or no valid rounds also produce an incomplete/degraded report.

## Install

Use Python 3.11/3.12 on Linux or WSL2. Choose a PyTorch 2.8.0 build appropriate for your device, then install the package:

```bash
cd hivemind-tools
python3 -m venv .venv
source .venv/bin/activate
# NVIDIA example, if your driver supports CUDA 12.8:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# CPU alternative:
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

Native Windows and Apple MPS are not implemented. The integration test uses Linux CPU processes; CUDA and actual cross-home networking still need the team's own benchmark.

## Private connectivity and team keys

Use a trusted LAN or private mesh VPN. Every agent and the rendezvous must be reachable by the other participants, not just by the website. Run the VPN in the Linux environment that runs Hivemind, or configure equivalent routing into WSL. Bind `--host` to that environment's actual private IP. The starter disables automatic router port mapping and relay discovery. It does not install a VPN or modify firewalls.

The organizer distributes each person their own team access key privately. Keys are **not in Git**, command-line arguments, URLs, or the static website bundle. The CLI asks for the key through a hidden prompt; alternatively set `GLM_TEAM_KEY` locally without committing it. The website keeps a manually entered key in sessionStorage for that tab session. Use **Disconnect** to clear it. The API stores only SHA-256 key hashes and checks a key on every read/write. GitHub membership does not automatically authenticate an agent. Private-network membership protects Hivemind peer traffic; the HTTP key protects the coordination API.

The team has five named accounts: `onlyspxctre`, `michaelkrymski`, `jsaluja09`, `jacklowry08`, and `carlimarquilage-ctrl`. The `michaelkrymski` key is the coordinator; the others can ready their own agents and read results. The organizer can sign in to the website as admin **and** run an agent on the same computer. The browser controls the benchmark; only the agent contributes compute. The lightweight coordinator/DHT process is rendezvous infrastructure and does not train or count as a benchmark computer.

A lobby supports **1–10 computer processes**, independent of the number of named accounts. Multiple computers can use one member's key; each receives a distinct random instance ID and reports its hostname and hardware. Up to ten registrations are retained in the lobby; remove stopped/stale entries in the website to free slots. Every online computer must be marked ready. Start atomically freezes the online participant list; offline computers are excluded and late joins are rejected. A stopped computer must not restart into a running test; wait for the next lobby.

### Admin password

On the dashboard, Michael uses his coordinator key **once to activate** an admin account, choosing and confirming a password of at least 16 characters. After activation, sign in with the password. The key still serves as a CLI/node credential; activation does not revoke it. Member keys remain available for voting and agent enrollment; there are no member passwords to manage. Keep every key private.

The password is salted and hashed using PBKDF2-SHA256 (100,000 iterations, the workerd limit). Browser sessions expire after 12 hours, use an HttpOnly/Secure/SameSite=Strict cookie in production, and are stored as token hashes. Sign out revokes the current session. Cookie-authenticated writes require the dashboard's Origin. Login checks are atomically limited to five attempts per 15-minute window. There is no email-based reset: if locked out, the site owner must reset the admin account and revoke sessions through the hosting database; never share a database export or put credentials in Git.

## Start the rendezvous

On a machine that will stay online, using the coordinator key:

```bash
python -m glmbench coordinator \
  --api https://your-backend.example \
  --host YOUR_PRIVATE_IP --port 40100
```

It starts a real Hivemind DHT, creates the shared lobby, publishes its peer address to the protected API, and sends heartbeats. Keep the process running. Its private identity key is under ignored `output/coordinator.key`; preserve it locally but do not share it. A cancelled or finished session can be replaced by restarting the coordinator. An active lobby must first be cancelled from the website before a replacement coordinator creates a new one.

## Join as an agent

Every teammate, including the coordinator, starts this locally:

```bash
python -m glmbench agent \
  --api https://your-backend.example \
  --host YOUR_PRIVATE_IP --port 40101
```

Use **Mark ready** on the website after connecting, or append `--ready` to opt in from the CLI. The page cannot launch a process on another person's machine. By default agents use a GPU when available in the installed PyTorch CUDA/ROCm build, otherwise CPU. Select `--device cpu`, `--device cuda`, or `--device rocm` explicitly if needed. `--device-index 0` selects a GPU; start a second process with index 1 for a second visible GPU. The default agent port is dynamically allocated, so multiple processes can coexist on one host. GPU GEMM uses BF16 when supported, otherwise FP32. No optimizer updates or actual model training happen during this test.

The website shows the participant's device and connection/readiness status. Multiple live computers are allowed per member. A disconnected agent expires from the readiness gate after 20 seconds. The coordinator and 1–10 computers must be online, with every online computer ready before Start is accepted. The server enforces this independently of the button state, including concurrent Start requests.

The measurement runs until the shared deadline. The browser may close after Start without stopping the agents. Ctrl+C stops a local process; missing/incomplete reports remain visible rather than being fabricated. Local results are saved under ignored `results/<session-id>/<member>.json` before upload. If upload fails, retry with:

```bash
python -m glmbench upload --api https://your-backend.example \
  --file results/SESSION_ID/YOUR_MEMBER.json
```

Reports are bound to that member and agent instance; an old report cannot overwrite a newly registered agent. The API retains session records and reports in D1. The current dashboard shows the active/latest session; download the JSON results before opening a new lobby if you want an easy local copy.

## AMD Radeon setup and device check

The agent supports AMD **ROCm PyTorch**, using PyTorch's `torch.cuda` API under HIP, and labels AMD devices as `rocm:N` in uploaded metadata. This is a software path, not a claim that every Radeon card/OS combination has been validated.

- **RX 7800 XT:** AMD lists it in the ROCm 6.4.3 Linux matrix. Use an OS/driver/ROCm/PyTorch combination supported by AMD. Native Linux is the documented starting point; HIP SDK availability on Windows is not proof that this Linux/WSL Hivemind setup will work.
- **RX 6700 XT:** absent from that supported matrix. Treat GPU execution as experimental if your installed ROCm build supports it; CPU mode is available. The launcher does not force architecture overrides or hide failed GPU initialization.

For a compatible Linux ROCm 6.4 installation, the pinned PyTorch 2.8 wheel is:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
python -m pip install -e '.[test]'
python -m glmbench doctor --device rocm --device-index 0
```

`doctor` performs a real matrix multiplication and backward pass, checks finite outputs/gradients, and prints the detected card, ROCm version, and dtype. It requires no team key and does not join a lobby. Run it before starting the six-minute test. The benchmark itself remains synthetic GEMM plus CPU-tensor Hivemind averaging, not a model training run. CUDA/ROCm use BF16 only when reported as supported without emulation; otherwise they use FP32.

Then join the existing Tailscale network and run the normal `agent` command with `--device rocm --ready`. No VPN installation, routing, or firewall configuration is changed by these tools.

References: [AMD ROCm 6.4.3 Linux requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.3/reference/system-requirements.html), [PyTorch 2.8 installation builds](https://pytorch.org/get-started/previous-versions/#v280), [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).

## Model-independent training adapter

`glmbench.runtime.collaborative_optimizer(...)` wraps externally supplied PyTorch parameters and an optimizer factory. It does not construct a model. It uses Hivemind gradient accumulation/averaging, not a DiLoCo outer optimizer. A future actual training workload can use it independently of this synthetic benchmark.

## Verification

```bash
python -m pytest -q
```

These are explicitly **unit tests** for timing/accounting and URL handling. They are not a backend acceptance test.

The real backend acceptance test is `team-dashboard/tests/api-integration.mjs`: it sends actual HTTP requests to the actual local API and local Cloudflare D1 database, including invalid credentials, coordinator permissions, readiness, duplicate Start, and voting behavior. A test registration fixture checks the API contract; it does not pretend to be a connected Hivemind agent.

The full end-to-end test launches a real rendezvous and **five separate CPU Hivemind processes** against the same actual API/D1, with the **full 60-second warmup and 300-second measurement**, then checks stored result uploads. It has no fake HTTP server and no mocked DHT:

```bash
# First run a locally migrated dashboard backend as documented in its README.
# Generate LOCAL test keys with the provided script, never production keys in CI.
python tests/e2e_real_backend.py --agents 5 --api http://127.0.0.1:5173 \
  --keys-file ../team-dashboard/work/team-access-keys.json
```

This test refuses remote API origins to avoid creating test sessions in production. It writes test output only under ignored `results/`, cleans up its child processes, and never publishes those files to Git. Passing it proves the local software path, not participation by five people, five GPUs, or five different home networks.

CI runs the full-duration end-to-end test with both `--agents 1` and `--agents 5`. The real HTTP/D1 contract tests cover ten registrations, capacity enforcement, shared-account machine identities, frozen membership, admin password sessions, CSRF rejection, and concurrent login throttling. AMD execution and multi-home Tailscale connectivity require hardware/network testing by the team. Protocol 3 requires updated agents; restart any older lobby before upgrading.


## Configurable timing and real model workloads

The admin can save 0–600 warmup seconds and 10–3600 measurement seconds in the
Benchmark settings card before Start. The default remains 60 + 300 seconds;
there is an additional 15-second countdown. Agents pick up the final saved times
when the participant group freezes. Neither timing nor workload can change after
Start. Choose the workload before connecting agents; changing workload or batch
size requires removing all registered agents.

`synthetic` remains available on main without any model dependency. For actual
model throughput, check out `michaelkrymski/theoretical-model-implementation` on
both machines and install **both** packages (`pip install -e ./training -e
./hivemind-tools`). Select **24.39M model · real training · FP32** before launching
agents. See that branch's [two-computer guide](../training/BENCHMARK.md).

The supplied workload uses generated next-token batches, real forward/backward
passes, accumulation toward 32768 global tokens, CPU gradient staging, Hivemind
averaging, and FP32 AdamW. It checks matching model/initial-weight fingerprints.
Failed gradient groups stop model updates to prevent continuing with divergent
epochs. Reports include only measured successful steps and actual local token
counts. The dashboard estimates 500M-token time only after all group reports are
complete, using the smallest common round count and full measured wall duration.
Evaluation/checkpoint/corpus-I/O costs are excluded; a separate 20% contingency is
shown. This does not infer LLM tokens/sec from GEMM.

The end-to-end test accepts `--warmup`, `--measure`, `--workload`, and
`--batch-tokens`; defaults preserve the full 60+300 synthetic acceptance test.
The theoretical branch also tests two real full-baseline CPU training workers
with a smaller 1024-token global batch, ten warmup seconds and 45 measured
seconds. That is software validation, not GPU or production-batch performance.
