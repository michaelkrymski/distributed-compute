# Kaggle one-cell training test

1. Open a [new Kaggle notebook](https://www.kaggle.com/code/new).
2. In **Settings**, select an available **GPU** accelerator. Complete any Kaggle verification needed to enable it. Internet can stay off.
3. Copy [the entire smoke test](https://raw.githubusercontent.com/michaelkrymski/distributed-compute/main/setup/kaggle_smoke.py) into one Python cell and run it.

The cell bundles the model, config and smoke test from a pinned Git commit and uses Kaggle's preinstalled PyTorch. No network access, package install, dataset, team key or VPN setup is needed. On a successful run, the baseline 24.39M-parameter model performs three FP32 AdamW updates on generated tokens, restores a checkpoint, and performs one more update. The report includes GPU model, loss, peak memory and elapsed time. Download `/kaggle/working/glm-kaggle-smoke.json` from Kaggle's Files panel to share it with the group.

The test targets under five minutes once the notebook starts executing, with a four-minute execution budget. GPU allocation or account verification can add time before the cell runs. On a T4 x2 session this test uses the first GPU. The smoke test validates local training and checkpoint resume on that GPU. Measuring throughput, training on your corpus, and testing Hivemind peer connectivity require separate runs.

## Compare two Kaggle accounts

1. Invite your teammate to the same [Compute team](https://michaelkrymski.me/compute/).
2. Each person selects the same decoder project and creates their own key in **Worker keys**.
3. Both notebooks select **GPU T4 x2**, enable **Internet**, and paste the latest script above. Set `MODE = "benchmark"` at the top of each copy. Keep `MEASURE_SECONDS` the same.
4. Run the cell on each account and enter your own `dc_worker_` key in the hidden prompt. Keep keys out of notebook source and outputs.
5. Compare the final **tokens/s** values or exchange `/kaggle/working/glm-kaggle-benchmark.json`.

The notebook authenticates with the shared Compute service and reads the selected project's decoder configuration. It uses one T4 per notebook, warms up for 5 seconds, then measures real FP32 AdamW updates on generated tokens. It supports decoder projects up to 25M parameters and 512 context. The report records a configuration hash and code revision so you can confirm both accounts ran the same workload.

These are independent measurements. Their sum describes the combined throughput of two independent jobs. The service does not collect these reports as shared benchmark results. To measure two GPUs updating one model using the existing Hivemind benchmark, both workers must reach each other and a coordinator over the trusted peer network. Kaggle peer connectivity has not been verified.

### Reuse the original GatorLLM Team 2 setup

In both Kaggle notebooks, set `MODE = "benchmark"` and `BACKEND = "team2"`, enable Internet, and enter your own existing Team 2 access key at the prompt, including its original prefix (such as `gator_` or `glm_`). The [Team 2 dashboard](https://gatorllm-team-2-lab.netlify.app) validates the full key. The organizer password is for dashboard sign-in. The notebook uses the bundled 24.39M Team 2 reference decoder and saves its throughput report locally. You can run it without creating a room or a new Compute account.

## Test Tailscale between Kaggle notebooks

1. Each person opens a private Kaggle notebook with **Internet on**. Leave the accelerator on **None** for this network test. Both users need access to the same trusted tailnet.
2. Paste the [Tailscale setup cell](https://raw.githubusercontent.com/michaelkrymski/distributed-compute/main/setup/kaggle_tailscale.py) into both notebooks and run it. Open each notebook's own sign-in link and select the same tailnet. No Compute key or Tailscale auth key goes into the cell.
3. After both notebooks print `Ready`, exchange the two Tailscale IPs. On each side, run `test_kaggle_peer("OTHER_NOTEBOOK_IP")` in a new cell. Expect `TCP PASS` on both sides.
4. When finished, run `stop_kaggle_tailnet()` in both notebooks or stop both sessions.

The cell runs the official Tailscale binaries in userspace with an ephemeral node and a tailnet-only TCP listener on port 40190. It downloads a checksum-pinned archive and does not need root, a TUN device, or an accelerator. Both sides must be online, and the tailnet policy must allow TCP 40190 between them. The probe verifies bidirectional TCP using the userspace proxy. The existing Hivemind worker still needs a compatible transport before it can use this path for gradient exchange.
