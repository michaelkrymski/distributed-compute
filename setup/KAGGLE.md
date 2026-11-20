# Kaggle one-cell training test

1. Open a [new Kaggle notebook](https://www.kaggle.com/code/new).
2. In **Settings**, select an available **GPU** accelerator. Complete any Kaggle verification needed to enable it. Internet can stay off.
3. Copy [the entire smoke test](https://raw.githubusercontent.com/michaelkrymski/distributed-compute/main/setup/kaggle_smoke.py) into one Python cell and run it.

The cell bundles the model, config and smoke test from a pinned Git commit and uses Kaggle's preinstalled PyTorch. No network access, package install, dataset, team key or VPN setup is needed. On a successful run, the baseline 24.39M-parameter model performs three FP32 AdamW updates on generated tokens, restores a checkpoint, and performs one more update. The report includes GPU model, loss, peak memory and elapsed time. Download `/kaggle/working/glm-kaggle-smoke.json` from Kaggle's Files panel to share it with the group.

The test targets under five minutes once the notebook starts executing, with a four-minute execution budget. GPU allocation or account verification can add time before the cell runs. On a T4 x2 session this test uses the first GPU. The smoke test validates local training and checkpoint resume on that GPU. Measuring throughput, training on your corpus, and testing Hivemind peer connectivity require separate runs.
