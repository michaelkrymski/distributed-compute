# Distributed Compute

Create your shared-service account, teams and projects at https://michaelkrymski.me/compute/. Generate a worker key for the chosen project, then enter that key and the shared service origin in this app. The key selects the project and inherits your current team role; it can be remembered in the OS credential store. Shared-service account passwords are managed on the website and are separate from legacy self-hosted administrator-password login.

For models larger than 25M parameters, review device capacity and set the local parameter limit in Settings > Training & storage. External architectures also need an explicitly selected, reviewed local `module:function` workload factory. The server never chooses that import. See [workload adapters](../platform/WORKLOADS.md).

> Personal repository migration: see [MIGRATION.md](../MIGRATION.md). Hosting URLs below are examples; configure your own server or enter your existing team server manually. No backend was moved by this source extraction.

A Slint desktop client for contributors and organizers. Windows and Linux installers are built by the **Desktop installers** GitHub Actions workflow. Download its Windows setup executable or Linux tarball from the latest successful run for `codex/contributor-app`.

## Install and connect

Windows: run `Distributed-Compute-Setup.exe`. It installs for your user without administrator access and offers to open enrollment. Enter the team URL and your reusable member key, or your existing administrator password. Credentials are stored with Windows Credential Manager. The current build is unsigned.

Linux: extract `Distributed-Compute-Linux.tar.gz`, then run `sh install.sh`. It installs to your user data folder and adds a desktop launcher. An unlocked Secret Service / supported OS keyring is required to remember credentials; plaintext storage is refused.

The GUI includes its own Python. GPU computation uses a separate Linux/WSL Python 3.11/3.12 environment with compatible CUDA/ROCm PyTorch, Hivemind, the training tools, and Tailscale. First-run setup detects your Python, offers package installation with stage progress and Debug output, checks your GPU/Tailscale, and saves your computer preferences. Missing WSL, GPU drivers and Tailscale are guided through the platform instructions, not silently installed.

## Contribute

The unified **Team session** screen identifies a benchmark or real model-training room. Join from your computer, toggle the large Ready control, and wait for your organizer to start. The right panel lists participants. The organizer can also contribute by joining a worker. Closing the app stops only its own processes.

Benchmark progress follows the backend clock. After all required reports complete, the forecast uses measured throughput and the benchmark's saved training-token target. It applies to the benchmarked model and excludes corpus loading, evaluation and checkpoint overhead. GPU contribution bars show measured processed-token shares, not hardware utilization.

For corpus training, each contributor must select the same immutable run bundle (local paths may differ). The organizer verifies its bundle and creates a room in **Administration**; this starts a local discovery node. Ready participants receive unique shards and one is selected to initialize a fresh model. Start is a shared startup gate; it is not a permanent quorum. Progress is actual local worker telemetry with nominal global-token estimates. Shared checkpoint recovery remains the existing CLI workflow. Cancel the room and stop discovery when finished.

**Settings** can replace/forget a member key, sign in as administrator, change local paths or rerun the computer wizard. Member keys can be reused across computers. Admin sign-in stores a 30-day bearer device token, never the password. Forget removes the local credential; server-side token revocation UI remains future work. **Debug** contains installation and worker output with credentials redacted.

## Project owners

Read [PROJECT-SETUP.md](PROJECT-SETUP.md). The local owner wizard `node team-dashboard/scripts/manage-team.mjs` must be run from `team-dashboard` as `node scripts/manage-team.mjs`. It adds, rotates, revokes and assigns roles without discarding unrelated keys. Deploy the resulting hash configuration explicitly to the backend; local edits alone do not change production. Provider provisioning is documented and shown in the app's project setup guide; it is not yet an automatic graphical deployment wizard.

Only authorized coordinator keys and the existing owner account can create/start rooms. Additional coordinators can be configured for independent projects using keys; multi-admin password enrollment is not implemented. Keep all keys, datasets and results private and out of Git.

## Run from source

```sh
python desktop/launch.py
```

Requires Python 3.12+ for the GUI. The launcher creates `desktop/.venv`. GPU environments are independent. To run checks:

```sh
python -m pip install -e './desktop[test]'
python -m pytest desktop/tests -q
glm-desktop --check-ui
```

CI tests actual HTTP/D1 room authorization and real two-worker CPU corpus training. Native Windows GUI and Credential Manager have also been checked. Installer builds are tested on Windows and Linux; this does not establish long-running GPU stability or training quality.
