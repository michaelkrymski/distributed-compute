# Setting up your own team

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

Contributors install the client, enter the team URL and their reusable member key, and follow the computer wizard. One member key can enroll many computers. Keys are saved using Windows Credential Manager or Linux Secret Service; Linux users need an unlocked keyring (usually GNOME Keyring). No plaintext fallback is enabled.

Project owners need GitHub, Netlify, and a backend hosting account. Netlify hosts the static website; the existing backend runs on Cloudflare Workers with a D1 database. Netlify alone does not run Hivemind or store shared team data. A computer must keep the discovery process and training workers running. Provider account creation and billing remain owner-operated.

1. Fork/clone the repository. Use `team-dashboard/` for the website/backend. Install Node 24 and run `npm ci` there.
2. Run `node scripts/manage-team.mjs`. Add your account as `coordinator`; add each teammate as `member`. The wizard can be rerun to add, rotate, revoke, or change roles without replacing unrelated keys. Keep `work/` private. Existing deployments must preserve their hash map before editing; never initialize over production keys.
3. Provision a new backend Site using Sites, with a new project ID and D1 binding `DB`. The repository's existing `.openai/hosting.json` points to Michael's deployment: do not publish a fork to that project. Configure secret `TEAM_KEYS_JSON` from `work/team-key-hashes.json`. Keys themselves are not deployed.
4. Update the allowed origins in `lib/admin-auth.ts`, the public `_redirects`, and the root `netlify.toml` to your backend/frontend URLs. Build the backend (`npm run build`) and deploy with its migrations. Existing migrations must run exactly once, including `0003_desktop_rooms.sql`.
5. Build the frontend (`npm run build:frontend`). Import the repository into Netlify using its root `netlify.toml`, or upload the contents of `frontend-dist`. Keep `_redirects` so `/api/*` reaches your backend. Never put a secret into a `VITE_*` variable.
6. Share the frontend URL and individual member keys privately. The client Settings screen can switch servers and replace saved keys. Administrator password activation currently uses the existing owner account; additional selected coordinators can administer using their keys. Configurable multi-admin password accounts are not implemented yet.

## Real training rooms

Prepare an immutable training bundle using `glmtrain prepare` (see `training/README.md`) and distribute it privately to participants. Each computer selects its local bundle path in Settings. The organizer verifies its bundle in Administration and creates a room. The app starts a local rendezvous on Tailscale port 40101. Keep it open. Members join, verify the matching fingerprint, and ready up. Start freezes the online ready computers, assigns unique shards, and selects one fresh-model leader. All selected workers start through the existing trainer startup barrier. Each room is bounded by the bundle's shard count. A reusable member key can join from several computers under separate instance IDs.

Rooms currently start fresh models; recovery of an interrupted team run remains the existing CLI checkpoint workflow. Room readiness is a startup gate, not a permanent all-worker quorum. Model-training progress is nominal worker telemetry, not an exact global token ledger. Stop affects your own worker; the organizer can cancel a room through the room API. Do not reuse the same checkpoint output directory across simultaneous workers.

## Installers and platform setup

GitHub Actions builds Windows and Linux GUI bundles. Windows uses an Inno Setup per-user installer; Linux uses a bundle with `install.sh` and a desktop launcher. No system Python is needed for the GUI. Enrollment opens on first launch after install, so keys never enter installer command lines or logs. Windows binaries are unsigned until the project owner configures code signing.

GPU computation still needs an independent Python 3.11/3.12 environment. Use the platform commands in `training/BENCHMARK.md` to install CUDA/ROCm-compatible PyTorch first. WSL/driver/Tailscale installation may require administrator access or reboot. The client finds existing Python and runs package installation only on request, with live output in Debug. It does not silently change GPU drivers or system networking. An unprepared computer is not yet a fully automated one-click setup.

## Credentials

Administrators can exchange their existing password for a 30-day desktop token; only that token is stored. Failed sign-ins use the existing account's five-attempt lockout. Member keys have their existing lifetime until owner rotation/revocation. Forgetting a saved credential removes the local copy. Server-side device-token revocation and multi-admin password enrollment still need UI support.

## Public server discovery

The desktop begins with an unauthenticated directory from the default team deployment (`GET /api/servers`). Only entries explicitly marked public are returned: display name, description and HTTPS origin. Keys, membership, device details, rooms, peer addresses and results are never included. Choosing a private origin does not register it, contact the directory with that address or forward a remembered credential to it.

This is a curated directory, not automatic federation. Coordinator credentials for the directory deployment can create/update a listing with `POST /api/servers`: `{name, description, origin, isPublic}`. Set `isPublic` to false to unlist it. Public listing does not grant access to protected team APIs. Migration 0004 creates an empty directory; add your own approved records after deployment. Private server entry stays separate from publication.
