# Automated computer setup

> Personal repository migration: see [MIGRATION.md](../MIGRATION.md). Hosting URLs below are examples; configure your own server or enter your existing team server manually. No backend was moved by this source extraction.

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
