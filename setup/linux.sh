#!/usr/bin/env bash
# Interactive Ubuntu 24.04 x86_64 setup; never run this whole script as root.
set -Eeuo pipefail
backend=auto
plan=0
while (($#)); do
  case "$1" in
    --device) backend="${2:?Choose auto, cuda, rocm or cpu}"; shift 2;;
    --plan) plan=1; shift;;
    --help) echo 'bash setup/linux.sh [--device auto|cuda|rocm|cpu] [--plan]'; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
case "$backend" in auto|cuda|rocm|cpu) ;; *) echo 'Invalid device' >&2; exit 2;; esac
repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv="$HOME/.local/share/glm/training-venv"
if [[ $backend == auto ]]; then
  if command -v nvidia-smi >/dev/null || [[ -x /usr/lib/wsl/lib/nvidia-smi ]] || { compgen -G '/sys/class/drm/card*/device/vendor' >/dev/null && grep -ql '0x10de' /sys/class/drm/card*/device/vendor; }; then backend=cuda
  elif compgen -G '/sys/class/drm/card*/device/vendor' >/dev/null && grep -ql '0x1002' /sys/class/drm/card*/device/vendor; then backend=rocm
  else backend=cpu; fi
fi
if ((plan)); then
  printf 'PLAN ONLY: no changes.\nDevice: %s\nSource: %s\nEnvironment: %s\n' "$backend" "$repo" "$venv"
  printf '1/6 Confirm Ubuntu/source\n2/6 Apt prerequisites\n3/6 Driver check (reboot/login handoff if needed)\n4/6 Tailscale installation and interactive sign-in\n5/6 Python 3.12, PyTorch 2.8.0 and local packages\n6/6 Local device check\n'
  exit 0
fi
[[ $EUID != 0 ]] || { echo 'Run as your normal user; sudo is requested when needed.' >&2; exit 2; }
[[ -t 0 ]] || { echo 'An interactive terminal is required for sudo and Tailscale.' >&2; exit 2; }
[[ $(uname -m) == x86_64 ]] || { echo 'Only x86_64 is supported.' >&2; exit 2; }
source /etc/os-release
[[ $ID == ubuntu && $VERSION_ID == 24.04 ]] || { echo 'Automated setup supports Ubuntu 24.04. Use the manual guide for other OS versions.' >&2; exit 2; }
[[ -f "$repo/training/pyproject.toml" && -f "$repo/hivemind-tools/pyproject.toml" ]] || { echo 'A complete source checkout is required.' >&2; exit 2; }
trap 'printf "\nStopped at line %s. Fix the error above and rerun. No automatic reboot.\n" "$LINENO" >&2' ERR
step() { printf '\n=== %s ===\n' "$1"; }
step '1/6 Confirm installation'
printf 'Device: %s\nEnvironment: %s\nInstalls apt dependencies, Tailscale and several GB of Python packages.\n' "$backend" "$venv"
read -r -p 'Continue? [y/N] ' answer
[[ $answer == y || $answer == Y ]] || exit 0
sudo -v
step '2/6 System prerequisites'
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip build-essential libssl-dev libffi-dev pkg-config curl ca-certificates pciutils
step '3/6 Driver and GPU permissions'
in_wsl=0
if grep -qi microsoft /proc/sys/kernel/osrelease; then in_wsl=1; fi
if [[ $backend == rocm ]]; then
  ((in_wsl == 0)) || { echo 'AMD training uses native Ubuntu in this project.'; exit 20; }
  if [[ ! -e /dev/kfd ]]; then
    echo 'USER TAKEOVER: AMD compute driver is not ready (/dev/kfd missing). Install the OS/GPU-specific driver, reboot if requested, then rerun:'
    echo 'https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html'
    exit 20
  fi
  if [[ ! -r /dev/kfd || ! -w /dev/kfd ]] || ! find /dev/dri -name 'renderD*' -readable -writable -print -quit 2>/dev/null | grep -q .; then
    for group in render video; do
      if getent group "$group" >/dev/null; then sudo usermod -aG "$group" "$(id -un)"; fi
    done
    echo 'GPU groups updated. Sign out of Ubuntu completely, sign in, then rerun.'
    exit 20
  fi
elif [[ $backend == cuda ]]; then
  smi=$(command -v nvidia-smi || true)
  if [[ -z $smi && -x /usr/lib/wsl/lib/nvidia-smi ]]; then smi=/usr/lib/wsl/lib/nvidia-smi; fi
  if [[ -z $smi ]] || ! "$smi" >/dev/null 2>&1; then
    if ((in_wsl)); then
      echo 'USER TAKEOVER: install/update the Windows NVIDIA driver. Restart if requested, then rerun. Never install a Linux NVIDIA driver inside WSL.'
      echo 'https://www.nvidia.com/drivers/'
    else
      sudo apt-get install -y ubuntu-drivers-common
      sudo ubuntu-drivers install
      echo 'NVIDIA driver installation requested. Reboot Ubuntu, then rerun. Secure Boot enrollment may require input at reboot.'
    fi
    exit 20
  fi
fi
step '4/6 Tailscale and interactive sign-in'
if ! command -v tailscale >/dev/null; then
  installer=$(mktemp)
  curl --proto '=https' --tlsv1.2 -fsSL https://tailscale.com/install.sh -o "$installer"
  sh "$installer"
  rm -f -- "$installer"
fi
if [[ $(ps -p 1 -o comm=) == systemd ]]; then
  sudo systemctl enable --now tailscaled
else
  echo 'USER TAKEOVER: enable systemd in WSL, restart that distribution, then rerun. Native Linux must run tailscaled.'
  echo 'https://learn.microsoft.com/windows/wsl/systemd'
  exit 20
fi
status_json=$(tailscale status --json 2>/dev/null || true)
state=$(printf '%s' "$status_json" | python3.12 -c 'import json,sys; print(json.load(sys.stdin).get("BackendState",""))')
if [[ $state != Running ]]; then
  echo 'USER TAKEOVER: use the link/QR below to sign into the SAME tailnet as your team. Complete MFA/approval yourself. This terminal waits here.'
  sudo tailscale up --qr
fi
tailscale status
read -r -p 'Confirm this is the team tailnet, not a separate personal tailnet. Continue? [y/N] ' answer
[[ $answer == y || $answer == Y ]] || { echo 'Ask the organizer for a tailnet invitation, then rerun.'; exit 20; }
step '5/6 Dedicated training environment'
mkdir -p "$(dirname -- "$venv")"
if [[ ! -e "$venv" ]]; then python3.12 -m venv "$venv"; fi
python="$venv/bin/python"
"$python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Setup environment must use Python 3.12"'
"$python" -m pip install --upgrade pip
case "$backend" in cuda) wheel=cu128;; rocm) wheel=rocm6.4;; cpu) wheel=cpu;; esac
"$python" -m pip install "torch==2.8.0+$wheel" --index-url "https://download.pytorch.org/whl/$wheel"
"$python" -m pip install -e "$repo/training" -e "$repo/hivemind-tools"
step '6/6 Local device verification'
"$python" -m glmbench doctor --device "$backend"
printf '\nSetup complete. Training Python: %s\n' "$python"
printf 'Tailscale IPv4: '; tailscale ip -4
printf 'Device: %s\n' "$backend"
if ((in_wsl)); then echo 'Enable WSL in the app and choose this exact distribution name.'; fi
echo 'Check this computer in the app, join the session, then Ready. Setup has not started training.'
