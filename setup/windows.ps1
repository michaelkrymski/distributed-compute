# Interactive Windows NVIDIA/CPU entry point; rerun after required restarts.
[CmdletBinding()]
param(
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [ValidatePattern('^[A-Za-z0-9._-]+$')][string]$Distro = 'Ubuntu-24.04',
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($Plan) {
    Write-Output "PLAN ONLY: no changes. Check Windows driver ($Device); install WSL2 $Distro; Ubuntu user handoff; Linux setup; Tailscale sign-in handoff; Python packages and device check."
    Write-Output "Source: $sourceRoot"
    return
}
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'x86_64 Windows is required.' }
if ($Device -eq 'cuda') {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-Host 'USER TAKEOVER: install the NVIDIA Windows driver for this laptop/GPU. Use the vendor installer and restart if requested.'
        Start-Process 'https://www.nvidia.com/drivers/'
        Read-Host 'After installing (or after reboot, rerun this script), press Enter to check again' | Out-Null
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $smi) { throw 'Windows NVIDIA driver is not available yet. Complete installation/reboot, then rerun.' }
    }
    & $smi.Source
    if ($LASTEXITCODE -ne 0) { throw 'Windows NVIDIA driver check failed. Repair/update the driver and rerun.' }
}
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) { throw 'WSL command is missing. Update Windows and follow https://learn.microsoft.com/windows/wsl/install then rerun.' }
$distributions = @()
if ($wsl) { $distributions = @(& wsl.exe --list --quiet 2>$null | ForEach-Object { ($_ -replace "`0",'').Trim() }) }
if ($Distro -notin $distributions) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) { throw 'Open PowerShell as Administrator and rerun to install WSL/Ubuntu. After installation, use your normal account terminal.' }
    Write-Host "Installing WSL2 and $Distro. BIOS virtualization or a restart may be needed. No automatic restart."
    & wsl.exe --install --distribution $Distro --no-launch
    if ($LASTEXITCODE -notin @(0,3010)) { throw 'WSL installation did not finish. Follow its error, restart if needed, then rerun.' }
    Write-Host "USER TAKEOVER: restart if requested. Run wsl -d $Distro, create your Linux username/password, and exit. Rerun this script in your normal PowerShell."
    return
}
& wsl.exe --set-version $Distro 2
if ($LASTEXITCODE -ne 0) { throw 'WSL2 is not ready. Complete its setup/restart, then rerun.' }
Write-Host "USER TAKEOVER: if Ubuntu is not initialized, open wsl -d $Distro, create a Linux username/password, and exit."
Read-Host 'Press Enter when your normal Linux account is ready' | Out-Null
$uid = (& wsl.exe --distribution $Distro -- id -u | Out-String).Trim() -replace "`0",''
if ($LASTEXITCODE -ne 0 -or $uid -eq '0') { throw "Set up a non-root default Ubuntu user in $Distro, then rerun." }
$linuxPath = (& wsl.exe --distribution $Distro -- wslpath -a -u $sourceRoot | Out-String).Trim() -replace "`0",''
if ($LASTEXITCODE -ne 0 -or -not $linuxPath.StartsWith('/')) { throw 'Could not translate the source path into WSL.' }
Write-Host 'Continuing in Ubuntu. Enter sudo passwords and complete Tailscale sign-in yourself when prompted.'
& wsl.exe --distribution $Distro -- bash "$linuxPath/setup/linux.sh" --device $Device
if ($LASTEXITCODE -eq 20) { Write-Host 'Complete the manual action/restart above, then rerun the same setup command.'; return }
if ($LASTEXITCODE -ne 0) { throw 'Linux setup stopped. Read the error above and rerun after fixing it.' }
Write-Host "In the app enable WSL and set distribution to $Distro. Use the Python path printed above."
