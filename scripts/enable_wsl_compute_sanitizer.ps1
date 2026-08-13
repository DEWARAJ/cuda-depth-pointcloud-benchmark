param(
    [switch]$IUnderstandSecurityRisk
)

$ErrorActionPreference = 'Stop'
$keyPath = 'HKLM:\SOFTWARE\NVIDIA Corporation\GPUDebugger'

if (-not $IUnderstandSecurityRisk) {
    throw 'This machine-wide setting enables NVIDIA GPU debugging. Re-run from an Administrator PowerShell with -IUnderstandSecurityRisk only if you accept that security impact.'
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator PowerShell is required.'
}

New-Item -Path $keyPath -Force | Out-Null
New-ItemProperty -Path $keyPath -Name EnableInterface -PropertyType DWord -Value 1 -Force | Out-Null
Write-Host 'NVIDIA WSL debugger interface enabled. Run `wsl --shutdown`, reopen Ubuntu, then run scripts/run_compute_sanitizer.sh.'
