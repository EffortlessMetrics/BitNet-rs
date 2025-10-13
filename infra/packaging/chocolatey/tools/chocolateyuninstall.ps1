# Chocolatey uninstall script for BitNet.rs

$ErrorActionPreference = 'Stop'

$packageName = 'bitnet-rs'

# Remove shims
$binaries = @('bitnet-cli', 'bitnet-server')

foreach ($binary in $binaries) {
    try {
        Uninstall-BinFile -Name $binary
        Write-Host "Removed $binary shim" -ForegroundColor Green
    } catch {
        Write-Warning "Could not remove $binary shim: $_"
    }
}

Write-Host "BitNet.rs has been uninstalled successfully!" -ForegroundColor Green
