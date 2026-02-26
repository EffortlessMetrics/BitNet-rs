# Chocolatey uninstall script for bitnet-rs

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

Write-Host "bitnet-rs has been uninstalled successfully!" -ForegroundColor Green
