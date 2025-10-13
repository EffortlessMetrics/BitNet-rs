# Chocolatey install script for BitNet.rs

$ErrorActionPreference = 'Stop'

$packageName = 'bitnet-rs'
$toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"
$version = $env:ChocolateyPackageVersion

# Determine architecture
$arch = if ([Environment]::Is64BitOperatingSystem) {
    if ([Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITEW6432") -eq "ARM64" -or
        [Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE") -eq "ARM64") {
        "aarch64"
    } else {
        "x86_64"
    }
} else {
    throw "32-bit Windows is not supported"
}

$target = "$arch-pc-windows-msvc"
$url = "https://github.com/microsoft/BitNet/releases/download/v$version/bitnet-$target.zip"

$packageArgs = @{
    packageName    = $packageName
    unzipLocation  = $toolsDir
    url            = $url
    checksum       = 'PLACEHOLDER_CHECKSUM'
    checksumType   = 'sha256'
}

# Download and extract
Install-ChocolateyZipPackage @packageArgs

# Create shims for executables
$binaries = @('bitnet-cli.exe', 'bitnet-server.exe')

foreach ($binary in $binaries) {
    $binaryPath = Join-Path $toolsDir $binary
    if (Test-Path $binaryPath) {
        # Create shim
        Install-BinFile -Name ($binary -replace '\.exe$', '') -Path $binaryPath

        Write-Host "Installed $binary" -ForegroundColor Green
    } else {
        Write-Warning "Binary not found: $binary"
    }
}

# Verify installation
try {
    $cliPath = Join-Path $toolsDir 'bitnet-cli.exe'
    if (Test-Path $cliPath) {
        $version = & $cliPath --version 2>&1
        Write-Host "BitNet.rs CLI installed successfully: $version" -ForegroundColor Green
    }

    $serverPath = Join-Path $toolsDir 'bitnet-server.exe'
    if (Test-Path $serverPath) {
        $version = & $serverPath --version 2>&1
        Write-Host "BitNet.rs Server installed successfully: $version" -ForegroundColor Green
    }
} catch {
    Write-Warning "Could not verify installation: $_"
}

Write-Host ""
Write-Host "BitNet.rs has been installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Yellow
Write-Host "  bitnet-cli --help" -ForegroundColor White
Write-Host "  bitnet-server --help" -ForegroundColor White
Write-Host ""
Write-Host "Documentation: https://docs.rs/bitnet" -ForegroundColor Cyan
Write-Host "Repository: https://github.com/microsoft/BitNet" -ForegroundColor Cyan
