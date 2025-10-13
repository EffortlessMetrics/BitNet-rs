# BitNet.rs Installation Script for Windows
# This script installs the latest BitNet.rs binaries for Windows

param(
    [string]$InstallDir = "$env:USERPROFILE\.local\bin",
    [string]$Version = "latest",
    [switch]$CliOnly,
    [switch]$ServerOnly,
    [switch]$Force,
    [switch]$Help
)

# Configuration
$Repo = "microsoft/BitNet"
$GitHubAPI = "https://api.github.com/repos/$Repo"
$TempDir = [System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()

# Create temp directory
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

# Cleanup function
function Cleanup {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Register cleanup
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup }

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Help function
function Show-Help {
    @"
BitNet.rs Installation Script for Windows

USAGE:
    .\install.ps1 [OPTIONS]

OPTIONS:
    -InstallDir DIR     Installation directory (default: ~/.local/bin)
    -Version VER        Install specific version (default: latest)
    -CliOnly            Install only the CLI tool
    -ServerOnly         Install only the server
    -Force              Force reinstallation even if already installed
    -Help               Show this help message

EXAMPLES:
    .\install.ps1                                    # Install latest version
    .\install.ps1 -InstallDir "C:\Program Files"     # Install to system directory
    .\install.ps1 -Version "v1.0.0"                  # Install specific version
    .\install.ps1 -CliOnly                           # Install only bitnet-cli

ENVIRONMENT VARIABLES:
    BITNET_INSTALL_DIR          Installation directory
    GITHUB_TOKEN                GitHub token for API access (optional)

For more information, visit: https://github.com/$Repo
"@
}

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Override install directory from environment variable
if ($env:BITNET_INSTALL_DIR) {
    $InstallDir = $env:BITNET_INSTALL_DIR
}

# Set installation flags
$InstallCli = $true
$InstallServer = $true

if ($CliOnly) {
    $InstallServer = $false
}
if ($ServerOnly) {
    $InstallCli = $false
}

# Detect platform and architecture
function Get-Platform {
    $arch = $env:PROCESSOR_ARCHITECTURE

    switch ($arch) {
        "AMD64" { return "x86_64-pc-windows-msvc" }
        "ARM64" { return "aarch64-pc-windows-msvc" }
        default {
            Write-Error "Unsupported architecture: $arch"
            exit 1
        }
    }
}

# Get latest release version
function Get-LatestVersion {
    $apiUrl = "$GitHubAPI/releases/latest"
    $headers = @{}

    if ($env:GITHUB_TOKEN) {
        $headers["Authorization"] = "token $env:GITHUB_TOKEN"
    }

    try {
        $response = Invoke-RestMethod -Uri $apiUrl -Headers $headers
        return $response.tag_name
    }
    catch {
        Write-Error "Failed to get latest version: $_"
        exit 1
    }
}

# Download and extract binary
function Install-BitNet {
    $platform = Get-Platform

    if ($Version -eq "latest") {
        $version = Get-LatestVersion
        if (-not $version) {
            Write-Error "Failed to get latest version"
            exit 1
        }
    }
    else {
        $version = $Version
    }

    Write-Info "Installing BitNet.rs $version for $platform"

    # Construct download URL
    $filename = "bitnet-$platform.zip"
    $downloadUrl = "https://github.com/$Repo/releases/download/$version/$filename"
    $downloadPath = Join-Path $TempDir $filename

    Write-Info "Downloading from: $downloadUrl"

    # Download
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $downloadPath -UseBasicParsing
    }
    catch {
        Write-Error "Download failed: $_"
        exit 1
    }

    # Verify download
    if (-not (Test-Path $downloadPath)) {
        Write-Error "Download failed: $filename not found"
        exit 1
    }

    # Extract
    Write-Info "Extracting binaries..."
    $extractPath = Join-Path $TempDir "extracted"
    try {
        Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force
    }
    catch {
        Write-Error "Extraction failed: $_"
        exit 1
    }

    # Create installation directory
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    }

    # Install binaries
    $installedCount = 0

    if ($InstallCli) {
        $cliSource = Join-Path $extractPath "bitnet-cli.exe"
        $cliTarget = Join-Path $InstallDir "bitnet-cli.exe"

        if (Test-Path $cliSource) {
            if ($Force -or -not (Test-Path $cliTarget)) {
                Copy-Item -Path $cliSource -Destination $cliTarget -Force
                Write-Success "Installed bitnet-cli.exe to $InstallDir"
                $installedCount++
            }
            else {
                Write-Warning "bitnet-cli.exe already exists (use -Force to overwrite)"
            }
        }
    }

    if ($InstallServer) {
        $serverSource = Join-Path $extractPath "bitnet-server.exe"
        $serverTarget = Join-Path $InstallDir "bitnet-server.exe"

        if (Test-Path $serverSource) {
            if ($Force -or -not (Test-Path $serverTarget)) {
                Copy-Item -Path $serverSource -Destination $serverTarget -Force
                Write-Success "Installed bitnet-server.exe to $InstallDir"
                $installedCount++
            }
            else {
                Write-Warning "bitnet-server.exe already exists (use -Force to overwrite)"
            }
        }
    }

    if ($installedCount -eq 0) {
        Write-Warning "No binaries were installed"
        exit 1
    }
}

# Check if installation directory is in PATH
function Test-PathVariable {
    $pathDirs = $env:PATH -split ';'
    $normalizedInstallDir = [System.IO.Path]::GetFullPath($InstallDir)

    $inPath = $false
    foreach ($dir in $pathDirs) {
        if ($dir -and ([System.IO.Path]::GetFullPath($dir) -eq $normalizedInstallDir)) {
            $inPath = $true
            break
        }
    }

    if (-not $inPath) {
        Write-Warning "Installation directory $InstallDir is not in your PATH"
        Write-Info "Add it to your PATH using one of these methods:"
        Write-Info "  1. System Properties > Environment Variables > PATH"
        Write-Info "  2. PowerShell: `$env:PATH += ';$InstallDir'"
        Write-Info "  3. Command Prompt: setx PATH `"%PATH%;$InstallDir`""
    }
}

# Verify installation
function Test-Installation {
    $verified = 0

    if ($InstallCli) {
        $cliPath = Join-Path $InstallDir "bitnet-cli.exe"
        if (Test-Path $cliPath) {
            Write-Info "Verifying bitnet-cli.exe installation..."
            try {
                $output = & $cliPath --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "bitnet-cli.exe is working correctly"
                    $verified++
                }
                else {
                    Write-Error "bitnet-cli.exe verification failed"
                }
            }
            catch {
                Write-Error "bitnet-cli.exe verification failed: $_"
            }
        }
    }

    if ($InstallServer) {
        $serverPath = Join-Path $InstallDir "bitnet-server.exe"
        if (Test-Path $serverPath) {
            Write-Info "Verifying bitnet-server.exe installation..."
            try {
                $output = & $serverPath --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "bitnet-server.exe is working correctly"
                    $verified++
                }
                else {
                    Write-Error "bitnet-server.exe verification failed"
                }
            }
            catch {
                Write-Error "bitnet-server.exe verification failed: $_"
            }
        }
    }

    return $verified -gt 0
}

# Main installation process
function Main {
    Write-Info "🦀 BitNet.rs Installation Script for Windows"
    Write-Info "Installing to: $InstallDir"

    # Check prerequisites
    if (-not (Get-Command Expand-Archive -ErrorAction SilentlyContinue)) {
        Write-Error "PowerShell 5.0 or later is required for Expand-Archive"
        exit 1
    }

    # Perform installation
    try {
        Install-BitNet

        # Verify installation
        if (Test-Installation) {
            Write-Success "🎉 BitNet.rs installation completed successfully!"

            # Show usage examples
            Write-Host ""
            Write-Info "Quick start examples:"
            if ($InstallCli) {
                Write-Host "  $InstallDir\bitnet-cli.exe --help"
            }
            if ($InstallServer) {
                Write-Host "  $InstallDir\bitnet-server.exe --port 8080"
            }

            # Check PATH
            Test-PathVariable

            Write-Host ""
            Write-Info "For documentation and examples, visit:"
            Write-Info "  https://github.com/$Repo"
        }
        else {
            Write-Error "Installation verification failed"
            exit 1
        }
    }
    catch {
        Write-Error "Installation failed: $_"
        exit 1
    }
    finally {
        Cleanup
    }
}

# Run main function
Main
