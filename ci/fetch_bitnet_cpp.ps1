# Fetch and build the external BitNet C++ implementation
# PowerShell version for Windows systems

param(
    [string]$Tag = $(if ($env:BITNET_CPP_TAG) { $env:BITNET_CPP_TAG } else { "v1.0.0" }),
    [string]$CachePath = $(if ($env:BITNET_CPP_PATH) { $env:BITNET_CPP_PATH } else { "$env:USERPROFILE\.cache\bitnet_cpp" }),
    [switch]$Force,
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$BitNetCppRepo = "https://github.com/microsoft/BitNet.git"
$BuildDir = Join-Path $CachePath "build"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Debug {
    param([string]$Message)
    Write-Host "[DEBUG] $Message" -ForegroundColor Blue
}

function Show-Usage {
    @"
Usage: .\fetch_bitnet_cpp.ps1 [OPTIONS]

Fetch and build the external BitNet C++ implementation for cross-validation.

OPTIONS:
    -Tag TAG            Specify BitNet.cpp tag/version (default: $Tag)
    -CachePath PATH     Specify cache directory (default: $CachePath)
    -Force              Force rebuild even if already built
    -Clean              Clean build directory before building
    -Help               Show this help message

ENVIRONMENT VARIABLES:
    BITNET_CPP_TAG      Override default tag/version
    BITNET_CPP_PATH     Override default cache directory

EXAMPLES:
    .\fetch_bitnet_cpp.ps1                      # Use defaults
    .\fetch_bitnet_cpp.ps1 -Tag v1.1.0         # Use specific version
    .\fetch_bitnet_cpp.ps1 -Force              # Force rebuild
    .\fetch_bitnet_cpp.ps1 -Clean -Force       # Clean rebuild

After successful build, set environment variables:
    `$env:BITNET_CPP_PATH = "$CachePath"
"@
}

function Test-Dependencies {
    $MissingDeps = @()
    
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $MissingDeps += "git"
    }
    
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        $MissingDeps += "cmake"
    }
    
    # Check for Visual Studio or Build Tools
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        $MissingDeps += "Visual Studio Build Tools"
    }
    
    if ($MissingDeps.Count -gt 0) {
        Write-Error "Missing required dependencies: $($MissingDeps -join ', ')"
        Write-Error "Please install them and try again:"
        Write-Error "  Git: https://git-scm.com/download/win"
        Write-Error "  CMake: https://cmake.org/download/"
        Write-Error "  Visual Studio: https://visualstudio.microsoft.com/downloads/"
        exit 1
    }
}

function Get-SourceCode {
    Write-Info "Fetching BitNet C++ implementation..."
    Write-Info "Repository: $BitNetCppRepo"
    Write-Info "Tag/Version: $Tag"
    Write-Info "Cache directory: $CachePath"
    
    if (Test-Path (Join-Path $CachePath ".git")) {
        Write-Info "Existing repository found, updating..."
        Push-Location $CachePath
        
        try {
            # Fetch latest changes
            git fetch origin
            
            # Check if we're already on the right tag
            $CurrentTag = git describe --tags --exact-match 2>$null
            if ($CurrentTag -eq $Tag) {
                Write-Info "Already on correct tag: $Tag"
                return
            }
            
            # Clean any local changes
            git reset --hard
            git clean -fd
            
            # Checkout the specified tag
            git checkout $Tag
        }
        finally {
            Pop-Location
        }
    } else {
        Write-Info "Cloning fresh repository..."
        
        # Create cache directory
        $ParentDir = Split-Path $CachePath -Parent
        if (-not (Test-Path $ParentDir)) {
            New-Item -ItemType Directory -Path $ParentDir -Force | Out-Null
        }
        
        # Clone the repository
        git clone --depth 1 --branch $Tag $BitNetCppRepo $CachePath
    }
    
    Write-Info "Source code fetched successfully"
}

function Invoke-Build {
    Write-Info "Building BitNet C++ implementation..."
    
    Push-Location $CachePath
    
    try {
        # Create build directory
        if (-not (Test-Path $BuildDir)) {
            New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
        }
        
        Push-Location $BuildDir
        
        try {
            # Find Visual Studio
            $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
            $VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            
            if (-not $VsPath) {
                throw "Visual Studio with C++ tools not found"
            }
            
            # Configure with CMake
            Write-Info "Configuring build with CMake..."
            cmake .. `
                -DCMAKE_BUILD_TYPE=Release `
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
                -DBUILD_SHARED_LIBS=ON `
                "-DCMAKE_INSTALL_PREFIX=$BuildDir\install"
            
            if ($LASTEXITCODE -ne 0) {
                throw "CMake configuration failed"
            }
            
            # Build
            Write-Info "Building (this may take a few minutes)..."
            cmake --build . --config Release --parallel
            
            if ($LASTEXITCODE -ne 0) {
                throw "Build failed"
            }
            
            # Install to local directory
            Write-Info "Installing to local directory..."
            cmake --install . --config Release
            
            if ($LASTEXITCODE -ne 0) {
                throw "Installation failed"
            }
            
            Write-Info "Build completed successfully"
        }
        finally {
            Pop-Location
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-ApplyPatches {
    Write-Info "Checking for patches to apply..."
    
    $PatchScript = Join-Path $PSScriptRoot "apply_patches.ps1"
    if (Test-Path $PatchScript) {
        Write-Info "Applying patches..."
        & $PatchScript -CppPath $CachePath
    } else {
        Write-Info "No patch application script found - using C++ implementation as-is"
    }
}

function Test-Build {
    Write-Info "Validating build..."
    
    $LibDir = Join-Path $BuildDir "install\lib"
    $IncludeDir = Join-Path $BuildDir "install\include"
    
    # Check for expected directories
    if (-not (Test-Path $LibDir)) {
        Write-Error "Library directory not found: $LibDir"
        return $false
    }
    
    if (-not (Test-Path $IncludeDir)) {
        Write-Error "Include directory not found: $IncludeDir"
        return $false
    }
    
    # Look for library files
    $LibFiles = Get-ChildItem -Path $LibDir -Recurse -Include "*.lib", "*.dll" -ErrorAction SilentlyContinue
    if ($LibFiles.Count -eq 0) {
        Write-Warn "No library files found in $LibDir"
        Write-Warn "This may be expected if only static libraries were built"
    } else {
        Write-Info "Found $($LibFiles.Count) library file(s)"
    }
    
    # Look for header files
    $HeaderFiles = Get-ChildItem -Path $IncludeDir -Recurse -Include "*.h", "*.hpp" -ErrorAction SilentlyContinue
    if ($HeaderFiles.Count -eq 0) {
        Write-Error "No header files found in $IncludeDir"
        return $false
    } else {
        Write-Info "Found $($HeaderFiles.Count) header file(s)"
    }
    
    Write-Info "Build validation passed"
    return $true
}

function New-EnvScript {
    $EnvScript = Join-Path $CachePath "setup_env.ps1"
    
    Write-Info "Creating environment setup script: $EnvScript"
    
    $EnvContent = @"
# Environment setup for BitNet C++ cross-validation
# Run this script to set up environment variables

`$env:BITNET_CPP_PATH = "$CachePath"
`$env:BITNET_CPP_LIB_PATH = "$BuildDir\install\lib"
`$env:BITNET_CPP_INCLUDE_PATH = "$BuildDir\install\include"

# Add to PATH for DLLs
`$env:PATH = "`$env:BITNET_CPP_LIB_PATH;`$env:PATH"

Write-Host "BitNet C++ environment configured:" -ForegroundColor Green
Write-Host "  Path: `$env:BITNET_CPP_PATH" -ForegroundColor Green
Write-Host "  Libraries: `$env:BITNET_CPP_LIB_PATH" -ForegroundColor Green
Write-Host "  Headers: `$env:BITNET_CPP_INCLUDE_PATH" -ForegroundColor Green
"@
    
    Set-Content -Path $EnvScript -Value $EnvContent -Encoding UTF8
}

# Main execution
function Main {
    if ($Help) {
        Show-Usage
        return
    }
    
    Write-Info "BitNet C++ Fetch and Build Script"
    Write-Info "=================================="
    
    # Check if already built and not forcing rebuild
    if ((Test-Path $BuildDir) -and (Test-Path (Join-Path $BuildDir "install")) -and (-not $Force)) {
        Write-Info "BitNet C++ already built at $CachePath"
        Write-Info "Use -Force to rebuild or -Clean -Force for clean rebuild"
        Write-Info "To use: . $CachePath\setup_env.ps1"
        return
    }
    
    # Check dependencies
    Test-Dependencies
    
    # Clean if requested
    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Info "Cleaning build directory..."
        Remove-Item -Path $BuildDir -Recurse -Force
    }
    
    # Fetch source code
    Get-SourceCode
    
    # Apply patches
    Invoke-ApplyPatches
    
    # Build
    Invoke-Build
    
    # Validate
    if (-not (Test-Build)) {
        Write-Error "Build validation failed"
        exit 1
    }
    
    # Create environment script
    New-EnvScript
    
    Write-Info "BitNet C++ setup completed successfully!"
    Write-Info ""
    Write-Info "To use in your shell:"
    Write-Info "  . $CachePath\setup_env.ps1"
    Write-Info ""
    Write-Info "To use in Rust cross-validation:"
    Write-Info "  `$env:BITNET_CPP_PATH = `"$CachePath`""
    Write-Info "  cargo test --features crossval"
    Write-Info ""
    Write-Info "Cache location: $CachePath"
    Write-Info "Build artifacts: $BuildDir"
}

# Run main function
Main