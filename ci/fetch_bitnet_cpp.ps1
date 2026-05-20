# Fetch and build the external BitNet C++ implementation
# PowerShell version for Windows systems

param(
    # Microsoft BitNet publishes the reference implementation from the main
    # branch rather than release tags; keep the default aligned with the bash
    # fetch script.
    [string]$Tag = $(if ($env:BITNET_CPP_TAG) { $env:BITNET_CPP_TAG } else { "main" }),
    [string]$CachePath = $(if ($env:BITNET_CPP_PATH) { $env:BITNET_CPP_PATH } else { "$env:USERPROFILE\.cache\bitnet_cpp" }),
    [switch]$Force,
    [switch]$Clean,
    [switch]$SkipPatches,
    [int]$ConfigureTimeoutMinutes = $(if ($env:BITNET_CPP_CONFIGURE_TIMEOUT_MINUTES) { [int]$env:BITNET_CPP_CONFIGURE_TIMEOUT_MINUTES } else { 0 }),
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$BitNetCppRepo = "https://github.com/microsoft/BitNet.git"

if (-not [System.IO.Path]::IsPathRooted($CachePath)) {
    $CachePath = Join-Path (Get-Location) $CachePath
}
$CachePath = [System.IO.Path]::GetFullPath($CachePath)
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

function Test-IsWindows {
    return [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
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
    -SkipPatches        Use upstream C++ source as-is without applying local patches
    -ConfigureTimeoutMinutes MIN
                        Stop CMake configure after MIN minutes (default: 0, disabled)
    -Help               Show this help message

ENVIRONMENT VARIABLES:
    BITNET_CPP_TAG      Override default tag/version
    BITNET_CPP_PATH     Override default cache directory
    BITNET_CPP_CONFIGURE_TIMEOUT_MINUTES
                        Optional CMake configure timeout in minutes

EXAMPLES:
    .\fetch_bitnet_cpp.ps1                      # Use defaults
    .\fetch_bitnet_cpp.ps1 -Tag v1.1.0         # Use specific version
    .\fetch_bitnet_cpp.ps1 -Force              # Force rebuild
    .\fetch_bitnet_cpp.ps1 -Clean -Force       # Clean rebuild
    .\fetch_bitnet_cpp.ps1 -SkipPatches        # Build upstream source without patches
    .\fetch_bitnet_cpp.ps1 -ConfigureTimeoutMinutes 120
                                                # Bound CMake configure without external cleanup

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

    if (Test-IsWindows) {
        $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $VsWhere)) {
            $MissingDeps += "Visual Studio Build Tools"
        } else {
            $VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if (-not $VsPath) {
                $MissingDeps += "Visual Studio C++ build tools"
            }

            $Generator = if ($env:CMAKE_GENERATOR) {
                $env:CMAKE_GENERATOR
            } else {
                "Visual Studio 17 2022"
            }
            if ($Generator.StartsWith("Visual Studio", [System.StringComparison]::OrdinalIgnoreCase)) {
                $Platform = if ($env:CMAKE_GENERATOR_PLATFORM) {
                    $env:CMAKE_GENERATOR_PLATFORM
                } else {
                    "x64"
                }
                $Toolset = if ($env:CMAKE_GENERATOR_TOOLSET) {
                    $env:CMAKE_GENERATOR_TOOLSET
                } else {
                    "ClangCL"
                }
                $ToolsetPath = Join-Path $VsPath "MSBuild\Microsoft\VC\v170\Platforms\$Platform\PlatformToolsets\$Toolset"
                if (-not (Test-Path $ToolsetPath)) {
                    $MissingDeps += "Visual Studio $Toolset toolset for $Platform"
                }
            } else {
                # Non-Visual-Studio generators need compiler executables on PATH.
                if (-not (Get-Command clang -ErrorAction SilentlyContinue)) {
                    $MissingDeps += "clang"
                }

                if (-not (Get-Command clang++ -ErrorAction SilentlyContinue)) {
                    $MissingDeps += "clang++"
                }
            }
        }
    }

    if ($MissingDeps.Count -gt 0) {
        Write-Error "Missing required dependencies: $($MissingDeps -join ', ')"
        Write-Error "Please install them and try again:"
        Write-Error "  Git: https://git-scm.com/download/win"
        Write-Error "  CMake: https://cmake.org/download/"
        Write-Error "  Visual Studio: https://visualstudio.microsoft.com/downloads/"
        if (Test-IsWindows) {
            Write-Error "  Visual Studio components: C++ Clang Compiler for Windows and MS-Build Support for LLVM-Toolset"
        }
        exit 1
    }
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed"
    }
}

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process
    )

    if ($Process.HasExited) {
        return
    }

    if (Test-IsWindows) {
        & taskkill.exe /PID $Process.Id /T /F | ForEach-Object {
            Write-Warn $_
        }
    } else {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [string]$Description = $FilePath,

        [int]$TimeoutMinutes = 0
    )

    Write-Info "Running $Description"
    Write-Debug "$FilePath $($Arguments -join ' ')"

    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $FilePath
    $StartInfo.UseShellExecute = $false
    foreach ($Argument in $Arguments) {
        [void]$StartInfo.ArgumentList.Add($Argument)
    }

    $Process = [System.Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    [void]$Process.Start()

    if ($TimeoutMinutes -gt 0) {
        $TimeoutMs = [int64]$TimeoutMinutes * 60 * 1000
        if (-not $Process.WaitForExit($TimeoutMs)) {
            Write-Error "$Description exceeded timeout: $TimeoutMinutes minute(s)"
            Stop-ProcessTree -Process $Process
            throw "$Description timed out after $TimeoutMinutes minute(s)"
        }
    } else {
        $Process.WaitForExit()
    }

    if ($Process.ExitCode -ne 0) {
        throw "$Description failed with exit code $($Process.ExitCode)"
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
            Invoke-Git @("fetch", "origin")

            $OldErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $CurrentTag = git describe --tags --exact-match 2>$null
            $DescribeExitCode = $LASTEXITCODE
            $ErrorActionPreference = $OldErrorActionPreference

            $CurrentBranch = git rev-parse --abbrev-ref HEAD
            if ($LASTEXITCODE -ne 0) {
                throw "git rev-parse --abbrev-ref HEAD failed"
            }

            # Clean any local changes before moving refs.
            Invoke-Git @("reset", "--hard")
            Invoke-Git @("clean", "-fd")

            if ($DescribeExitCode -eq 0 -and $CurrentTag -eq $Tag) {
                Write-Info "Already on correct tag: $Tag"
                Invoke-Git @("submodule", "update", "--init", "--recursive")
            } elseif ($CurrentBranch -eq $Tag) {
                Write-Info "Already on branch $Tag; fast-forwarding to origin/$Tag"
                Invoke-Git @("reset", "--hard", "origin/$Tag")
                Invoke-Git @("submodule", "update", "--init", "--recursive")
            } else {
                # Checkout the specified tag or branch.
                Invoke-Git @("checkout", $Tag)
                $CheckoutBranch = git rev-parse --abbrev-ref HEAD
                if ($LASTEXITCODE -ne 0) {
                    throw "git rev-parse --abbrev-ref HEAD failed after checkout"
                }
                if ($CheckoutBranch -eq $Tag) {
                    Write-Info "Checked out branch $Tag; fast-forwarding to origin/$Tag"
                    Invoke-Git @("reset", "--hard", "origin/$Tag")
                }
                Invoke-Git @("submodule", "update", "--init", "--recursive")
            }
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
        Invoke-Git @(
            "clone",
            "--depth",
            "1",
            "--recurse-submodules",
            "--shallow-submodules",
            "--branch",
            $Tag,
            $BitNetCppRepo,
            $CachePath
        )
        Push-Location $CachePath
        try {
            Invoke-Git @("submodule", "update", "--init", "--recursive")
        }
        finally {
            Pop-Location
        }
    }

    $LlamaCMake = Join-Path $CachePath "3rdparty\llama.cpp\CMakeLists.txt"
    if (-not (Test-Path $LlamaCMake)) {
        throw "llama.cpp submodule is not initialized; expected $LlamaCMake"
    }

    Write-Info "Source code fetched successfully"
}

function Ensure-KernelHeader {
    $KernelHeader = Join-Path $CachePath "include\bitnet-lut-kernels.h"
    $IncludeDir = Split-Path $KernelHeader -Parent
    if ((Test-Path $KernelHeader) -or (-not (Test-Path $IncludeDir))) {
        return
    }

    $PresetRoot = Join-Path $CachePath "preset_kernels"
    $PresetKernel = $null
    if (Test-Path $PresetRoot) {
        foreach ($PresetDir in (Get-ChildItem -Path $PresetRoot -Directory | Sort-Object Name)) {
            $Tl2 = Join-Path $PresetDir.FullName "bitnet-lut-kernels-tl2.h"
            $Generic = Join-Path $PresetDir.FullName "bitnet-lut-kernels.h"
            if (Test-Path $Tl2) {
                $PresetKernel = $Tl2
                break
            }
            if (Test-Path $Generic) {
                $PresetKernel = $Generic
                break
            }
        }
    }

    if ($PresetKernel) {
        Write-Warn "bitnet-lut-kernels.h missing, copying from preset: $PresetKernel"
        Copy-Item -Path $PresetKernel -Destination $KernelHeader -Force
    }

    if (-not (Test-Path $KernelHeader)) {
        Write-Warn "bitnet-lut-kernels.h not found. Build may succeed without it."
        Write-Warn "If build fails, check the Microsoft BitNet repository structure."
    }
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

            $CMakeArgs = @(
                "..",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DBUILD_SHARED_LIBS=ON",
                "-DCMAKE_INSTALL_PREFIX=$BuildDir\install"
            )

            if (Test-IsWindows) {
                $Generator = if ($env:CMAKE_GENERATOR) {
                    $env:CMAKE_GENERATOR
                } else {
                    "Visual Studio 17 2022"
                }
                $CMakeArgs += @("-G", $Generator)

                if ($Generator.StartsWith("Visual Studio", [System.StringComparison]::OrdinalIgnoreCase)) {
                    $Platform = if ($env:CMAKE_GENERATOR_PLATFORM) {
                        $env:CMAKE_GENERATOR_PLATFORM
                    } else {
                        "x64"
                    }
                    $Toolset = if ($env:CMAKE_GENERATOR_TOOLSET) {
                        $env:CMAKE_GENERATOR_TOOLSET
                    } else {
                        "ClangCL"
                    }
                    $CMakeArgs += @("-A", $Platform, "-T", $Toolset)
                } else {
                    $CMakeArgs += @(
                        "-DCMAKE_C_COMPILER=clang",
                        "-DCMAKE_CXX_COMPILER=clang++"
                    )
                }
            }

            # Configure with CMake
            if ($ConfigureTimeoutMinutes -gt 0) {
                Write-Info "CMake configure timeout: $ConfigureTimeoutMinutes minute(s)"
            } else {
                Write-Info "CMake configure timeout: disabled"
            }
            Invoke-NativeCommand -FilePath "cmake" -Arguments $CMakeArgs -Description "CMake configuration" -TimeoutMinutes $ConfigureTimeoutMinutes

            # Build
            Write-Info "Building (this may take a few minutes)..."
            Invoke-NativeCommand -FilePath "cmake" -Arguments @("--build", ".", "--config", "Release", "--parallel") -Description "CMake build"

            # Install to local directory
            Write-Info "Installing to local directory..."
            Invoke-NativeCommand -FilePath "cmake" -Arguments @("--install", ".", "--config", "Release") -Description "CMake install"

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
        if ($LASTEXITCODE -ne 0) {
            throw "Patch application failed"
        }
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

    if ($ConfigureTimeoutMinutes -lt 0) {
        Write-Error "ConfigureTimeoutMinutes must be 0 or greater"
        exit 1
    }

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

    # Apply patches, unless explicitly disabled for upstream reference checks.
    if ($SkipPatches) {
        Write-Info "Skipping local patches; using upstream C++ implementation as-is"
    } else {
        Invoke-ApplyPatches
    }

    # Upstream main no longer always provides the generated LUT header in the
    # include directory. Recover it after patch application so the patch guard
    # can still enforce a clean external checkout.
    Ensure-KernelHeader

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
