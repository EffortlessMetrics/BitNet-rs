# Fetch and build the external BitNet C++ implementation
# PowerShell version for Windows systems

param(
    [string]$Tag = $(if ($env:BITNET_CPP_TAG) { $env:BITNET_CPP_TAG } else { "v1.0.0" }),
    [string]$CachePath = $(if ($env:BITNET_CPP_PATH) { $env:BITNET_CPP_PATH } else { "$env:USERPROFILE\.cache\bitnet_cpp" }),
    [string]$ModelDir = $(if ($env:BITNET_CPP_MODEL_DIR) { $env:BITNET_CPP_MODEL_DIR } else { "models\BitNet-b1.58-2B-4T" }),
    [string]$QuantType = $(if ($env:BITNET_CPP_QUANT_TYPE) { $env:BITNET_CPP_QUANT_TYPE } else { "i2_s" }),
    [switch]$Force,
    [switch]$Clean,
    [switch]$SkipPatches,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$BitNetCppRepo = "https://github.com/microsoft/BitNet.git"

if (-not [System.IO.Path]::IsPathRooted($CachePath)) {
    $CachePath = Join-Path (Get-Location) $CachePath
}
$CachePath = [System.IO.Path]::GetFullPath($CachePath)
if (-not [System.IO.Path]::IsPathRooted($ModelDir)) {
    $ModelDir = Join-Path (Get-Location) $ModelDir
}
$ModelDir = [System.IO.Path]::GetFullPath($ModelDir)
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

function Get-VisualStudioBuildToolsPath {
    if (-not (Test-IsWindows)) {
        return $null
    }

    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        return $null
    }

    $VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($VsPath) {
        return $VsPath.Trim()
    }
    return $null
}

function Find-WindowsClangTool {
    param([Parameter(Mandatory = $true)][string]$Name)

    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    $Executable = if ($Name.EndsWith(".exe")) { $Name } else { "$Name.exe" }
    $Candidates = @()

    if ($env:ProgramFiles) {
        $Candidates += (Join-Path $env:ProgramFiles "LLVM\bin\$Executable")
    }

    $VsPath = Get-VisualStudioBuildToolsPath
    if ($VsPath) {
        $Candidates += (Join-Path $VsPath "VC\Tools\Llvm\x64\bin\$Executable")
        $Candidates += (Join-Path $VsPath "VC\Tools\Llvm\bin\$Executable")
    }

    if (${env:ProgramFiles(x86)}) {
        $BuildTools = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools"
        $Candidates += (Join-Path $BuildTools "VC\Tools\Llvm\x64\bin\$Executable")
        $Candidates += (Join-Path $BuildTools "VC\Tools\Llvm\bin\$Executable")
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }

    return $null
}

function Add-ToolDirectoryToPath {
    param([string]$ToolPath)

    if (-not $ToolPath) {
        return
    }

    $ToolDir = Split-Path $ToolPath -Parent
    $PathParts = $env:PATH -split ';'
    if ($PathParts -notcontains $ToolDir) {
        $env:PATH = "$ToolDir;$env:PATH"
    }
}

function Quote-CmdArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Join-CmdArguments {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    return (($Arguments | ForEach-Object { Quote-CmdArgument $_ }) -join " ")
}

function Update-Submodules {
    Write-Info "Updating BitNet C++ submodules..."
    git submodule update --init --recursive --force

    if ($LASTEXITCODE -ne 0) {
        throw "Submodule update failed"
    }

    git submodule foreach --recursive git reset --hard

    if ($LASTEXITCODE -ne 0) {
        throw "Submodule reset failed"
    }
}

function Invoke-GenerateReferenceKernels {
    if (-not (Test-IsWindows)) {
        return
    }

    if ($QuantType -ne "i2_s") {
        throw "Reference helper only has a Windows codegen rule for i2_s right now; got QuantType=$QuantType"
    }

    $ModelName = Split-Path $ModelDir -Leaf
    $CodegenArgs = switch ($ModelName) {
        "BitNet-b1.58-2B-4T" {
            @("utils\codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32")
        }
        "bitnet_b1_58-3B" {
            @("utils\codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32")
        }
        "bitnet_b1_58-large" {
            @("utils\codegen_tl2.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "96,192,96", "--bm", "32,32,32")
        }
        default {
            throw "No Windows reference kernel codegen rule for model directory '$ModelName'"
        }
    }

    Push-Location $CachePath
    try {
        Write-Info "Generating BitNet reference kernels for $ModelName ($QuantType)..."
        python @CodegenArgs
        if ($LASTEXITCODE -ne 0) {
            throw "BitNet reference kernel codegen failed"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-WindowsReferenceCompatibilityFixes {
    if (-not (Test-IsWindows)) {
        return
    }

    $MadPath = Join-Path $CachePath "src\ggml-bitnet-mad.cpp"
    if (-not (Test-Path $MadPath)) {
        throw "Reference source missing: $MadPath"
    }

    $Before = "        int8_t * y_col = y + col * by;"
    $After = "        const int8_t * y_col = y + col * by;"
    $Content = Get-Content -LiteralPath $MadPath -Raw
    if ($Content.Contains($Before)) {
        Write-Info "Applying Windows reference const-compatibility fix..."
        $Content = $Content.Replace($Before, $After)
        Set-Content -LiteralPath $MadPath -Value $Content -Encoding UTF8
    } elseif (-not $Content.Contains($After)) {
        throw "Expected Windows reference const-compatibility patch target was not found"
    }

    $CommonPath = Join-Path $CachePath "3rdparty\llama.cpp\common\common.cpp"
    if (-not (Test-Path $CommonPath)) {
        throw "Reference common source missing: $CommonPath"
    }
    $CommonContent = Get-Content -LiteralPath $CommonPath -Raw
    if (-not $CommonContent.Contains("#include <chrono>")) {
        Write-Info "Applying Windows reference chrono include fix..."
        $CommonNeedle = "#include <ctime>"
        if (-not $CommonContent.Contains($CommonNeedle)) {
            throw "Expected Windows reference chrono include insertion point was not found"
        }
        $CommonContent = $CommonContent.Replace($CommonNeedle, "$CommonNeedle`r`n#include <chrono>")
        Set-Content -LiteralPath $CommonPath -Value $CommonContent -Encoding UTF8
    }

    $LogPath = Join-Path $CachePath "3rdparty\llama.cpp\common\log.cpp"
    if (-not (Test-Path $LogPath)) {
        throw "Reference log source missing: $LogPath"
    }
    $LogContent = Get-Content -LiteralPath $LogPath -Raw
    if (-not $LogContent.Contains("#include <chrono>")) {
        Write-Info "Applying Windows reference log chrono include fix..."
        $LogNeedle = "#include <condition_variable>"
        if (-not $LogContent.Contains($LogNeedle)) {
            throw "Expected Windows reference log chrono include insertion point was not found"
        }
        $LogContent = $LogContent.Replace($LogNeedle, "$LogNeedle`r`n#include <chrono>")
        Set-Content -LiteralPath $LogPath -Value $LogContent -Encoding UTF8
    }

    $ChronoFixes = @(
        @{
            RelativePath = "3rdparty\llama.cpp\examples\imatrix\imatrix.cpp"
            Needle = "#include <ctime>"
        },
        @{
            RelativePath = "3rdparty\llama.cpp\examples\perplexity\perplexity.cpp"
            Needle = "#include <ctime>"
        },
        @{
            RelativePath = "3rdparty\llama.cpp\examples\server\httplib.h"
            Needle = "#include <condition_variable>"
        }
    )

    foreach ($Fix in $ChronoFixes) {
        $SourcePath = Join-Path $CachePath $Fix.RelativePath
        if (-not (Test-Path $SourcePath)) {
            throw "Reference source missing: $SourcePath"
        }
        $SourceContent = Get-Content -LiteralPath $SourcePath -Raw
        if (-not $SourceContent.Contains("#include <chrono>")) {
            Write-Info "Applying Windows reference chrono include fix to $($Fix.RelativePath)..."
            if (-not $SourceContent.Contains($Fix.Needle)) {
                throw "Expected Windows reference chrono include insertion point was not found in $($Fix.RelativePath)"
            }
            $SourceContent = $SourceContent.Replace($Fix.Needle, "$($Fix.Needle)`r`n#include <chrono>")
            Set-Content -LiteralPath $SourcePath -Value $SourceContent -Encoding UTF8
        }
    }
}

function Show-Usage {
    @"
Usage: .\fetch_bitnet_cpp.ps1 [OPTIONS]

Fetch and build the external BitNet C++ implementation for cross-validation.

OPTIONS:
    -Tag TAG            Specify BitNet.cpp tag/version (default: $Tag)
    -CachePath PATH     Specify cache directory (default: $CachePath)
    -ModelDir PATH      Local model directory for reference kernel setup (default: $ModelDir)
    -QuantType TYPE     Quantization type for reference kernel setup (default: $QuantType)
    -Force              Force rebuild even if already built
    -Clean              Clean build directory before building
    -SkipPatches        Use upstream C++ source as-is without applying local patches
    -Help               Show this help message

ENVIRONMENT VARIABLES:
    BITNET_CPP_TAG      Override default tag/version
    BITNET_CPP_PATH     Override default cache directory
    BITNET_CPP_MODEL_DIR Override default model directory
    BITNET_CPP_QUANT_TYPE Override default quantization type

EXAMPLES:
    .\fetch_bitnet_cpp.ps1                      # Use defaults
    .\fetch_bitnet_cpp.ps1 -Tag v1.1.0         # Use specific version
    .\fetch_bitnet_cpp.ps1 -Force              # Force rebuild
    .\fetch_bitnet_cpp.ps1 -Clean -Force       # Clean rebuild
    .\fetch_bitnet_cpp.ps1 -SkipPatches        # Build upstream source without patches

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
        # Upstream BitNet.cpp uses the Windows ClangCL toolset.
        $script:BitNetReferenceClang = Find-WindowsClangTool "clang"
        $script:BitNetReferenceClangxx = Find-WindowsClangTool "clang++"
        Add-ToolDirectoryToPath $script:BitNetReferenceClang
        Add-ToolDirectoryToPath $script:BitNetReferenceClangxx

        if (-not $script:BitNetReferenceClang) {
            $MissingDeps += "clang"
        }

        if (-not $script:BitNetReferenceClangxx) {
            $MissingDeps += "clang++"
        }

        # Check for Visual Studio or Build Tools.
        if (-not (Get-VisualStudioBuildToolsPath)) {
            $MissingDeps += "Visual Studio Build Tools"
        }

        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            $MissingDeps += "python"
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

            $OldErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $CurrentTag = git describe --tags --exact-match 2>$null
            $DescribeExitCode = $LASTEXITCODE
            $ErrorActionPreference = $OldErrorActionPreference

            $CurrentBranch = git rev-parse --abbrev-ref HEAD

            # Clean any local changes before moving refs.
            git reset --hard
            git clean -fd

            if ($DescribeExitCode -eq 0 -and $CurrentTag -eq $Tag) {
                Write-Info "Already on correct tag: $Tag"
                Update-Submodules
                return
            }

            if ($CurrentBranch -eq $Tag) {
                Write-Info "Already on branch $Tag; fast-forwarding to origin/$Tag"
                git reset --hard "origin/$Tag"
                Update-Submodules
                return
            }

            # Checkout the specified tag or branch.
            git checkout $Tag
            Update-Submodules
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
        Push-Location $CachePath
        try {
            Update-Submodules
        }
        finally {
            Pop-Location
        }
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
            $VsPath = Get-VisualStudioBuildToolsPath

            if (-not $VsPath) {
                throw "Visual Studio with C++ tools not found"
            }
            $ClangPath = if ($script:BitNetReferenceClang) { $script:BitNetReferenceClang } else { Find-WindowsClangTool "clang" }
            $ClangxxPath = if ($script:BitNetReferenceClangxx) { $script:BitNetReferenceClangxx } else { Find-WindowsClangTool "clang++" }
            Add-ToolDirectoryToPath $ClangPath
            Add-ToolDirectoryToPath $ClangxxPath

            if (Test-IsWindows) {
                $VsDevCmd = Join-Path $VsPath "Common7\Tools\VsDevCmd.bat"
                if (-not (Test-Path $VsDevCmd)) {
                    throw "Visual Studio developer command prompt not found: $VsDevCmd"
                }

                $ConfigureArgs = @(
                    "..",
                    "-G",
                    "NMake Makefiles",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_SYSTEM_PROCESSOR=AMD64",
                    "-DCMAKE_C_FLAGS=-mavx2 -mfma -mf16c",
                    "-DCMAKE_CXX_FLAGS=-mavx2 -mfma -mf16c",
                    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DCMAKE_INSTALL_PREFIX=$BuildDir\install",
                    "-DCMAKE_C_COMPILER=$ClangPath",
                    "-DCMAKE_CXX_COMPILER=$ClangxxPath"
                )

                $BuildCmd = Join-Path $BuildDir "build_bitnet_reference.cmd"
                $ConfigureLine = "cmake " + (Join-CmdArguments $ConfigureArgs)
                $Lines = @(
                    "@echo off",
                    "call $(Quote-CmdArgument $VsDevCmd) -arch=x64 -host_arch=x64 || exit /b 1",
                    "$ConfigureLine || exit /b 1",
                    "cmake --build . --config Release || exit /b 1"
                )
                Set-Content -LiteralPath $BuildCmd -Value $Lines -Encoding ASCII

                Write-Info "Configuring and building with Visual Studio developer environment..."
                & cmd.exe /d /s /c (Quote-CmdArgument $BuildCmd)

                if ($LASTEXITCODE -ne 0) {
                    throw "CMake build failed"
                }

                $ReferenceExe = Join-Path $BuildDir "bin\llama-cli.exe"
                if (-not (Test-Path $ReferenceExe)) {
                    throw "CMake build did not produce reference executable: $ReferenceExe"
                }
                Write-Info "Reference executable built: $ReferenceExe"
            } else {
                $CMakeArgs = @(
                    "..",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                    "-DBUILD_SHARED_LIBS=ON",
                    "-DCMAKE_INSTALL_PREFIX=$BuildDir\install"
                )

                # Configure with CMake
                Write-Info "Configuring build with CMake..."
                cmake @CMakeArgs

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
        if ($LASTEXITCODE -ne 0) {
            throw "Patch application failed"
        }
    } else {
        Write-Info "No patch application script found - using C++ implementation as-is"
    }
}

function Test-Build {
    Write-Info "Validating build..."

    if (Test-IsWindows) {
        $BinDir = Join-Path $BuildDir "bin"
        $ReferenceExe = Join-Path $BinDir "llama-cli.exe"
        $LlamaDll = Join-Path $BinDir "llama.dll"
        $GgmlDll = Join-Path $BinDir "ggml.dll"

        if (-not (Test-Path $ReferenceExe)) {
            Write-Error "Reference executable not found: $ReferenceExe"
            return $false
        }
        if (-not (Test-Path $LlamaDll)) {
            Write-Error "Reference runtime DLL not found: $LlamaDll"
            return $false
        }
        if (-not (Test-Path $GgmlDll)) {
            Write-Error "Reference runtime DLL not found: $GgmlDll"
            return $false
        }

        Write-Info "Found Windows reference executable and runtime DLLs"
        Write-Info "Build validation passed"
        return $true
    }

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

    if (Test-IsWindows) {
        $BinDir = Join-Path $BuildDir "bin"
        $ReferenceExe = Join-Path $BinDir "llama-cli.exe"
        $EnvContent = @"
# Environment setup for BitNet C++ cross-validation
# Run this script to set up environment variables

`$env:BITNET_CPP_PATH = "$CachePath"
`$env:BITNET_CPP_BIN_PATH = "$BinDir"
`$env:BITNET_CPP_REFERENCE_EXE = "$ReferenceExe"

# Add to PATH for DLLs
`$env:PATH = "`$env:BITNET_CPP_BIN_PATH;`$env:PATH"

Write-Host "BitNet C++ environment configured:" -ForegroundColor Green
Write-Host "  Path: `$env:BITNET_CPP_PATH" -ForegroundColor Green
Write-Host "  Binary directory: `$env:BITNET_CPP_BIN_PATH" -ForegroundColor Green
Write-Host "  Reference executable: `$env:BITNET_CPP_REFERENCE_EXE" -ForegroundColor Green
"@

        Set-Content -Path $EnvScript -Value $EnvContent -Encoding UTF8
        return
    }

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

    # Apply optional local patches, unless explicitly disabled for upstream reference checks.
    if ($SkipPatches) {
        Write-Info "Skipping optional local patches; only setup/toolchain compatibility fixes may be applied"
    } else {
        Invoke-ApplyPatches
    }

    # Upstream BitNet.cpp requires generated LUT headers before CMake configure.
    Invoke-GenerateReferenceKernels
    Invoke-WindowsReferenceCompatibilityFixes

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
