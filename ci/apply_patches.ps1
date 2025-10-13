# Apply patches to the external BitNet C++ implementation
# PowerShell version for Windows systems

param(
    [string]$CppPath = "$env:USERPROFILE\.cache\bitnet_cpp",
    [string]$PatchesDir = "$PSScriptRoot\..\patches"
)

$ErrorActionPreference = "Stop"

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

# Check if C++ implementation exists
if (-not (Test-Path $CppPath)) {
    Write-Error "BitNet C++ implementation not found at: $CppPath"
    Write-Error "Run ci/fetch_bitnet_cpp.ps1 first"
    exit 1
}

# Check if patches directory exists
if (-not (Test-Path $PatchesDir)) {
    Write-Info "No patches directory found - nothing to apply"
    exit 0
}

# Count patches
$PatchFiles = Get-ChildItem -Path $PatchesDir -Filter "*.patch" -File
$PatchCount = $PatchFiles.Count

if ($PatchCount -eq 0) {
    Write-Info "No patches found - C++ implementation will be used as-is"
    exit 0
}

Write-Warn "Found $PatchCount patches to apply"
Write-Warn "Remember: patches should be avoided when possible"
Write-Warn "See patches/README.md for patch policy"

# Change to C++ directory
Push-Location $CppPath

try {
    # Check if we're in a git repository
    if (-not (Test-Path ".git")) {
        Write-Error "C++ implementation is not a git repository"
        Write-Error "Cannot apply patches safely"
        exit 1
    }

    # Check for uncommitted changes
    $GitStatus = git status --porcelain
    if ($GitStatus) {
        Write-Error "C++ implementation has uncommitted changes"
        Write-Error "Cannot apply patches safely"
        exit 1
    }

    # Apply patches in order
    $AppliedCount = 0
    $FailedCount = 0

    foreach ($PatchFile in ($PatchFiles | Sort-Object Name)) {
        $PatchName = $PatchFile.Name
        Write-Info "Applying patch: $PatchName"

        # Check if patch has upstream issue reference
        $PatchContent = Get-Content $PatchFile.FullName -Raw
        if (-not ($PatchContent -match "issue|Issue")) {
            Write-Error "Patch $PatchName does not reference an upstream issue"
            Write-Error "All patches must reference upstream issues (see patches/README.md)"
            $FailedCount++
            continue
        }

        # Try to apply the patch
        $CheckResult = git apply --check $PatchFile.FullName 2>&1
        if ($LASTEXITCODE -eq 0) {
            git apply $PatchFile.FullName
            if ($LASTEXITCODE -eq 0) {
                Write-Info "Successfully applied: $PatchName"
                $AppliedCount++
            } else {
                Write-Error "Failed to apply patch: $PatchName"
                $FailedCount++
            }
        } else {
            Write-Error "Failed to apply patch: $PatchName"
            Write-Error "Patch may be outdated or conflict with current C++ version"
            Write-Host "Patch application error:" -ForegroundColor Red
            Write-Host $CheckResult -ForegroundColor Red
            $FailedCount++
        }
    }

    # Summary
    Write-Host ""
    Write-Info "Patch application summary:"
    Write-Info "  Applied: $AppliedCount"
    if ($FailedCount -gt 0) {
        Write-Error "  Failed: $FailedCount"
    } else {
        Write-Info "  Failed: $FailedCount"
    }

    # Exit with error if any patches failed
    if ($FailedCount -gt 0) {
        Write-Error "Some patches failed to apply"
        Write-Error "Check patch compatibility with current C++ version"
        exit 1
    }

    if ($AppliedCount -gt 0) {
        Write-Warn "Applied $AppliedCount patches to C++ implementation"
        Write-Warn "Consider contributing these changes upstream"
    }

    Write-Info "Patch application completed successfully"
}
finally {
    Pop-Location
}
