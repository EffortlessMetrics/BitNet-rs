# Comprehensive code quality validation for BitNet-rs
# This script runs all quality checks required for crates.io publication

param(
    [switch]$SkipTests = $false,
    [switch]$SkipBench = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🔍 BitNet-rs Code Quality Validation" -ForegroundColor Blue
Write-Host "====================================" -ForegroundColor Blue

# Function to print status
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[PASS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

# Check if we're in the right directory
if (-not (Test-Path "Cargo.toml") -or -not (Test-Path "crates")) {
    Write-Error "This script must be run from the BitNet-rs root directory"
    exit 1
}

# Check Rust version
Write-Status "Checking Rust version..."
try {
    $rustVersion = (rustc --version).Split(' ')[1]
    $requiredVersion = "1.89.0"
    if ([version]$rustVersion -lt [version]$requiredVersion) {
        Write-Error "Rust version $rustVersion is below required $requiredVersion"
        exit 1
    }
    Write-Success "Rust version $rustVersion meets requirements"
} catch {
    Write-Error "Failed to check Rust version: $_"
    exit 1
}

# Check for required tools
Write-Status "Checking required tools..."
$requiredTools = @("cargo")
foreach ($tool in $requiredTools) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "$tool is not available in PATH"
        exit 1
    }
}
Write-Success "All required tools are available"

# Format check
Write-Status "Checking code formatting..."
try {
    $formatResult = cargo fmt --all -- --check 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Code is not properly formatted"
        Write-Host $formatResult
        Write-Host "Run: cargo fmt --all"
        exit 1
    }
    Write-Success "Code formatting is correct"
} catch {
    Write-Error "Failed to check formatting: $_"
    exit 1
}

# Clippy check with pedantic lints
Write-Status "Running Clippy with pedantic lints..."
try {
    cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Clippy found issues"
        exit 1
    }
    Write-Success "Clippy checks passed"
} catch {
    Write-Error "Failed to run Clippy: $_"
    exit 1
}

# Build check
Write-Status "Building all crates..."
try {
    cargo build --workspace --all-features
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed"
        exit 1
    }
    Write-Success "Build successful"
} catch {
    Write-Error "Failed to build: $_"
    exit 1
}

# Test check
if (-not $SkipTests) {
    Write-Status "Running tests..."
    try {
        cargo test --workspace --all-features
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Tests failed"
            exit 1
        }
        Write-Success "All tests passed"
    } catch {
        Write-Error "Failed to run tests: $_"
        exit 1
    }
} else {
    Write-Warning "Skipping tests as requested"
}

# Documentation check
Write-Status "Checking documentation..."
try {
    cargo doc --workspace --all-features --no-deps
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Documentation generation failed"
        exit 1
    }
    Write-Success "Documentation generated successfully"
} catch {
    Write-Error "Failed to generate documentation: $_"
    exit 1
}

# Security audit (if cargo-audit is available)
Write-Status "Running security audit..."
try {
    if (Get-Command cargo-audit -ErrorAction SilentlyContinue) {
        cargo audit
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Security audit found vulnerabilities"
            exit 1
        }
        Write-Success "Security audit passed"
    } else {
        Write-Warning "cargo-audit not found, skipping security audit"
        Write-Host "Install with: cargo install cargo-audit"
    }
} catch {
    Write-Warning "Failed to run security audit: $_"
}

# License compliance check (if cargo-deny is available)
Write-Status "Checking license compliance..."
try {
    if (Get-Command cargo-deny -ErrorAction SilentlyContinue) {
        cargo deny check
        if ($LASTEXITCODE -ne 0) {
            Write-Error "License compliance check failed"
            exit 1
        }
        Write-Success "License compliance verified"
    } else {
        Write-Warning "cargo-deny not found, skipping license compliance check"
        Write-Host "Install with: cargo install cargo-deny"
    }
} catch {
    Write-Warning "Failed to run license compliance check: $_"
}

# Check for TODO/FIXME comments in release-critical code
Write-Status "Checking for TODO/FIXME comments..."
try {
    $todoFiles = Get-ChildItem -Path "src", "crates" -Recurse -Filter "*.rs" |
                 Where-Object { (Get-Content $_.FullName -Raw) -match "TODO|FIXME" }

    if ($todoFiles.Count -gt 0) {
        Write-Warning "Found $($todoFiles.Count) files with TODO/FIXME comments"
        foreach ($file in $todoFiles) {
            $lines = Get-Content $file.FullName | Select-String "TODO|FIXME" -AllMatches
            foreach ($line in $lines) {
                Write-Host "$($file.FullName):$($line.LineNumber): $($line.Line.Trim())" -ForegroundColor Yellow
            }
        }
        Write-Host "Consider resolving these before release"
    }
} catch {
    Write-Warning "Failed to check for TODO/FIXME comments: $_"
}

# Check crate metadata completeness
Write-Status "Validating crate metadata..."
$requiredFields = @("description", "license", "repository", "homepage", "keywords", "categories")
$crateDirectories = @(".") + (Get-ChildItem -Path "crates" -Directory | ForEach-Object { $_.FullName })

foreach ($crateDir in $crateDirectories) {
    $cargoToml = Join-Path $crateDir "Cargo.toml"
    if (Test-Path $cargoToml) {
        $crateName = Split-Path $crateDir -Leaf
        Write-Status "Checking metadata for $crateName..."

        $content = Get-Content $cargoToml -Raw
        foreach ($field in $requiredFields) {
            if (-not ($content -match "^$field\s*=" -or $content -match "^$field\.workspace\s*=")) {
                Write-Error "Missing $field in $cargoToml"
                exit 1
            }
        }
    }
}
Write-Success "All crate metadata is complete"

# Check for proper feature flags
Write-Status "Validating feature flags..."
$mainCargoToml = Get-Content "Cargo.toml" -Raw
if (-not ($mainCargoToml -match "\[features\]")) {
    Write-Error "Main crate is missing [features] section"
    exit 1
}
Write-Success "Feature flags are properly configured"

# Check README exists and is not empty
Write-Status "Checking README..."
if (-not (Test-Path "README.md") -or (Get-Item "README.md").Length -eq 0) {
    Write-Error "README.md is missing or empty"
    exit 1
}
Write-Success "README.md exists and is not empty"

# Check CHANGELOG exists
Write-Status "Checking CHANGELOG..."
if (-not (Test-Path "CHANGELOG.md")) {
    Write-Error "CHANGELOG.md is missing"
    exit 1
}
Write-Success "CHANGELOG.md exists"

# Check examples compile
Write-Status "Checking examples..."
try {
    cargo check --examples --all-features
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Examples failed to compile"
        exit 1
    }
    Write-Success "All examples compile successfully"
} catch {
    Write-Error "Failed to check examples: $_"
    exit 1
}

# Performance regression check (if benchmarks exist)
if ((Test-Path "benches") -and (-not $SkipBench)) {
    Write-Status "Running benchmark compilation check..."
    try {
        cargo bench --no-run --workspace
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Benchmarks failed to compile"
            exit 1
        }
        Write-Success "Benchmarks compile successfully"
    } catch {
        Write-Error "Failed to compile benchmarks: $_"
        exit 1
    }
} elseif ($SkipBench) {
    Write-Warning "Skipping benchmark check as requested"
}

# Final summary
Write-Host ""
Write-Host "🎉 All quality checks passed!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Success "Code formatting: ✓"
Write-Success "Clippy lints: ✓"
Write-Success "Build: ✓"
if (-not $SkipTests) { Write-Success "Tests: ✓" }
Write-Success "Documentation: ✓"
Write-Success "Crate metadata: ✓"
Write-Success "Feature flags: ✓"
Write-Success "README/CHANGELOG: ✓"
Write-Success "Examples: ✓"

Write-Host ""
Write-Host "🚀 BitNet-rs is ready for crates.io publication!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Update version numbers if needed"
Write-Host "2. Update CHANGELOG.md with release notes"
Write-Host "3. Create a git tag for the release"
Write-Host "4. Run 'cargo publish --dry-run' to verify"
Write-Host "5. Run 'cargo publish' to publish to crates.io"
