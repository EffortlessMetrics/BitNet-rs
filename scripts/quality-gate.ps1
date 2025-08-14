# PowerShell quality gate - runs all local checks before committing
$ErrorActionPreference = "Stop"

Write-Host "🔍 Running BitNet-rs quality gate..." -ForegroundColor Cyan
Write-Host ""

Write-Host "📝 Formatting code..." -ForegroundColor Yellow
cargo fmt --all
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "🔎 Running clippy with all features..." -ForegroundColor Yellow
$env:RUSTFLAGS = "-Dwarnings"
cargo clippy --workspace --all-features --all-targets -- -D warnings -D clippy::ptr_arg
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "✓ Checking tests compile (CPU only)..." -ForegroundColor Yellow
cargo check --workspace --tests --no-default-features --features cpu
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "🔒 Running dependency security audit..." -ForegroundColor Yellow
cargo deny check --hide-inclusion-graph
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "🚫 Checking for banned patterns..." -ForegroundColor Yellow
& "$PSScriptRoot\hooks\banned-patterns.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "✅ All quality checks passed!" -ForegroundColor Green