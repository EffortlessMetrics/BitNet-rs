$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

Write-Host "=== Command 1: Set PATH ===" -ForegroundColor Green
Write-Host "PATH updated successfully" -ForegroundColor Green
Write-Host ""

Write-Host "=== Command 2: cargo build ===" -ForegroundColor Green
Write-Host "Running: cargo build -p bitnet-tokenizers --no-default-features --features cpu"
Write-Host ""
cargo build -p bitnet-tokenizers --no-default-features --features cpu 2>&1 | Select-Object -Last 10
Write-Host ""

Write-Host "=== Command 3: cargo test ===" -ForegroundColor Green
Write-Host "Running: cargo test -p bitnet-tokenizers --no-default-features --features cpu -- vocab_analyzer"
Write-Host ""
cargo test -p bitnet-tokenizers --no-default-features --features cpu -- vocab_analyzer 2>&1 | Select-Object -Last 25
