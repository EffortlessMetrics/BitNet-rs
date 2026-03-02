$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
Set-Location C:\Code\Rust\BitNet-rs-wt1
Write-Host "=== BUILD ==="
cargo build -p bitnet-tokenizers --no-default-features --features cpu 2>&1 | Select-Object -Last 5
Write-Host "=== TEST ==="
cargo test -p bitnet-tokenizers --no-default-features --features cpu -- vocab_analyzer 2>&1 | Select-Object -Last 25
Write-Host "=== FMT ==="
cargo fmt --all 2>&1
Write-Host "=== DONE ==="
