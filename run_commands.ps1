# Script to run the requested commands

Write-Host "=== Command 1: rustfmt ===" -ForegroundColor Cyan
$cmd1 = "rustfmt --edition 2024 crates\bitnet-kernels\tests\avx2_scalar_parity_tests.rs"
Write-Host "Running: $cmd1" -ForegroundColor Yellow
try {
    & rustfmt --edition 2024 'crates\bitnet-kernels\tests\avx2_scalar_parity_tests.rs' 2>&1
    Write-Host "✓ rustfmt completed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ rustfmt failed: $_" -ForegroundColor Red
}

Write-Host "`n=== Command 2: cargo test ===" -ForegroundColor Cyan
$cmd2 = "cargo test -p bitnet-kernels --test avx2_scalar_parity_tests --no-default-features --features cpu --no-run"
Write-Host "Running: $cmd2" -ForegroundColor Yellow
try {
    & cargo test -p bitnet-kernels --test avx2_scalar_parity_tests --no-default-features --features cpu --no-run 2>&1
    Write-Host "✓ cargo test completed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ cargo test failed: $_" -ForegroundColor Red
}
