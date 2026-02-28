@echo off
cd /d E:\Code\Rust\BitNet-rs-worktrees\server-gpu-backend
cargo test -p bitnet-server --no-default-features --features cpu -- gpu_backend 2>&1 > E:\Code\Rust\BitNet-rs\test_output.txt
echo EXIT_CODE=%ERRORLEVEL% >> E:\Code\Rust\BitNet-rs\test_output.txt
