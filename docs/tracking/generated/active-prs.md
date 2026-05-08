<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-004 | #4121 | `codex/cuda-dense-004-persistent-gemm` | Add persistent fixture-session residency for the dense regular-LLM CUDA FP16 GEMM path, proving one CUDA context/module, upload-once input device buffers, one output device buffer, repeated launches, per-run host-to-device bytes equal zero, fallback_used=false, and no BitNet packed I2S/QK256, speedup, dense GGUF inference, or full CUDA residency claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
