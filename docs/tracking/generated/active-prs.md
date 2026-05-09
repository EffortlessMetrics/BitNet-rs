<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-010 | #4207 | `codex/cuda-dense-010-live-linear-receipt` | Record a live RTX 5070 Ti dense GGUF single-linear CUDA parity receipt for the verified Qwen2.5 0.5B Q8_0 artifact using the existing dense-gguf-linear-parity harness, preserving dense_regular_llm_cuda routing, fallback_used=false, BitNet packed QK256 proof false, dense GGUF inference false, speedup_claim=false, and full_cuda_residency_claimed=false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
