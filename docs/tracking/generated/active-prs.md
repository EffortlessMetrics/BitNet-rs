<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-001 | #3802 | `codex/intel-258v-platform/CPU258V-001-validation-harness` | Add a validation-only Core Ultra 7 258V CPU BitNet preflight command that emits structured blocked_preflight or preflight_ready artifacts without changing GGUF loader, tokenizer, QK256 layout, QK256 dispatch, CPU kernels, or transformer decode internals. |
| nvidia-5070ti | CUDA-BITNET-006 | #3801 | `codex/cuda-bitnet-006-one-token-proof` | Add strict one-token BitNet CUDA proof with official GGUF, real tokenizer, CUDA kernel invocations greater than zero, zero CPU fallback, CPU/CUDA greedy or top-1 agreement, fallback_used=false, and speedup_claim=false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
