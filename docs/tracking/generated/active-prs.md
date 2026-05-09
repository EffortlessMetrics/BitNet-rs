<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-011 | #4209 | `codex/cuda-dense-011-layer0-linear-sweep` | Record live RTX 5070 Ti dense GGUF linear role-sweep receipts for the verified Qwen2.5 0.5B Q8_0 artifact using the existing dense-gguf-linear-parity harness, covering attention_q, attention_k, attention_v, attention_output, mlp_gate, mlp_up, mlp_down, and output while preserving dense_regular_llm_cuda routing, fallback_used=false, BitNet packed QK256 proof false, dense GGUF inference false, speedup_claim=false, and full_cuda_residency_claimed=false. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
