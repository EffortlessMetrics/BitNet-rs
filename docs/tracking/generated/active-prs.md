<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-MODEL-014B | #5918 | `codex/cuda-model-014-qwen3-server-ready-accept` | Accept Qwen3 0.6B Q8_0 exact-profile server readiness only for the committed current-source non-streaming /v1/chat/completions RTX 5070 Ti shared-engine receipt. Promote server_ready=true in the model coverage row only for that exact profile while preserving speedup_claim=false, benchmark_qualified=false, full_residency_claim=false, broad dense GGUF readiness false, Qwen2.5 inheritance false, and BitNet QK256 proof false. |
