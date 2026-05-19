<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-MODEL-015 | #5931 | `codex/cuda-model-015-qwen3-repeated-comparator-plan` | Define and queue the Qwen3 repeated same-artifact CPU/CUDA comparator baseline for one_token, short_decode_8, short_decode_32, warm_session_3_turns, and decode_128_from_warm_context on Windows 9950X3D + RTX 5070 Ti. Record that the existing 2026-05-15 Qwen3 benchmark review is insufficient for speed or benchmark qualification because it has runs_per_backend=1 and repeated_evidence=false. Preserve speedup_claim=false, benchmark_qualified=false, full_residency_claim=false, exact-profile-only server readiness, broad dense GGUF false, Qwen2.5 inheritance false, and BitNet QK256 proof false. |
