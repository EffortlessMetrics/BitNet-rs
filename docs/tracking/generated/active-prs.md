<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU-BITNET-PERF-003 | #4689 | `codex/lunar-lake/CPU-BITNET-PERF-003-applied-threads` | Record a narrow 258V CPU QK256/I2_S applied-thread microbench receipt with sampled GEMV row-parallel and GEMM token-parallel scoped worker timings, preserving fallback=false and speedup_claim=false while not changing the full BitNet decode path or claiming answer-quality, sustained throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness. |
| nvidia-5070ti | CUDA-DENSE-052 | #4695 | `cuda/dense-qwen25-short-decode-unblock` | Add an 8-32 token deterministic Qwen2.5 0.5B Q8_0 short-decode strict CUDA receipt with fallback_used=false, stable greedy token sequence, valid UTF-8 answer, no raw special-token garbage, and recorded prefill, KV, logits, sampler, kernel, and transfer evidence. The current-source rerun supersedes the stale-binary diagnostic blocker and records decoded text `The answer is 4. What is` with CPU/CUDA generated-token equality. |
