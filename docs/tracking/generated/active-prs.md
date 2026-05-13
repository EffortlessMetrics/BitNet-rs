<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU-BITNET-PERF-002 | #4609 | `codex/lunar-lake/CPU-BITNET-PERF-002-tiling-matrix` | Record a narrow 258V CPU QK256/I2_S tiling/thread candidate matrix receipt with sampled GEMV and GEMM timings, preserving fallback=false and speedup_claim=false while explicitly not applying thread-count scheduling yet and not claiming answer-quality, sustained throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness. |
| slm-cpu | SLM-CPU-008T | #4611 | `codex/slm-cpu-008t-first-token-parity` | Fix the post-008 Qwen3-0.6B Q8_0 first-token parity candidate by selecting the dedicated GGUF output.weight head before tied-embedding fallback, then require a real i5-8250U artifact refresh to prove whether bitnet-rs now matches the reference token 19 / '4' for the same model SHA, rendered prompt, prompt IDs, greedy settings, strict tokenizer, selected CPU backend, and fallback=false. This slice must not claim answer quality, tiny corpus, multi-token stability, warm-session performance, Q4/Q5 quant expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
