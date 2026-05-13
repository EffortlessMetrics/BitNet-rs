<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008S | #4606 | `codex/slm-cpu-008s-qwen3-output-head-root-cause` | Use the SLM-CPU-008R post-#4434 artifact and trace to localize the remaining Qwen3-0.6B Q8_0 first-token divergence, prioritizing output head/vocab indexing and shared transformer math. The item must identify the next concrete drift point or add the smallest missing diagnostic needed to do so, while preserving strict loader/tokenizer provenance and fallback=false. No answer-quality, tiny corpus, sustained throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, Q4/Q5 quant expansion, or BitNet QK256 claim is allowed. |
| intel-258v-platform | CPU-BITNET-PERF-001 | #4607 | `codex/lunar-lake/CPU-BITNET-PERF-001-i2s-microbench` | Record a narrow 258V CPU QK256/I2_S GEMV and GEMM microbench receipt using the existing BitNet benchmark receipt path, preserving fallback=false and speedup_claim=false without answer-quality, sustained decode throughput, Arc/NPU execution, acceleration, QK256 semantic changes, or full model correctness claims. |
