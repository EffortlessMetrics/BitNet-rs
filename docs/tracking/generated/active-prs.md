<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-PERF-002 | #4440 | `codex/cuda-dense-perf-002-repeated-comparator` | Add repeated same-artifact dense Qwen CPU/CUDA comparator receipts after the CUDA-DENSE-PERF-001 baseline, covering one_token, short_decode_8, and warm_session_3_turns profiles with at least three runs per backend where practical, the same verified qwen2.5-0.5b-instruct-q8_0 GGUF SHA, tokenizer authority, prompt template, deterministic generation policy, generated-token comparison or first-divergence report, fallback_used=false for both CPU and CUDA profiles, CPU and CUDA timing splits, CUDA kernel timing, transfer byte evidence and transfer-timing status, VRAM/power/thermal context where available, and speedup_claim=false plus benchmark_qualified_speedup=false pending a later profile-specific qualification review. |
