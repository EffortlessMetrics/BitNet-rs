<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| ci-coverage | CI-COVERAGE-002 | TBD | `codex/coverage-container-cleanup-3394` | Run coverage as a rust-ci job container and remove hosted-runner disk cleanup and nested Docker execution from the coverage workflow. |
| slm-cpu | SLM-CPU-042 | #5773 | `codex/slm-cpu-042-q8-dense-linear-locality` | Start the next Kaby Qwen3 Q8_0 dense-math performance slice after SLM-CPU-041 proved reusable FeedForward::down_proj output storage is blocked by Candle Tensor matmul/bias-add owned returns. The slice must identify and instrument the first behavior-preserving Q8_0 dense linear locality or matmul/dequant boundary that can be optimized without changing generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, or fallback=false; if no safe runtime change is made, it must produce a concrete implementation boundary for the next slice. It must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
