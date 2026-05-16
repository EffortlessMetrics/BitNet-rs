<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-EXCELLENCE-004 | #5089 | `codex/apple-m4-inference-excellence/M4-EXCELLENCE-004-drift-thresholds` | Publish model-family drift thresholds for M4 dense SLM and BitNet reports, separating identity mismatches, quality failures, timing warnings, memory drift warnings, timeout failures, and unsupported claim boundaries. |
| slm-cpu | SLM-CPU-019 | #5081 | `model/slm-cpu-019-smollm2-normalization-validation` | Implement exact metadata-scoped SmolLM2 360M normalization validation in the strict GGUF loader while preserving generic llama strict LayerNorm/RMSNorm gamma fail-closed behavior. The accepted SmolLM2 exception must require the pinned artifact SHA plus exact GGUF architecture, tokenizer pre-tokenizer, tensor dimensions, vocab size, block count, and intermediate size, and must be receipt-visible. This item must not retry CPU answer sanity, start CUDA planning, or claim CPU answer readiness, throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, broad dense GGUF support, or BitNet QK256 behavior. |
