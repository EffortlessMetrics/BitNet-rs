<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-EXCELLENCE-002 | #5066 | `codex/apple-m4-inference-excellence/M4-EXCELLENCE-002-bitnet-warm-refresh` | Run and commit a second BitNet variable warm-session refresh for the accepted artifact/tokenizer identity so the BitNet warm dashboard group can move from insufficient_history to comparable matching history while chat and serve remain disabled. |
| slm-cpu | SLM-CPU-018 | #5067 | `docs/slm-cpu-018-smollm2-normalization-policy-linear` | Record the governed SmolLM2 360M normalization-policy decision after the SLM-CPU-017 strict CPU preflight blocker. The item must preserve the generic llama strict LayerNorm/RMSNorm gamma guard as fail-closed, define the exact metadata required before any SmolLM2-family exception can be implemented, and commit a machine-readable policy audit. It must not change loader behavior, retry CPU answer sanity, start CUDA planning, or claim CPU answer readiness, throughput, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, broad dense GGUF support, or BitNet QK256 behavior. |
