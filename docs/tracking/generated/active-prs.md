<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m3-macbook-air | M3MBA-016 | #5043 | `codex/apple-m3-macbook-air/M3MBA-016-backend-visibility-preflight` | Add M3 Air Metal and MPSGraph backend visibility receipts plus a bounded Apple preflight contract that records host/runtime visibility, requested backend identity, machine identity, fallback status, and claim boundaries without loading models, downloading artifacts, or claiming full Metal/MPSGraph inference. |
| slm-cpu | SLM-CPU-018 | #5067 | `docs/slm-cpu-018-smollm2-normalization-policy-linear` | Record the governed SmolLM2 360M normalization-policy decision after the SLM-CPU-017 strict CPU preflight blocker. The item must preserve the generic llama strict LayerNorm/RMSNorm gamma guard as fail-closed, define the exact metadata required before any SmolLM2-family exception can be implemented, and commit a machine-readable policy audit. It must not change loader behavior, retry CPU answer sanity, start CUDA planning, or claim CPU answer readiness, throughput, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, broad dense GGUF support, or BitNet QK256 behavior. |
