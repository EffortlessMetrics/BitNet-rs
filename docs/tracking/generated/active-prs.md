<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008 | #4434 | `codex/slm-cpu-008-qwen3-first-token-parity` | Fix the first Qwen3-0.6B Q8_0 divergent operation using the SLM-CPU-007B checkpoint evidence and any required known-good reference checkpoints so bitnet-rs matches the reference first generated token for the same model SHA, prompt/template/BOS policy, prompt IDs, greedy settings, and fallback=false. The target reference token is 19 / '4'. No answer-quality, throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, quant-expansion, or BitNet QK256 claim is allowed. |
