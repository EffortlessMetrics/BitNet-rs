<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-047 | #5868 | `codex/slm-cpu-047-q8-sidecar-equivalence-gate` | Add the behavior-preserving packed Q8_0 sidecar equivalence gate after SLM-CPU-046. The gate must connect the fixture-level Q8_0 sidecar prototype to the dense-linear selector by recording fixture-vs-eager-F32 parity, selector identity, remaining generated-ID/text receipt blockers, and sidecar_runtime_compute_allowed=false. It must preserve the existing eager F32 Candle runtime as the selected production path and must not execute packed Q8_0 sidecar compute, claim speedup, alter generated IDs/text, change tokenizer or backend semantics, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
