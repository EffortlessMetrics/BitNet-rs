<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-BENCH-007 | #5869 | `codex/apple-m4-inference-excellence/M4-BENCH-007-harness-calibration` | Calibrate the M4 benchmark harness before using timing envelopes: record clock source and resolution, runner overhead, warm-up policy, sample discard policy, synthetic timing fixtures, profile timeout rules, and invalid-comparison reasons. |
| slm-cpu | SLM-CPU-048 | #5873 | `codex/slm-cpu-048-q8-sidecar-runtime-preflight` | Add the first non-executing packed Q8_0 sidecar runtime preflight after SLM-CPU-047. The slice must consume the sidecar equivalence gate and selector evidence to produce an explicit runtime eligibility report naming each remaining blocker before packed sidecar compute can be selected. It must keep selected_path=eager_f32_candle and sidecar_runtime_compute_allowed=false unless generated-ID/text receipt equivalence and a production compute hook both exist. It must not execute packed Q8_0 sidecar compute, claim speedup, alter generated IDs/text, change tokenizer or backend semantics, start Q4/Q5 runtime support, or touch server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 paths. |
