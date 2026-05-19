<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-BENCH-007 | #5869 | `codex/apple-m4-inference-excellence/M4-BENCH-007-harness-calibration` | Calibrate the M4 benchmark harness before using timing envelopes: record clock source and resolution, runner overhead, warm-up policy, sample discard policy, synthetic timing fixtures, profile timeout rules, and invalid-comparison reasons. |
| intel-258v-platform | LNL258V-PROFILE-RUN-003 | #5874 | `codex/lunar-lake/LNL258V-PROFILE-RUN-003-cpu-baseline-input` | Add optional Rust GGUF CPU profile-run input to bitnet lunar-lake profile-compare so future fallback-free, profile-specific CPU baseline receipts for prefill_heavy/decode_heavy can replace bounded math-ask timing for the dense CPU default route; verify matching CPU profile cases satisfy route timing applicability while preserving current route decisions when the receipt is absent; make no route promotion, new inference, speedup/power claim, OpenVINO GPU/NPU promotion, or BitNet QK256/I2_S behavior change. |
