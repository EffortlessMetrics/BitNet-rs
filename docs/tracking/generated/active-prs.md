<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-PROFILE-RUN-003 | #5874 | `codex/lunar-lake/LNL258V-PROFILE-RUN-003-cpu-baseline-input` | Add optional Rust GGUF CPU profile-run input to bitnet lunar-lake profile-compare so future fallback-free, profile-specific CPU baseline receipts for prefill_heavy/decode_heavy can replace bounded math-ask timing for the dense CPU default route; verify matching CPU profile cases satisfy route timing applicability while preserving current route decisions when the receipt is absent; make no route promotion, new inference, speedup/power claim, OpenVINO GPU/NPU promotion, or BitNet QK256/I2_S behavior change. |
