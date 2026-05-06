<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4 | M4-012 | #3775 | `codex/apple-m4/M4-012-tl1-arm-investigation` | Investigate TL1 as an Apple CPU/NEON-oriented BitNet path and document any Metal conversion boundaries honestly. |
| cpu-proof | CPU-BITNET-005c | #3753 | `codex/cpu-bitnet-005c-avx2-parity-hardening` | AVX2 decode GEMV parity is hardened against scalar across rows, tail-column shapes, deterministic patterns, and repeated-run equality. |
| intel-258v-platform | ARC140V-002 | #3727 | `codex/intel-arc/ARC140V-002-runtime-probe` | Probe exact Arc 140V runtime visibility by name or PCI ID 0x64A0 across OpenCL, Level Zero, and OpenVINO GPU.0 while recording proof_stage=runtime_detected, requested/selected backend identity, runtime API, and fallback_used=false. |
| intel-npu | NPU-003 | #3739 | `codex/intel-npu/NPU-003-openvino-runtime-probe` | Add Intel NPU runtime detection fields that keep OS accelerator evidence separate from OpenVINO NPU visibility and record OpenVINO NPU full name, driver/compiler/memory properties, runtime device, proof_stage=runtime_detected, and fallback_used=false without graph execution claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
