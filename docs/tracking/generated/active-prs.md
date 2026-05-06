<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4 | M4-011 | #3769 | `codex/apple-m4/M4-011-metal-i2s-smoke-parity` | Run an I2_S-adjacent native Metal smoke or parity target against Apple CPU/NEON without claiming full inference. |
| intel-258v-platform | ARC140V-002 | #3727 | `codex/intel-arc/ARC140V-002-runtime-probe` | Probe exact Arc 140V runtime visibility by name or PCI ID 0x64A0 across OpenCL, Level Zero, and OpenVINO GPU.0 while recording proof_stage=runtime_detected, requested/selected backend identity, runtime API, and fallback_used=false. |
| intel-npu | NPU-003 | #3739 | `codex/intel-npu/NPU-003-openvino-runtime-probe` | Add Intel NPU runtime detection fields that keep OS accelerator evidence separate from OpenVINO NPU visibility and record OpenVINO NPU full name, driver/compiler/memory properties, runtime device, proof_stage=runtime_detected, and fallback_used=false without graph execution claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
