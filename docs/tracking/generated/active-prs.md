<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| cpu-proof | CPU-BITNET-005c | #3753 | `codex/cpu-bitnet-005c-avx2-parity-hardening` | AVX2 decode GEMV parity is hardened against scalar across rows, tail-column shapes, deterministic patterns, and repeated-run equality. |
| intel-258v-platform | ARC140V-002 | #3727 | `codex/intel-arc/ARC140V-002-runtime-probe` | Probe exact Arc 140V runtime visibility by name or PCI ID 0x64A0 across OpenCL, Level Zero, and OpenVINO GPU.0 while recording proof_stage=runtime_detected, requested/selected backend identity, runtime API, and fallback_used=false. |
| intel-npu | NPU-003 | #3739 | `codex/intel-npu/NPU-003-openvino-runtime-probe` | Add Intel NPU runtime detection fields that keep OS accelerator evidence separate from OpenVINO NPU visibility and record OpenVINO NPU full name, driver/compiler/memory properties, runtime device, proof_stage=runtime_detected, and fallback_used=false without graph execution claims. |
| nvidia-5070ti | RTX5070TI-008 | #3770 | `codex/nvidia-5070ti/RTX5070TI-008-benchmark-baseline` | Benchmark parity-tested RTX 5070 Ti CUDA kernels/subgraphs against the 9950X3D CPU reference with driver/runtime/VRAM/power/thermal context and no full inference or unproven speedup claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
