<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-performance | M4-SLM-PERF-007 | #4081 | `codex/apple-m4-slm-performance/M4-SLM-PERF-007-performance-envelope` | Publish a measured Apple M4 SLM performance envelope for supported models and profiles only, recording machine context, backend, profile, timings, phase contributions, fallback status, and explicit unsupported claims. |
| intel-npu | NPU-010 | #4080 | `codex/intel-npu/NPU-010-live-openvino-receipts` | Record live 258V OpenVINO 2026.1 Intel NPU runtime visibility, tiny static graph smoke, and selected BitNet RMSNorm and linear-projection static subgraph parity receipts with selected_backend=intel-npu-openvino, runtime_api=openvino, runtime_device=NPU, fallback_used=false, and no full BitNet inference, acceleration, QK256 decode, or CPU fallback proof claims. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
