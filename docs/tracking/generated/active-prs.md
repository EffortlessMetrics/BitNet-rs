<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-npu | NPU-008 | #3963 | `codex/intel-npu/NPU-008-linear-subgraph-parity` | Add selected static BitNet linear-projection subgraph parity through OpenVINO NPU with CPU reference comparison, selected backend/runtime identity, timing, fallback_used=false, and no full BitNet inference, acceleration, or QK256 decode claims. |
| slm-cpu | SLM-CPU-005 | #3969 | `codex/slm-cpu-005-reference-divergence` | Add a machine-checkable reference divergence artifact schema and validator comparing bitnet-rs against a known-good external run by model SHA, prompt/template/BOS policy, prompt IDs, generated IDs, decoded text, top-k when available, and first divergence. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
