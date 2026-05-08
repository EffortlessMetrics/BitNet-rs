<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-npu | NPU-008 | #3963 | `codex/intel-npu/NPU-008-linear-subgraph-parity` | Add selected static BitNet linear-projection subgraph parity through OpenVINO NPU with CPU reference comparison, selected backend/runtime identity, timing, fallback_used=false, and no full BitNet inference, acceleration, or QK256 decode claims. |
| model-artifacts | MODEL-ARTIFACT-004 | #3939 | `codex/model-artifacts/MODEL-ARTIFACT-004-ikllama-intended-runner` | Record intended ik_llama.cpp runner evidence for official-derived BitNet GGUF candidates, including tdh111 IQ2_BN_R4 prompt-suite output and official Microsoft I2_S prompt-suite output, without promoting an answer_ready artifact or changing runtime behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
