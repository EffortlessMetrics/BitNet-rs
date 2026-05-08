<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | ARC140V-004 | #3953 | `codex/intel-258v-platform/ARC140V-004-opencl-smoke` | Run a tiny native OpenCL vector-add kernel on Arc 140V, record runtime/device identity, kernel shape/timing/tolerance fields, fallback=false, and no BitNet/QK256/OpenVINO acceleration claims. |
| model-artifacts | MODEL-ARTIFACT-004 | #3939 | `codex/model-artifacts/MODEL-ARTIFACT-004-ikllama-intended-runner` | Record intended ik_llama.cpp runner evidence for official-derived BitNet GGUF candidates, including tdh111 IQ2_BN_R4 prompt-suite output and official Microsoft I2_S prompt-suite output, without promoting an answer_ready artifact or changing runtime behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
