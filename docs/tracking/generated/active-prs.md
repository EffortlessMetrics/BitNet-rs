<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-RUN-001 | #3714 | `codex/intel-258v/LNL258V-RUN-001-platform-probe` | Add a JSON-ready Lunar Lake 258V platform probe that records CPU AVX2 facts, Arc 140V OpenCL/Level Zero/OpenVINO GPU visibility, Intel NPU OS/OpenVINO visibility, memory, power, OS, proof_stage=runtime_detected, and fallback_used=false without inference claims. |
| intel-npu | NPU-003 | #3739 | `codex/intel-npu/NPU-003-openvino-runtime-probe` | Add Intel NPU runtime detection fields that keep OS accelerator evidence separate from OpenVINO NPU visibility and record OpenVINO NPU full name, driver/compiler/memory properties, runtime device, proof_stage=runtime_detected, and fallback_used=false without graph execution claims. |
| nvidia-5070ti | RTX5070TI-005 | #3723 | `codex/nvidia-5070ti/RTX5070TI-005-smoke-receipt` | Compile and run a tiny CUDA kernel on the selected RTX 5070 Ti with a fallback-free smoke receipt and no BitNet inference or speedup claim. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
