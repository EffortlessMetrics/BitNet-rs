<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Campaign Lane Dashboard

| Campaign | Title | Current item | Boundary |
|---|---|---|---|
| amd-cpu-baselines | AMD CPU baselines | AMD5700X-003 | These lanes are CPU proof lanes, not accelerator lanes. |
| apple-m4 | Apple M4 Mac mini validation | M4-018 | Do not touch QK256 before a BitNet-specific Apple item explicitly allows it. |
| apple-m4-local-answer | Apple M4 local answer usability | M4-QA-001 | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-operational | Apple M4 operational readiness | M4-OP-006 | Do not reopen the completed apple-m4 proof campaign. |
| apple-m4-productization | Apple M4 local answer productization | M4-PROD-005 | Do not reopen the completed apple-m4, apple-m4-operational, or apple-m4-slm-answer campaigns. |
| apple-m4-slm-answer | Apple M4 SLM local answer usability | SLM-M4-007 | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-slm-performance | Apple M4 SLM performance | M4-SLM-PERF-004 | Do not reopen the completed apple-m4, apple-m4-operational, apple-m4-slm-answer, or apple-m4-productization campaigns. |
| ci-coverage | CI coverage | CI-COVERAGE-001 | Do not block unrelated runtime or tracker work on optional coverage uploads. |
| cpu-proof | BitNet CPU proof | CPU-ANSWER-007 | 258V CPU is the lead BitNet CPU reference; no GPU or NPU claims. |
| cpu-qk256-performance | CPU QK256 performance | KBL8250U-004 | Do not claim performance before strict proof receipts exist. |
| crate-collapse | Crate collapse | LEAF-001 | Do not combine crate movement with runtime proof. |
| intel-258v-platform | Intel 258V platform validation | CPU258V-015 | 258V CPU proof is first priority; NPU and Arc proofs must compare against the 258V CPU reference before BitNet-adjacent parity claims. |
| intel-a770 | Intel Arc A770 validation | A770-003 | OpenCL-first for native A770 proof. |
| intel-npu | Intel NPU validation | NPU-008 | Device-node detection is not inference. |
| model-artifacts | Model artifact answer authority | MODEL-ARTIFACT-002 | Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates. |
| nvidia-5070ti | NVIDIA RTX 5070 Ti validation | CUDA-ANSWER-012 | CUDA visibility is not kernel execution. |
| server-real-inference | Server real inference | SERVER-001 | Do not reintroduce simulated inference. |
| slm-cpu | Small dense model CPU proof | SLM-CPU-006 | Do not edit BitNet QK256/I2_S kernels. |
| tracker-infra | Tracker infrastructure | TRACKER-003 | Do not touch runtime code, kernels, or dependencies for tracker infrastructure. |
