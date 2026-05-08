<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# BitNet Campaign Dashboard

| Campaign | Active item | PR | State | Next | Notes |
|---|---|---:|---|---|---|
| amd-cpu-baselines | AMD5700X-003 | TBD | ready | AMD9950X3D-003 | These lanes are CPU proof lanes, not accelerator lanes. |
| apple-m4 | M4-018 | #3826 | merged | none | Do not touch QK256 before a BitNet-specific Apple item explicitly allows it. |
| apple-m4-local-answer | M4-QA-001 | #3904 | blocked | M4-QA-MODEL-002 | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-operational | M4-OP-006 | #3882 | merged | none | Do not reopen the completed apple-m4 proof campaign. |
| apple-m4-productization | M4-PROD-005 | #4034 | merged | none | Do not reopen the completed apple-m4, apple-m4-operational, or apple-m4-slm-answer campaigns. |
| apple-m4-slm-answer | SLM-M4-007 | #3991 | merged | none | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-slm-performance | M4-SLM-PERF-007 | #4081 | merged | none | Do not reopen the completed apple-m4, apple-m4-operational, apple-m4-slm-answer, or apple-m4-productization campaigns. |
| ci-coverage | CI-COVERAGE-001 | #3620 | merged | none | Do not block unrelated runtime or tracker work on optional coverage uploads. |
| cpu-proof | CPU-ANSWER-007 | #4019 | merged | none | 258V CPU is the lead BitNet CPU reference; no GPU or NPU claims. |
| cpu-qk256-performance | KBL8250U-004 | #3839 | merged | none | Do not claim performance before strict proof receipts exist. |
| crate-collapse | LEAF-001 | TBD | proposed | none | Do not combine crate movement with runtime proof. |
| intel-258v-platform | LNL258V-COMPARE-002 | TBD | ready | none | 258V CPU proof is first priority; NPU and Arc proofs must compare against the 258V CPU reference before BitNet-adjacent parity claims. |
| intel-a770 | A770-003 | TBD | ready | none | OpenCL-first for native A770 proof. |
| intel-npu | NPU-011 | #4097 | merged | none | Device-node detection is not inference. |
| model-artifacts | MODEL-ARTIFACT-002 | #3928 | blocked | none | Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates. |
| nvidia-5070ti | CUDA-DENSE-002 | TBD | proposed | none | CUDA visibility is not kernel execution. |
| server-real-inference | SERVER-001 | TBD | proposed | none | Do not reintroduce simulated inference. |
| slm-cpu | SLM-CPU-006B | #4096 | merged | none | Do not edit BitNet QK256/I2_S kernels. |
| tracker-infra | TRACKER-003 | #3724 | pr_open | none | Do not touch runtime code, kernels, or dependencies for tracker infrastructure. |
