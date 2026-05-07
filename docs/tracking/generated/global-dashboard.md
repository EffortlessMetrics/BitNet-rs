<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# BitNet Campaign Dashboard

| Campaign | Active item | PR | State | Next | Notes |
|---|---|---:|---|---|---|
| amd-cpu-baselines | AMD5700X-003 | TBD | ready | AMD9950X3D-003 | These lanes are CPU proof lanes, not accelerator lanes. |
| apple-m4 | M4-018 | #3826 | merged | none | Do not touch QK256 before a BitNet-specific Apple item explicitly allows it. |
| apple-m4-operational | M4-OP-006 | #3882 | pr_open | none | Do not reopen the completed apple-m4 proof campaign. |
| ci-coverage | CI-COVERAGE-001 | #3620 | merged | none | Do not block unrelated runtime or tracker work on optional coverage uploads. |
| cpu-proof | CPU-ANSWER-001 | TBD | ready | none | No GPU or NPU claims. |
| cpu-qk256-performance | KBL8250U-004 | #3839 | merged | none | Do not claim performance before strict proof receipts exist. |
| crate-collapse | LEAF-001 | TBD | proposed | none | Do not combine crate movement with runtime proof. |
| intel-258v-platform | CPU258V-001 | #3802 | merged | none | Arc 140V OpenCL proof is not NPU proof. |
| intel-a770 | A770-003 | TBD | ready | none | OpenCL-first for native A770 proof. |
| intel-npu | NPU-006 | #3860 | merged | none | Device-node detection is not inference. |
| nvidia-5070ti | CUDA-DENSE-001 | TBD | proposed | none | CUDA visibility is not kernel execution. |
| server-real-inference | SERVER-001 | TBD | proposed | none | Do not reintroduce simulated inference. |
| tracker-infra | TRACKER-003 | #3724 | pr_open | none | Do not touch runtime code, kernels, or dependencies for tracker infrastructure. |
