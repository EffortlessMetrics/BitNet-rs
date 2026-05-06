<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Campaign Lane Dashboard

| Campaign | Title | Current item | Boundary |
|---|---|---|---|
| amd-cpu-baselines | AMD CPU baselines | AMD5700X-003 | These lanes are CPU proof lanes, not accelerator lanes. |
| apple-m4 | Apple M4 Mac mini validation | M4-007 | Do not touch QK256 before a BitNet-specific Apple item explicitly allows it. |
| ci-coverage | CI coverage | CI-COVERAGE-001 | Do not block unrelated runtime or tracker work on optional coverage uploads. |
| cpu-proof | BitNet CPU proof | CPU-BITNET-004 | No GPU or NPU claims. |
| cpu-qk256-performance | CPU QK256 performance | KBL8250U-003 | Do not claim performance before strict proof receipts exist. |
| crate-collapse | Crate collapse | LEAF-001 | Do not combine crate movement with runtime proof. |
| intel-258v-platform | Intel 258V platform validation | LNL258V-002 | Arc 140V OpenCL proof is not NPU proof. |
| intel-a770 | Intel Arc A770 validation | A770-003 | OpenCL-first for native A770 proof. |
| intel-npu | Intel NPU validation | NPU-002 | Device-node detection is not inference. |
| nvidia-5070ti | NVIDIA RTX 5070 Ti validation | RTX5070TI-005 | CUDA visibility is not kernel execution. |
| server-real-inference | Server real inference | SERVER-001 | Do not reintroduce simulated inference. |
| tracker-infra | Tracker infrastructure | TRACKER-001 | Do not touch runtime code, kernels, or dependencies for tracker infrastructure. |
