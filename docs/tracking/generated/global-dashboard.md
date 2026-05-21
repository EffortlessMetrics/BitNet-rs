<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# BitNet Campaign Dashboard

| Campaign | Active item | PR | State | Next | Notes |
|---|---|---:|---|---|---|
| amd-cpu-baselines | AMD5700X-003 | TBD | ready | AMD9950X3D-003 | These lanes are CPU proof lanes, not accelerator lanes. |
| amd-rocm | ROCM-DOCS-000 | TBD | ready | none | Do not claim generic AMD GPU support. |
| apple-bitnet-artifact-sweep | ABAS-001 | TBD | proposed | ABAS-002 | Use MacBook first for larger artifact sweeps; do not manufacture MacBook receipts from the M4 Mac mini. |
| apple-m3-macbook-air | M3MBA-023 | TBD | proposed | M3MBA-024 | This is the Apple M3 MacBook Air lane, not the M4 Mac mini product, performance, or strict-proof lane. |
| apple-m4 | M4-018 | #3826 | merged | none | Do not touch QK256 before a BitNet-specific Apple item explicitly allows it. |
| apple-m4-bitnet-eval-and-benchmark | M4-BITNET-EVAL-005 | #4942 | merged | none | This is an M4 Mac mini BitNet campaign. |
| apple-m4-bitnet-productization | M4-BITNET-PROD-004 | #4957 | merged | none | This is an M4 Mac mini BitNet campaign. |
| apple-m4-continuity | M4-CONT-005 | #4270 | merged | none | This is an M4 Mac mini local campaign; do not execute MacBook artifact sweeps or MacBook receipts here. |
| apple-m4-dense-slm-regression | M4-SLM-REG-005 | #4198 | merged | none | Do not reopen the completed apple-m4, apple-m4-slm-answer, apple-m4-productization, or apple-m4-slm-performance campaigns. |
| apple-m4-durable-inference-evidence | M4-DURABLE-005 | #5025 | merged | none | This is an M4 Mac mini evidence-refresh campaign. |
| apple-m4-inference-excellence | M4-TREND-001 | TBD | proposed | M4-MODEL-LIFECYCLE-001 | This is an M4 Mac mini inference-excellence campaign. |
| apple-m4-inference-ops | M4-INF-OPS-004 | #4969 | merged | none | This is an M4 Mac mini operations campaign. |
| apple-m4-local-answer | M4-BITNET-WARM-002 | #4705 | merged | none | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-local-server | M4-SERVE-005 | #4374 | merged | none | This is an M4 Mac mini dense SLM service campaign. |
| apple-m4-operational | M4-OP-006 | #3882 | merged | none | Do not reopen the completed apple-m4 proof campaign. |
| apple-m4-productization | M4-PROD-005 | #4034 | merged | none | Do not reopen the completed apple-m4, apple-m4-operational, or apple-m4-slm-answer campaigns. |
| apple-m4-slm-answer | SLM-M4-007 | #3991 | merged | none | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-slm-eval-and-proof | M4-SLM-EVAL-006 | #4677 | merged | none | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-eval-v2 | M4-SLM-EVAL2-005 | #4886 | merged | none | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-excellence | M4-SLM-EX-010 | #4307 | merged | none | This is an M4 Mac mini local campaign. |
| apple-m4-slm-hardening | M4-SLM-HARDEN-004 | #4161 | merged | none | Do not reopen completed Apple M4 proof, operational, SLM answer, productization, or performance campaigns. |
| apple-m4-slm-metal-phases | M4-METAL-007 | #4397 | merged | none | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-model-breadth | M4-MODEL-008 | TBD | blocked | none | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-performance | M4-SLM-PERF-007 | #4081 | merged | none | Do not reopen the completed apple-m4, apple-m4-operational, apple-m4-slm-answer, or apple-m4-productization campaigns. |
| apple-silicon-macbook | MB-AS-002 | TBD | blocked | MB-AS-004 | Do not reopen the completed apple-m4 proof, operational, SLM answer, productization, performance, hardening, or regression campaigns. |
| bitnet-b158-3b | B158-3B-001 | TBD | ready | none | Do not commit model binaries. |
| ci-coverage | CI-COVERAGE-002 | #5775 | merged | none | Do not block unrelated runtime or tracker work on optional coverage uploads. |
| cpu-proof | CPU-AVX2-HOTPATH-001 | #5963 | merged | none | 258V CPU is the lead BitNet CPU reference; no GPU or NPU claims. |
| cpu-qk256-performance | KBL8250U-004 | #3839 | merged | none | Do not claim performance before strict proof receipts exist. |
| crate-collapse | LEAF-001 | TBD | proposed | none | Do not combine crate movement with runtime proof. |
| falcon-e-family | FE-000 | TBD | ready | FE-001 | Do not commit model binaries. |
| falcon3-family | F3-000 | TBD | ready | none | Do not commit model binaries. |
| i2s | I2S-DOCS-000 | #5880 | merged | none | Do not change runtime code in docs-only I2_S tracker slices. |
| intel-258v-platform | LNL258V-POWER-006 | TBD | blocked | none | 258V CPU proof is first priority; NPU and Arc proofs must compare against the 258V CPU reference before BitNet-adjacent parity claims. |
| intel-a770 | A770-014 | #198 | merged | none | OpenCL-first for native A770 proof. |
| intel-npu | NPU-013 | #5903 | merged | none | Device-node detection is not inference. |
| llama3-8b-158 | LLAMA3-158-000 | TBD | ready | LLAMA3-158-001 | Do not commit model binaries. |
| model-artifacts | MODEL-ARTIFACT-002 | #3928 | blocked | none | Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates. |
| nvidia-5070ti | CUDA-MODEL-017A | TBD | ready | CUDA-MODEL-017 | CUDA visibility is not kernel execution. |
| official-bitnet-2b | OFFICIAL-2B-000 | TBD | ready | OFFICIAL-2B-001 | Do not commit model binaries. |
| qwen36 | QWEN36-DOCS-000 | #5892 | merged | none | Qwen3.6 registration is not native BitNet-rs inference support. |
| server-real-inference | SERVER-005 | #4490 | merged | none | Do not reintroduce simulated inference. |
| slm-cpu | SLM-CPU-075 | TBD | ready | none | Do not edit BitNet QK256/I2_S kernels. |
| tl1 | TL1-PLAN-000 | TBD | ready | none | TL1 registration is not native BitNet-rs inference support. |
| tl2 | TL2-DOCS-000 | TBD | ready | none | TL2 registration is not native BitNet-rs inference support. |
| tracker-infra | TRACKER-003 | #3724 | merged | none | Do not touch runtime code, kernels, or dependencies for tracker infrastructure. |
| wasm-inference | WASM-002 | TBD | ready | WASM-003 | WASM detection is not inference. |
