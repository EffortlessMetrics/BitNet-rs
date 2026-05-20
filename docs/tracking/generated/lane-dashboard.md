<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Campaign Lane Dashboard

| Campaign | Title | Current item | Boundary |
|---|---|---|---|
| amd-cpu-baselines | AMD CPU baselines | AMD5700X-003 | These lanes are CPU proof lanes, not accelerator lanes. |
| amd-rocm | AMD ROCm productization | ROCM-DOCS-000 | Do not claim generic AMD GPU support. |
| apple-bitnet-artifact-sweep | Apple BitNet artifact sweep | ABAS-001 | Use MacBook first for larger artifact sweeps; do not manufacture MacBook receipts from the M4 Mac mini. |
| apple-m3-macbook-air | Apple M3 MacBook Air | M3MBA-006 | This is the Apple M3 MacBook Air lane, not the M4 Mac mini product, performance, or strict-proof lane. |
| apple-m4 | Apple M4 Mac mini validation | M4-018 | Do not touch QK256 before a BitNet-specific Apple item explicitly allows it. |
| apple-m4-bitnet-eval-and-benchmark | Apple M4 BitNet eval and benchmark | M4-BITNET-EVAL-005 | This is an M4 Mac mini BitNet campaign. |
| apple-m4-bitnet-productization | Apple M4 BitNet productization | M4-BITNET-PROD-004 | This is an M4 Mac mini BitNet campaign. |
| apple-m4-continuity | Apple M4 continuity | M4-CONT-005 | This is an M4 Mac mini local campaign; do not execute MacBook artifact sweeps or MacBook receipts here. |
| apple-m4-dense-slm-regression | Apple M4 dense SLM regression guardrails | M4-SLM-REG-005 | Do not reopen the completed apple-m4, apple-m4-slm-answer, apple-m4-productization, or apple-m4-slm-performance campaigns. |
| apple-m4-durable-inference-evidence | Apple M4 durable inference evidence | M4-DURABLE-005 | This is an M4 Mac mini evidence-refresh campaign. |
| apple-m4-inference-excellence | Apple M4 inference excellence | M4-BENCH-003 | This is an M4 Mac mini inference-excellence campaign. |
| apple-m4-inference-ops | Apple M4 inference ops | M4-INF-OPS-004 | This is an M4 Mac mini operations campaign. |
| apple-m4-local-answer | Apple M4 local answer usability | M4-BITNET-WARM-002 | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-local-server | Apple M4 local server | M4-SERVE-005 | This is an M4 Mac mini dense SLM service campaign. |
| apple-m4-operational | Apple M4 operational readiness | M4-OP-006 | Do not reopen the completed apple-m4 proof campaign. |
| apple-m4-productization | Apple M4 local answer productization | M4-PROD-005 | Do not reopen the completed apple-m4, apple-m4-operational, or apple-m4-slm-answer campaigns. |
| apple-m4-slm-answer | Apple M4 SLM local answer usability | SLM-M4-007 | Do not reopen the completed apple-m4 or apple-m4-operational campaigns. |
| apple-m4-slm-eval-and-proof | Apple M4 dense SLM eval and proof | M4-SLM-EVAL-006 | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-eval-v2 | Apple M4 dense SLM eval v2 | M4-SLM-EVAL2-005 | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-excellence | Apple M4 SLM excellence | M4-SLM-EX-010 | This is an M4 Mac mini local campaign. |
| apple-m4-slm-hardening | Apple M4 SLM hardening | M4-SLM-HARDEN-004 | Do not reopen completed Apple M4 proof, operational, SLM answer, productization, or performance campaigns. |
| apple-m4-slm-metal-phases | Apple M4 SLM Metal phases | M4-METAL-007 | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-model-breadth | Apple M4 SLM model breadth | M4-MODEL-008 | This is an M4 Mac mini dense SLM campaign. |
| apple-m4-slm-performance | Apple M4 SLM performance | M4-SLM-PERF-007 | Do not reopen the completed apple-m4, apple-m4-operational, apple-m4-slm-answer, or apple-m4-productization campaigns. |
| apple-silicon-macbook | Apple Silicon MacBook cross-reference | MB-AS-002 | Do not reopen the completed apple-m4 proof, operational, SLM answer, productization, performance, hardening, or regression campaigns. |
| bitnet-b158-3b | BitNet b1.58 3B TL candidate | B158-3B-001 | Do not commit model binaries. |
| ci-coverage | CI coverage | CI-COVERAGE-002 | Do not block unrelated runtime or tracker work on optional coverage uploads. |
| cpu-proof | BitNet CPU proof | CPU-AVX2-HOTPATH-001 | 258V CPU is the lead BitNet CPU reference; no GPU or NPU claims. |
| cpu-qk256-performance | CPU QK256 performance | KBL8250U-004 | Do not claim performance before strict proof receipts exist. |
| crate-collapse | Crate collapse | LEAF-001 | Do not combine crate movement with runtime proof. |
| falcon-e-family | Falcon-E Family compact 1.58-bit lane | FE-000 | Do not commit model binaries. |
| falcon3-family | Falcon3 multi-size BitNet-family onboarding | F3-000 | Do not commit model binaries. |
| i2s | I2_S productization | I2S-DOCS-000 | Do not change runtime code in docs-only I2_S tracker slices. |
| intel-258v-platform | Intel 258V platform validation | LNL258V-POWER-006 | 258V CPU proof is first priority; NPU and Arc proofs must compare against the 258V CPU reference before BitNet-adjacent parity claims. |
| intel-a770 | Intel Arc A770 validation | A770-009 | OpenCL-first for native A770 proof. |
| intel-npu | Intel NPU validation | NPU-013 | Device-node detection is not inference. |
| llama3-8b-158 | Llama3 8B 1.58 supported-model candidate | LLAMA3-158-000 | Do not commit model binaries. |
| model-artifacts | Model artifact answer authority | MODEL-ARTIFACT-002 | Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates. |
| nvidia-5070ti | NVIDIA RTX 5070 Ti validation | CUDA-DENSE-014 | CUDA visibility is not kernel execution. |
| official-bitnet-2b | Official Microsoft BitNet 2B productization | OFFICIAL-2B-000 | Do not commit model binaries. |
| qwen36 | Qwen3.6 governed model family | QWEN36-DOCS-000 | Qwen3.6 registration is not native BitNet-rs inference support. |
| server-real-inference | Server real inference | SERVER-005 | Do not reintroduce simulated inference. |
| slm-cpu | Small dense model CPU proof | SLM-CPU-067 | Do not edit BitNet QK256/I2_S kernels. |
| tl1 | TL1 ARM table lookup route | TL1-PLAN-000 | TL1 registration is not native BitNet-rs inference support. |
| tl2 | TL2 x86 table lookup route | TL2-DOCS-000 | TL2 registration is not native BitNet-rs inference support. |
| tracker-infra | Tracker infrastructure | TRACKER-003 | Do not touch runtime code, kernels, or dependencies for tracker infrastructure. |
| wasm-inference | WASM inference proof lane | WASM-002 | WASM detection is not inference. |
