# bitnet-rs Alignment Status

Updated: 2026-05-05

## Current Focus

P0 truth boundary, crate consolidation inventory, and docs-only hardware-lane scaffolding.

## Active PR queue

| Item | PR | State | Notes |
|---|---:|---|---|
| TRUTH-002 | TBD | ready | Make GGUF fallback explicit |
| TRUTH-003 | #3626 | pr_open | Null-byte model path validation |
| INV-001 | TBD | ready | Crate consolidation map |
| HW-001 | TBD | pr_open | Add shared hardware validation matrix and proof-stage contract |
| BITNET-001 | TBD | pr_open | Add BitNet model/kernel/receipt proof contract |
| NPU-001 | TBD | pr_open | Add Intel NPU backend lane without runtime execution |
| A770-001 | TBD | pr_open | Add Intel Arc A770 OpenCL-first backend lane without runtime execution |
| LNL258V-001 | TBD | pr_open | Add Core Ultra 7 258V tri-device platform profile without runtime execution |
| KBL8250U-001 | TBD | pr_open | Add Intel Core i5-8250U CPU AVX2 validation lane without runtime execution |
| M4-001 | TBD | pr_open | Add Apple M4 Mac mini Metal-first backend lane without runtime execution |
| RTX5070TI-001 | TBD | pr_open | Add NVIDIA RTX 5070 Ti CUDA-first backend lane without runtime execution |
| AMD9950X3D-001 | TBD | pr_open | Add AMD Ryzen 9 9950X3D AVX-512 CPU validation lane without runtime execution |
| AMD5700X-001 | TBD | pr_open | Add AMD Ryzen 7 5700X AVX2 CPU validation lane without runtime execution |

## Hardware Lanes

Track hardware as separate proof lanes:

| Hardware | Primary lane | Role |
|---|---|---|
| Core i5-8250U | `intel-i5-8250u-cpu-avx2` | Active low-power AVX2 CPU implementation and proof lane |
| Core Ultra 7 258V CPU | `intel-258v-cpu-avx2` | Parallel Lunar Lake CPU AVX2 validation and same-machine comparison lane |
| Ryzen 7 5700X | `amd-5700x-cpu-avx2` | Mainstream AM4 / DDR4 desktop CPU AVX2 baseline |
| Ryzen 9 9950X3D | `amd-9950x3d-cpu-avx512` | Modern AM5 / DDR5 / AVX-512 / large-cache CPU lane |
| Arc A770 16GB | `intel-arc-a770-opencl` | Discrete GPU performance lane for native OpenCL BitNet kernels |
| Arc 140V | `intel-arc-140v-opencl` | Lunar Lake integrated GPU comparison lane |
| Core Ultra 7 258V NPU | `intel-npu-openvino` | OpenVINO NPU static-shape graph lane |
| M4 Mac mini | `apple-m4-metal` | Apple Silicon Metal lane |
| RTX 5070 Ti | `nvidia-rtx-5070-ti-cuda` | Modern NVIDIA CUDA lane |

Metal, MPSGraph, CUDA, WGPU, OpenCL, Level Zero, OpenVINO GPU, and OpenVINO NPU receipts must preserve requested backend, selected backend, resolved device identity, and fallback status. Do not merge these into a generic `gpu`, `accelerator`, `metal`, `cuda`, `openvino`, `intel`, `npu`, or `oneapi` claim.

Shared contract docs:

- `docs/hardware/HARDWARE_MATRIX.md`
- `docs/hardware/PROOF_STAGES.md`
- `docs/hardware/LANE_OWNERSHIP.md`
- `docs/hardware/BENCHMARK_PROTOCOL.md`
- `docs/hardware/machine-profile.schema.yaml`
- `ci/hardware/README.md`

BitNet contract docs:

- `docs/bitnet/BITNET_MODEL_CONTRACT.md`
- `docs/bitnet/BITNET_QUANTIZATION_CONTRACT.md`
- `docs/bitnet/BITNET_KERNEL_MATRIX.md`
- `docs/bitnet/BITNET_RUNTIME_PHASES.md`
- `docs/bitnet/BITNET_REFERENCE_RUNS.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/bitnet/BITNET_PARITY_TOLERANCES.md`

Hardware artifacts say which machine/runtime/device ran. BitNet artifacts must also record model, tokenizer, quantization, kernel family, execution phase, reference path, and fallback status.

## Intel NPU Claim Boundary

The Intel NPU lane targets Lunar Lake through OpenVINO NPU. Device-node detection under `/dev/accel` is not inference, Linux driver evidence may use VPU names such as `intel_vpu`, OpenVINO NPU smoke is not full BitNet inference, Intel GPU/OpenCL is a separate backend lane, CPU fallback cannot count as Intel NPU execution, and QK256 CPU execution cannot count as NPU execution.

Planned follow-up items are NPU-002 backend identity cleanup, NPU-003 Intel NPU runtime detection, NPU-004 CLI or xtask smoke output, NPU-005 OpenVINO static tiny graph smoke, NPU-006 receipt backend identity, shape, cache, and fallback fields, NPU-007 BitNet subgraph parity, and NPU-008 OpenVINO llama.cpp GGUF reference evaluation.

The Lunar Lake laptop should provide the Windows-native or Linux hardware, kernel/driver, OpenVINO, and OpenCL comparison bundle from `docs/specs/intel-lunar-lake-npu-roadmap.md` before implementation claims move beyond scaffold. Do not assume WSL can see the NPU.

## Intel Arc Claim Boundary

The Arc A770 lane is OpenCL-first through bitnet-kernels, with OpenVINO GPU as a reference path. `clinfo` visibility is not execution, OpenVINO GPU smoke is not native BitNet kernel proof, CPU fallback cannot count as A770 execution, and performance claims require receipt-backed benchmark artifacts.

The Arc 140V lane is the Lunar Lake integrated GPU comparison path. It shares system memory and laptop power limits, so its receipts must record memory and power/thermal context before performance comparisons.

## Lunar Lake 258V Platform Boundary

The 258V platform lane ties together CPU AVX2, Arc 140V GPU, and Intel AI Boost NPU receipts on the same laptop. It does not merge their claims: Arc 140V OpenCL work is not NPU work, OpenVINO GPU.0 smoke is not packed BitNet kernel proof, OpenVINO NPU smoke is not full inference, and WSL only counts for NPU validation if OpenVINO sees `NPU` inside WSL.

The 258V CPU lane validates the same CPU path on Lunar Lake and provides same-machine comparison against Arc 140V and NPU results. It is not a replacement for the 8250U AVX2 implementation lane.

## Legacy Mobile CPU Boundary

The i5-8250U lane is the active AVX2 CPU implementation/proof lane. It owns CPU scalar correctness, CPU AVX2 dispatch validation, strict CPU proof receipts, and thermal-throttle-aware sustained baselines. UHD 620 visibility is optional/deferred and does not count as CPU progress. Short turbo results must not be reported as sustained mobile performance.

## AMD Desktop CPU Boundary

The 5700X and 9950X3D lanes are CPU proof lanes, not accelerator lanes. The 5700X owns mainstream Zen 3 AM4/DDR4 AVX2 proof and must not claim AVX-512. The 9950X3D owns modern Zen 5 AM5/DDR5 AVX-512, AVX2 comparison, and cache-sensitive proof; receipts must record cache-domain, scheduler/core placement, cooling, memory, and sustained-power context before performance claims.

## Apple Silicon Boundary

The M4 Mac mini lane is Metal-first, with MPSGraph as a graph/reference lane and CPU/NEON as fallback/parity. CPU fallback cannot count as Metal execution, MPSGraph smoke cannot count as native Metal kernel proof, and Neural Engine execution must not be claimed unless the resolved target is receipt-backed.

## NVIDIA CUDA Boundary

The RTX 5070 Ti lane is CUDA-first, with wgpu/Vulkan/D3D12 as a cross-platform reference lane. CUDA visibility is not kernel execution, WGPU smoke is not CUDA proof, CPU fallback cannot count as CUDA execution, and performance claims require driver, CUDA version, compute capability, VRAM, power, and thermal context.

## Queue hygiene

| Cluster | Decision | Notes |
|---|---|---|
| Codecov duplicates | Deferred canonical #3620 | #3609-#3612 and #3617-#3619 are recorded as superseded by #3620; handle this in a separate CI coverage review. |
| Null-byte Sentinel duplicates | Tied to TRUTH-003 | Do not merge pre-ledger Sentinel branches; close duplicates after the canonical TRUTH-003 PR lands. |
| Accessibility Palette duplicates | Deferred | Review after truth boundary and inventory are complete; #3607 is the latest matching candidate for later review. |
| Sampling/performance Bolt duplicates | Deferred | Hold until truth boundary and inventory are complete. |

## Completed

| Item | PR | Merge SHA | Notes |
|---|---:|---|---|
| TRUTH-001 | #3621 | 10ea2b409a1d4095205722da77a600d31bb57d04 | Fence server simulated inference merged. |
| QUEUE-001 | #3623 | 678a3ba1592ab10e9a7b473db077ec93b1d867fb | Codecov duplicate cluster normalized; #3620 retained as deferred canonical candidate. |

## Blocked

| Item | Blocker | Next action |
|---|---|---|

## Superseded

| Item/PR | Superseded by | Reason |
|---|---|---|
