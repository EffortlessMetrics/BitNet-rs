# bitnet-rs Alignment Status

Updated: 2026-05-06

## Current Focus

Real BitNet CPU path sequencing remains active, Apple Silicon M4-003 backend identity is merged, and the NVIDIA CUDA proof bench is documented as a separate staged lane. The next Apple Silicon item is M4-004 Metal probe, before Metal execution smoke, kernels, MPSGraph execution, parity, receipts, or benchmarks. The active NVIDIA item is `RTX5070TI-003` backend identity, before CUDA/NVML probing, kernel smoke, parity, receipts, benchmarks, or BitNet CUDA inference work.

Transition note: campaign-local `active.toml` files and append-only `events/*.toml` files are now the active tracker model. This legacy global status file remains a transition view; normal item PRs should use campaign-local tracking plus generated dashboards instead of hand-editing this table.

Model-family planning for non-BitNet targets lives separately in `docs/tracking/model-family-foundation/status.md` so model support states do not mix with CPU/A770/NPU/CUDA hardware proof lanes.

## Active / Coordination Queue

| Item | PR | State | Notes |
|---|---:|---|---|
| HW-001 | #3625 | merged | Shared hardware validation matrix and proof-stage contract; docs/contracts only, no runtime execution |
| BITNET-001 | #3625 | merged | BitNet model, kernel, and receipt proof contract; docs/contracts only |
| NPU-001 | #3625 | merged | Intel NPU backend lane scaffold; no runtime execution |
| A770-001 | #3625 | merged | Intel Arc A770 OpenCL-first lane scaffold; no runtime execution |
| LNL258V-001 | #3625 | merged | Core Ultra 7 258V platform scaffold; no runtime execution |
| KBL8250U-001 | #3625 | merged | i5-8250U AVX2 CPU lane scaffold; no runtime execution |
| M4-001 | #3625 | merged | Apple M4 Mac mini lane scaffold; no runtime execution |
| RTX5070TI-001 | #3625 | merged | RTX 5070 Ti CUDA lane scaffold; no runtime execution |
| AMD9950X3D-001 | #3625 | merged | AMD 9950X3D CPU lane scaffold; no runtime execution |
| AMD5700X-001 | #3625 | merged | AMD 5700X CPU lane scaffold; no runtime execution |
| NPU-002 | TBD | ready | Preserve Intel NPU backend identity before runtime probing |
| A770-003 | TBD | ready | Preserve Intel Arc A770 selected-device identity |
| KBL8250U-003 | TBD | ready | Prove i5-8250U scalar and AVX2 dispatch |
| M4-004 | TBD | ready | Add Apple M4 Metal device probe before execution smoke or inference claims |
| RTX5070TI-003 | #3679 | pr_open | Preserve RTX 5070 Ti CUDA selected-device identity |
| AMD9950X3D-003 | TBD | ready | Prove 9950X3D scalar AVX2 and AVX-512 dispatch |
| AMD5700X-003 | TBD | ready | Prove 5700X scalar and AVX2 dispatch |

These rows are coordination markers, not implementation proof. Merged scaffold
rows stay visible so the A770, NPU, 258V, 8250U, AMD, NVIDIA, and Mac lanes can
coordinate follow-up work without implying that runtime execution has been built
or tested.

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
- `docs/bitnet/BITNET_CPU_PATH_PLAN.md`
- `docs/bitnet/fixtures.yaml`

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

The Apple Silicon implementation order is M4-002 machine profile, M4-003 backend identity, M4-004 Metal probe, M4-005 Metal compute smoke, M4-006 CPU/Metal parity, M4-007 MPSGraph smoke, M4-008 receipt identity fields, and M4-009 benchmark baseline. M4-002 was docs/artifact prep only, and M4-003 preserved backend identity only. M4-004 is the next Apple Silicon item and must not touch Metal kernels, MPSGraph execution, QK256, server inference, or dependencies.

## NVIDIA CUDA Boundary

The RTX 5070 Ti lane is CUDA-first, with wgpu/Vulkan/D3D12 as a cross-platform reference lane. CUDA visibility is not kernel execution, WGPU smoke is not CUDA proof, CPU fallback cannot count as CUDA execution, and performance claims require driver, CUDA version, compute capability, VRAM, power, and thermal context.

The Windows CUDA proof bench is `windows-9950x3d-rtx5070ti`: AMD Ryzen 9 9950X3D is the CPU reference path and NVIDIA GeForce RTX 5070 Ti is the CUDA target. Receipts should use narrow backend labels such as `amd-9950x3d-cpu-avx512`, `nvidia-rtx-5070-ti-cuda`, and `nvidia-rtx-5070-ti-wgpu`; generic `gpu`, `cuda`, `nvidia`, `accelerated`, or `blackwell` labels are not enough for strict proof.

The current CUDA code is scaffolded kernel-provider infrastructure, not end-to-end CUDA inference. QK256 CUDA is scaffold-only until the packed fused dequant GEMV path is implemented and wired. The staged order is `RTX5070TI-003` backend identity, `RTX5070TI-004` CUDA/NVML probe, `RTX5070TI-005` tiny CUDA kernel smoke, `RTX5070TI-006` CPU/CUDA parity, `RTX5070TI-007` receipt/kernel counters, and `RTX5070TI-008` benchmark baseline. Only after those land should the `CUDA-BITNET-001` through `CUDA-BITNET-008` wave start persistent CUDA BitNet context, upload-once weights, reusable I2S linear, real QK256 CUDA, BitNetLinear routing, one-token strict proof, short decode proof, and full benchmark baselines.

Dense regular-LLM CUDA work is useful as a future `CUDA-DENSE-001` reference lane, but it must be labeled as dense regular LLM execution and cannot claim BitNet packed I2S/QK256 inference.

## Tracker Notes

HW-002 remains proposed. #3625 added `ci/hardware/README.md` with artifact naming guidance, but maintainers should confirm whether that fully satisfies HW-002 before marking it merged.

## Queue hygiene

| Cluster | Decision | Notes |
|---|---|---|
| Codecov duplicates | Deferred canonical #3620 | #3609-#3612 and #3617-#3619 are recorded as superseded by #3620; handle this in a separate CI coverage review. |
| Null-byte Sentinel duplicates | Superseded by #3626 | Pre-ledger Sentinel branches were closed after the canonical TRUTH-003 PR landed. |
| Accessibility Palette duplicates | Deferred | Review after truth boundary and inventory are complete; #3607 is the latest matching candidate for later review. |
| Sampling/performance Bolt duplicates | Deferred | Hold until truth boundary and inventory are complete. |

## Completed

| Item | PR | Merge SHA | Notes |
|---|---:|---|---|
| TRUTH-001 | #3621 | 10ea2b409a1d4095205722da77a600d31bb57d04 | Fence server simulated inference merged. |
| QUEUE-001 | #3623 | 678a3ba1592ab10e9a7b473db077ec93b1d867fb | Codecov duplicate cluster normalized; #3620 retained as deferred canonical candidate. |
| TRUTH-003 | #3626 | f44bbf5586b9bb97b336589d83bea85bedd13f4e | Null-byte model path validation merged. |
| TRUTH-002 | #3630 | 18e4f8142aaafec3528a0d54b5332a2b9f7583fd | GGUF minimal fallback made explicit. |
| INV-001 | #3632 | 457b36630906f2044e406e0dcf27ecb539e8a7a5 | Crate consolidation inventory completed for all workspace members. |
| CPU-001 | #3635 | d90f70f4410155077ffc9741e018e5d747d40a9f | Strict CPU proof command documented. |
| CPU-BITNET-000 | #3642 | c5f3480ac90cccfd1aec9766ae4628fd7e9fd3c3 | Real BitNet CPU path implementation plan merged. |
| CPU-BITNET-001 | #3651 | 5bf32dc335a52aeab5a790d3811a57dc06ed0d3d | Strict real GGUF loader authority merged. |
| HW-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Shared hardware validation matrix and proof-stage contract merged. |
| HW-002 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Hardware artifact naming policy merged. |
| BITNET-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | BitNet model/kernel/receipt proof contract merged. |
| BITNET-002 | #3628 | 1ac71e0ffc0c71584d3f3e3ec42143a40132ef83 | Canonical BitNet fixture manifest and hardware receipt templates merged. |
| BITNET-003 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Initial BitNet parity tolerance policy merged. |
| NPU-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Intel NPU backend lane merged. |
| A770-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Intel Arc A770 backend lane merged. |
| A770-002 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Intel Arc A770 machine profile merged. |
| ARC140V-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Intel Arc 140V integrated GPU lane merged. |
| LNL258V-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Core Ultra 7 258V platform profile merged. |
| CPU258V-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Core Ultra 7 258V CPU AVX2 validation lane merged. |
| KBL8250U-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Core i5-8250U CPU validation lane merged. |
| KBL8250U-002 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Core i5-8250U machine profile and probe bundle merged. |
| M4-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | Apple M4 Mac mini backend lane merged. |
| M4-002 | #3627 | 6362101c3249a0758100d22c31e084eba37c387b | Apple M4 Mac mini machine profile and probe bundle merged; docs/artifact prep only, no runtime execution. |
| M4-003 | #3652 | 849c3db73b786483bb7955371b95f733235119bb | Apple M4 backend identity merged; no Metal kernels, MPSGraph execution, QK256, server inference, dependencies, runtime probes, or hardware artifacts. |
| RTX5070TI-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | NVIDIA RTX 5070 Ti CUDA backend lane merged. |
| RTX5070TI-002 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | NVIDIA RTX 5070 Ti machine profile and probe bundle merged. |
| AMD9950X3D-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | AMD Ryzen 9 9950X3D CPU validation lane merged. |
| AMD9950X3D-002 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | AMD Ryzen 9 9950X3D machine profile and probe bundle merged. |
| AMD5700X-001 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | AMD Ryzen 7 5700X CPU validation lane merged. |
| AMD5700X-002 | #3625 | bbc5d563ce22c4a81e517992120c4ad5d8a6d0d3 | AMD Ryzen 7 5700X machine profile and probe bundle merged. |

## Blocked

| Item | Blocker | Next action |
|---|---|---|

## Superseded

| Item/PR | Superseded by | Reason |
|---|---|---|
