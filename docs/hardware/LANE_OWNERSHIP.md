# Hardware Lane Ownership

## Purpose

Each hardware lane owns a narrow proof path. Ownership prevents CPU, GPU, OpenVINO GPU, and OpenVINO NPU work from being mixed into ambiguous claims.

## CPU Work

Owns:

- 258V CPU `cpu-avx2` as the BitNet CPU lead for strict real-GGUF runs, scalar/AVX2 validation, answer parity, phase receipts, and same-machine comparison against Arc 140V and Intel NPU.
- i5-8250U `cpu-scalar` and `cpu-avx2` as the SLM CPU lead plus legacy/low-power BitNet comparison lane.
- Ryzen 7 5700X `cpu-scalar` and `cpu-avx2` as a BitNet support validation lane when it does not collide with A770 work.
- Ryzen 9 9950X3D `cpu-scalar`, `cpu-avx2`, and `cpu-avx512` as a BitNet support validation lane when it does not collide with RTX 5070 Ti work.
- CPU feature detection and selected CPU kernel-path receipts.
- Strict CPU proof runs.
- Sustained CPU baselines with power and thermal context.
- Cache-domain and scheduler context for X3D-sensitive CPU benchmarks.

Does not own:

- OpenCL kernels.
- OpenVINO GPU.
- OpenVINO NPU.
- A770 or Arc 140V selected-device identity.
- Treating CPU proof as GPU/NPU acceleration proof.

## CPU No-Trample Rule

The 258V CPU lane owns BitNet CPU sequencing. It leads strict real-GGUF BitNet CPU runs, scalar/AVX2 answer parity, phase receipts, and same-machine comparison against Arc 140V and Intel NPU artifacts. The 8250U lane no longer blocks BitNet CPU work; it owns SLM CPU work and remains useful for legacy/low-power BitNet comparison artifacts.

The 8250U, 258V, 5700X, and 9950X3D CPU lanes may all perform CPU validation, but they must not edit the same runtime surface in overlapping PRs. 9950X3D may validate BitNet CPU changes when that does not collide with RTX 5070 Ti work. 5700X may validate BitNet CPU changes when that does not collide with A770 work.

If a BitNet CPU change touches shared dispatch, QK256 CPU kernels, or inference hot paths, the ledger item must name the 258V lane as the owning BitNet CPU lane unless the item explicitly transfers ownership. Other CPU machines should be listed as validation targets, not co-owners.

PR scoping:

```text
258V BitNet CPU PR:
  may touch CPU detect, CPU kernels, scalar/AVX2 dispatch, answer parity, phase receipts
  must not touch Arc 140V OpenCL, A770 OpenCL, Intel NPU execution, Metal, CUDA

8250U SLM CPU PR:
  may touch SLM CPU proof, SLM receipts, and low-power comparison artifacts
  must not block new BitNet CPU implementation leadership

Accelerator PR:
  must not alter CPU dispatch or QK256 CPU kernels unless explicitly scoped
```

## Intel Arc GPU Work

Owns:

- A770 OpenCL.
- A770 OpenVINO GPU reference.
- Arc 140V OpenCL.
- Arc 140V OpenVINO GPU reference.
- OpenCL and Level Zero probes for Intel GPUs.
- OpenVINO `GPU.X` selected-device receipts.
- GPU kernel smoke, parity, and benchmark artifacts.

Does not own:

- NPU OpenVINO.
- CPU AVX2 runtime proof.
- CPU QK256 optimization.
- Full inference claims from OpenVINO GPU smoke alone.

## Intel NPU Work

Owns:

- 258V NPU OpenVINO visibility.
- OpenVINO `NPU` compile smoke.
- Static-shape NPU graph smoke.
- Selected static BitNet subgraph experiments.
- NPU driver/compiler/OpenVINO/shape/cache receipt fields.

Does not own:

- A770 OpenCL.
- Arc 140V OpenCL.
- CPU QK256 kernel optimization.
- Full decode until static graph and subgraph receipts exist.

## Apple Silicon Work

Owns:

- M4 Metal device visibility.
- M4 native Metal compute smoke.
- M4 CPU/NEON fallback and CPU/Metal parity.
- M4 MPSGraph reference smoke.
- Apple chip, GPU core count, unified memory, memory bandwidth, selected backend, and fallback receipt fields.

Does not own:

- CUDA work.
- Intel OpenCL work.
- OpenVINO NPU work.
- Neural Engine claims from MPSGraph smoke unless the resolved target is receipt-backed.
- Treating MPSGraph proof as native Metal kernel proof.

## NVIDIA CUDA Work

Owns:

- RTX 5070 Ti CUDA runtime and NVML probes.
- CUDA selected-device identity.
- CUDA kernel smoke.
- CPU/CUDA parity.
- CUDA benchmark artifacts with driver, CUDA version, compute capability, VRAM, power, and thermal context.
- wgpu/Vulkan/D3D12 reference smoke as a separate comparison lane.

Does not own:

- Apple Metal or MPSGraph.
- Intel Arc OpenCL.
- OpenVINO NPU.
- Treating wgpu smoke as CUDA kernel proof.
- Generic GPU claims without selected CUDA device identity.

## Platform Work

Owns:

- 258V whole-machine profile.
- Cross-device comparison on the same laptop.
- Shared memory, power, and thermal context.
- Mapping CPU, Arc 140V, and NPU receipts to separate proof lanes.

Does not own:

- Replacing lane-specific proof rules.
- Merging CPU/GPU/NPU claims.
- Calling one device's fallback another device's success.

## Practical Rule

If a change cannot name its requested backend, selected backend, runtime API, resolved device identity, fallback status, and artifact path, it is not ready to claim hardware proof.
