# Hardware Benchmark Protocol

## Purpose

Benchmark claims require machine context. Hardware detection, runtime visibility, smoke, parity, and inference proof are not performance proof.

## Minimum Fields

Every benchmark artifact must record:

- `machine_id`
- OS, kernel, or build.
- Driver and runtime versions.
- Requested backend.
- Selected backend.
- Runtime API.
- Resolved device identity.
- Fallback status and fallback reason.
- Power mode.
- Thermal state when available.
- Cold run versus warm run.
- First-token, first-infer, or first-dispatch timing where relevant.
- Steady-state timing where relevant.
- Model, graph, kernel, and cache state.
- Artifact path.

## Lane-Specific Fields

### A770

Record:

- ReBAR status.
- PCIe link width/generation.
- VRAM bytes.
- Driver version.
- OpenCL platform and device name.
- OpenCL device index when available.
- Level Zero visibility.
- `xpu-smi` utilization/power if available.

### Arc 140V

Record:

- Shared-memory pressure.
- Battery or AC mode.
- Power plan.
- Thermal profile.
- OpenCL platform and device name.
- OpenVINO `GPU.0` identity for reference runs.

### NPU

Record:

- Shape mode.
- Input/output shapes.
- OpenVINO cache directory.
- First-ever compile/inference latency.
- Cached compile latency.
- First inference latency.
- Steady-state inference latency.
- Driver version.
- Compiler version.
- Runtime device `NPU`.

### i5-8250U

Record:

- Cold turbo performance separately from warm sustained performance.
- CPU frequency during run when available.
- Temperature when available.
- Governor or Windows power plan.
- AC versus battery when available.
- Duration.

### Ryzen 7 5700X

Record:

- Scalar versus AVX2 path.
- AVX-512 absence.
- DDR4/AM4 memory context.
- CPU frequency during run when available.
- Temperature when available.
- Governor or Windows power plan.
- Duration.

### Ryzen 9 9950X3D

Record:

- Scalar, AVX2, and AVX-512 paths separately.
- X3D/cache-domain context when available.
- Scheduler and core-placement context when available.
- DDR5/AM5 memory context.
- Sustained frequency.
- Temperature when available.
- Cooling context.
- Power mode.
- Duration.

### M4 Mac mini

Record:

- macOS version.
- Apple chip name.
- Base M4 versus M4 Pro configuration.
- CPU and GPU core counts.
- Unified memory size.
- Memory bandwidth target.
- Runtime API, such as Metal or MPSGraph.
- Resolved MPSGraph target when available.
- Cold and warm timing.

### RTX 5070 Ti

Record:

- NVIDIA driver version.
- CUDA version.
- Compute capability.
- VRAM bytes.
- Power limit and draw when available.
- GPU temperature when available.
- CUDA selected device index/name.
- Optional wgpu backend API and adapter name for reference runs.

## Claim Rules

- A benchmark without fallback status cannot support a hardware claim.
- A benchmark without machine context cannot support a portable claim.
- A short turbo result on mobile CPU cannot be reported as sustained performance.
- CPU scalar, AVX2, and AVX-512 results must be reported as distinct selected backends.
- 5700X results cannot support AVX-512 claims.
- 9950X3D cache-sensitive results need cache-domain and sustained-power context.
- A770 benchmarks need ReBAR/PCIe/VRAM context before performance claims.
- Arc 140V benchmarks need shared-memory and power context.
- NPU benchmarks need OpenVINO cache and static-shape context.
- M4 benchmarks need exact chip, unified memory, and Metal/MPSGraph distinction.
- RTX 5070 Ti benchmarks need driver, CUDA, compute capability, VRAM, power, and thermal context.

## Artifact Kinds

Use the artifact naming policy in `ci/hardware/README.md`.

BitNet benchmarks must also follow:

```text
docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md
```

Any benchmark claiming BitNet progress must include model, tokenizer, quantization, kernel family, execution phase, reference path, and fallback fields from `docs/bitnet/BITNET_RECEIPT_FIELDS.md`.

Benchmark artifacts should use:

```text
benchmark.json
```

or a more specific name:

```text
matmul-i2s-benchmark.json
npu-static-graph-benchmark.json
strict-cpu-sustained-benchmark.json
```
