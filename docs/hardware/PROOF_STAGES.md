# Hardware Proof Stages

## Purpose

Use the same proof-stage words in every hardware lane. This prevents detection, smoke, parity, inference, and performance claims from being mixed.

## Stage Ladder

| Stage | Meaning | Claim allowed |
|---|---|---|
| `detected` | OS sees hardware | Device detected |
| `runtime_detected` | OpenCL, OpenVINO, or Level Zero can enumerate the device | Runtime sees device |
| `compile_smoke` | Kernel or graph compiles for the selected device | Compile path works |
| `kernel_smoke_tested` | Tiny kernel or graph executes | Runtime execution works |
| `parity_tested` | CPU reference and device output match tolerance | This kernel/subgraph works |
| `receipt_backed` | Artifact records selected backend and no hidden fallback | This path is proven |
| `benchmark_backed` | Receipt-backed benchmark exists | This path is faster/slower under stated conditions |

## Hard Rules

```text
Detection is not execution.
Execution is not parity.
Parity is not full inference.
Full inference is not performance.
Performance is not portable without machine context.
```

## Lane Examples

### i5-8250U CPU

- `detected`: CPU model is i5-8250U.
- `runtime_detected`: CPU feature probe records AVX2 and no AVX-512.
- `kernel_smoke_tested`: scalar and AVX2 kernels execute.
- `parity_tested`: scalar and AVX2 outputs match CPU reference expectations.
- `benchmark_backed`: cold turbo and sustained mobile baselines are recorded separately.

### AMD 5700X CPU

- `detected`: CPU model is Ryzen 7 5700X.
- `runtime_detected`: CPU feature probe records AVX2 and no AVX-512.
- `kernel_smoke_tested`: scalar and AVX2 kernels execute.
- `parity_tested`: scalar and AVX2 outputs match CPU reference expectations.
- `benchmark_backed`: DDR4/AM4 sustained desktop baseline is recorded.

### AMD 9950X3D CPU

- `detected`: CPU model is Ryzen 9 9950X3D.
- `runtime_detected`: CPU feature probe records AVX2 and AVX-512.
- `kernel_smoke_tested`: scalar, AVX2, and AVX-512 kernels execute.
- `parity_tested`: scalar, AVX2, and AVX-512 outputs match CPU reference expectations.
- `benchmark_backed`: cache-domain, scheduler/core placement, cooling, and sustained-power context are recorded.

### Arc A770

- `runtime_detected`: OpenCL sees Intel Arc A770.
- `compile_smoke`: OpenCL program compiles for A770.
- `kernel_smoke_tested`: tiny OpenCL kernel executes on A770.
- `parity_tested`: `matmul_i2s` or equivalent output matches CPU.
- `benchmark_backed`: receipt records ReBAR, PCIe link, VRAM, driver, power, and timing context.

### Arc 140V

- `runtime_detected`: OpenCL or Level Zero sees Arc 140V.
- `kernel_smoke_tested`: tiny OpenCL kernel executes on Arc 140V.
- `parity_tested`: one kernel/subgraph matches CPU.
- `benchmark_backed`: shared-memory pressure and laptop power context are recorded.

### 258V NPU

- `runtime_detected`: OpenVINO enumerates `NPU`.
- `compile_smoke`: OpenVINO compiles a static-shape graph to `NPU`.
- `kernel_smoke_tested`: tiny static graph executes on `NPU`.
- `parity_tested`: one static BitNet subgraph matches CPU within tolerance.
- `benchmark_backed`: OpenVINO cache, first-infer, steady-state, shape, power, and driver/compiler context are recorded.

### Apple M4

- `runtime_detected`: Metal sees an Apple M4 GPU device.
- `compile_smoke`: Metal compute pipeline compiles.
- `kernel_smoke_tested`: tiny Metal dispatch executes.
- `parity_tested`: CPU/NEON and Metal outputs match tolerance.
- `benchmark_backed`: macOS, chip, unified memory, selected backend, and fallback context are recorded.

### RTX 5070 Ti

- `runtime_detected`: CUDA runtime sees RTX 5070 Ti.
- `compile_smoke`: CUDA kernel compiles for compute capability 12.0.
- `kernel_smoke_tested`: tiny CUDA kernel executes.
- `parity_tested`: CPU and CUDA outputs match tolerance.
- `benchmark_backed`: driver, CUDA version, compute capability, VRAM, power, and thermal context are recorded.
