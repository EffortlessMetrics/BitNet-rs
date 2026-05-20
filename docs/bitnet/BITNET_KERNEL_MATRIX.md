# BitNet Kernel Matrix

## Purpose

This matrix defines which BitNet kernel formats are valid targets for each hardware lane. Hardware proof does not imply support for every BitNet layout.

## Kernel Formats

| Kernel family | Meaning | Valid first lanes | Not for |
|---|---|---|---|
| `i2_s` | Portable packed baseline | x86, ARM | Performance claim without receipt |
| `tl1` | ARM lookup-table path | M4 / ARM64 | x86 |
| `tl2` | x86 lookup-table path | 8250U, 258V CPU, 5700X, 9950X3D | ARM |
| `qk256` | Repo-local packed/dispatch path with precise scalar F32/no-scale and scaled I8_S IDs | CPU first, GPU later | Unproven accelerator claims or ambiguous scalar receipts |
| `openvino_graph` | Converted/static graph path | OpenVINO GPU/NPU references | Native packed kernel proof |

## Hard Rules

### x86 CPU Lanes

Valid targets:

```text
i2_s
tl2
qk256, after scalar parity
```

Invalid target:

```text
tl1
```

Applies to:

- `intel-i5-8250u-cpu-avx2`
- `intel-258v-cpu-avx2`
- `amd-5700x-cpu-avx2`
- `amd-9950x3d-cpu-avx512`

Kernel family and kernel implementation are different fields. For example:

| Kernel family | Example requested kernel | Meaning |
|---|---|---|
| `qk256` | `qk256-scalar-f32-gemv` | Scalar no-scale F32 decode diagnostic/reference kernel; compatibility alias: `qk256-scalar-gemv`. |
| `qk256` | `qk256-scalar-f32-gemm` | Scalar no-scale F32 prefill diagnostic/reference kernel; compatibility alias: `qk256-scalar-gemm`. |
| `qk256` | `qk256-scalar-i8s-scaled-gemv` | Scalar BitNet.cpp-style scaled I2_S x I8_S decode oracle. |
| `qk256` | `qk256-scalar-i8s-scaled-gemm` | Scalar BitNet.cpp-style scaled I2_S x I8_S prefill oracle. |
| `qk256` | `qk256-avx2-gemv` | AVX2 decode-first packed GEMV |
| `qk256` | `qk256-avx512-f32-gemv` | AVX-512 no-scale F32-style decode GEMV |
| `qk256` | `qk256-avx512-i8s-scaled-gemv` | AVX-512 BitNet I2_S × I8_S inline-scale decode GEMV |
| `qk256` | `qk256-avx512-i8s-scaled-gemm` | AVX-512 BitNet I2_S × I8_S inline-scale prefill GEMM |
| `qk256` | `qk256-neon-gemv` | ARM64 NEON decode-first packed GEMV |

AVX-512 is a wider x86 lane, not a separate proof family. It must inherit
scalar packed parity and compare against AVX2 before speed claims. AVX-512
detection or an AVX-512 receipt label is not optimized AVX-512 kernel proof;
strict receipts must prove selected kernel ID, fallback status, and AVX-512
hot-path invocation counters.

Strict receipts must record both `kernel_family` and requested/selected kernel IDs.

### ARM / M4 Lane

Valid targets:

```text
i2_s
tl1
scalar/NEON reference
```

Invalid target:

```text
tl2
```

Applies to:

- `apple-m4-cpu-neon`
- `apple-m4-metal` for CPU/Metal parity reference

### NPU / OpenVINO Lane

Valid first targets:

```text
openvino_graph
static FP16/INT8/INT4/NF4 graph smoke
selected static subgraph experiments
```

Invalid claims without further proof:

```text
I2_S native support
TL1 native support
TL2 native support
QK256 native support
full decode
```

OpenVINO graph smoke does not prove packed BitNet kernel support.

### Native GPU Lanes

Valid first targets:

```text
tiny native compute smoke
CPU/GPU parity
I2_S or QK256-adjacent kernels only when the layout contract is explicit
```

Native GPU kernels must declare whether they:

- consume `I2_S` directly,
- consume `QK256` directly,
- consume canonical QK256/I2_S layout from `bitnet-qk256-layout-core`,
- convert/dequantize before compute,
- or run a graph/reference path.

OpenVINO GPU, WGPU, MPSGraph, or other graph/shader smoke cannot be used as native packed-kernel proof.

## Hardware-To-BitNet Operation Map

| Lane | BitNet work it may own first | Must not claim |
|---|---|---|
| 8250U / 5700X / 258V CPU | scalar, AVX2, I2_S, TL2, QK256 CPU | GPU/NPU acceleration |
| 9950X3D | AVX2 vs AVX-512, TL2/QK256 CPU, cache-sensitive decode | GPU/NPU acceleration |
| M4 CPU | scalar/NEON, I2_S/TL1 reference | x86 TL2 |
| M4 Metal | native Metal kernels, CPU/Metal parity | ANE/NPU unless MPSGraph proves target |
| A770 | native OpenCL I2_S/QK256-adjacent kernels | OpenVINO graph proof as native kernel proof |
| Arc 140V | small OpenCL kernels, CPU/iGPU comparison | A770-equivalent performance |
| RTX 5070 Ti | CUDA kernels, CPU/CUDA parity | WGPU/Vulkan proof as CUDA proof |
| 258V NPU | static OpenVINO graph smoke/subgraph only | packed QK256/full decode until proven |

## Kernel Receipt Fields

Every kernel proof must record:

```json
{
  "bitnet": {
    "kernel_family": "i2_s|tl1|tl2|qk256|openvino_graph",
    "kernel_format": "i2_s",
    "layout": "...",
    "fallback_layout": null,
    "dequantizes_before_compute": false
  }
}
```

## Related Docs

- `docs/bitnet/BITNET_CPU_PATH_PLAN.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
- `docs/specs/BITNET-SPEC-CPU-SCALAR-KERNEL-CONTRACT.md`
- `docs/specs/BITNET-SPEC-CPU-SCALAR-HOTPATH.md`
- `docs/specs/BITNET-SPEC-CPU-SCALAR-PARITY.md`
- `docs/specs/BITNET-SPEC-CPU-SCALAR-PERFORMANCE.md`
