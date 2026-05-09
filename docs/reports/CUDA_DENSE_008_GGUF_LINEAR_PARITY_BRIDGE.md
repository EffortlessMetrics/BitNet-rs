# CUDA-DENSE-008: Dense GGUF Linear Parity Bridge

## Summary

`CUDA-DENSE-008` adds the first bridge from descriptor-extracted dense GGUF
linear fixture data into the existing strict CUDA FP16 GEMM lane.

The bridge accepts dense linear data already materialized by the model layer:

```text
weights: row-major [out, in] F32
input:   [in] F32
```

It converts that fixture into the GEMM layout used by the existing CUDA kernel:

```text
A = input[1, in] as F16
B = transpose(weights)[in, out] as F16
C = output[1, out] as F32
```

The CPU reference for this lane is the same FP16 bridge layout, not the earlier
F32 matvec extraction output. That keeps CPU/CUDA parity scoped to the data type
the CUDA kernel actually consumes.

## Added

- `DenseGgufLinearGemmFixture`, a kernel-layer input struct that avoids adding a
  `bitnet-models` dependency to `bitnet-kernels`.
- `prepare_dense_gguf_linear_f16_gemm`, which validates dense fixture metadata
  and performs the `[out, in] -> [in, out]` transpose.
- `dense_gguf_linear_f16_gemm_cpu_reference`, the FP16 CPU bridge reference.
- `run_dense_gguf_linear_f16_cuda_parity`, an optional strict CUDA parity entry
  point over the existing `dense_f16_gemm_cuda` kernel.
- `dense_gguf_linear_cuda_parity` receipt validation.

## Claim Boundary

May claim:

- dense GGUF single-linear fixture data can be routed into the existing FP16 GEMM
  CUDA layout;
- the receipt contract can validate fallback-free single-linear dense CUDA
  parity against the FP16 CPU bridge;
- dense GGUF linear CUDA parity remains separate from BitNet packed I2_S/QK256
  proof.

Must not claim:

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- full CUDA residency;
- tokenizer, prompt-template, transformer, QK256, or server behavior changed.

## Live Receipt Status

This PR does not commit a live dense GGUF CUDA execution receipt. The live RTX
5070 Ti test is environment-gated:

```powershell
$env:BITNET_RUN_RTX5070TI_DENSE_GGUF_LINEAR_CUDA_PARITY = "1"
$env:BITNET_RTX5070TI_DENSE_GGUF_LINEAR_CUDA_PARITY_RECEIPT = "target/bitnet/receipts/dense-gguf-linear-cuda-parity.json"
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda live_rtx5070ti_dense_gguf_linear_cuda_parity_when_enabled -- --nocapture
```

The committed validation is the bridge, shape/marker guards, and receipt
contract. A future lane should connect this bridge to a real downloaded dense
GGUF artifact and then record a live receipt.
