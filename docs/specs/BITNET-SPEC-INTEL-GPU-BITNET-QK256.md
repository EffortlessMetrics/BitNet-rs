# BITNET-SPEC-INTEL-GPU-BITNET-QK256

## Purpose

Define native Intel GPU BitNet QK256/I2_S proof, starting with A770 OpenCL.

## Required kernels and operations

- `qk256_i2s_gemv_opencl`
- `embedding_lookup_opencl`
- `lm_head_tied_logits_opencl`
- eventual `qk256_i2s_prefill_gemm_opencl`

## Production semantics

Claim-grade proof must match:

- Official Microsoft I2_S/QK256 GGUF weights.
- Canonical packed layout.
- BitNet.cpp-aligned activation quantization.
- I2_S by I8_S scaled math.
- Weight scale handling.
- Activation scale and sum correction.
- Tail-column behavior.
- Row-stride behavior.
- Strict tokenizer/template authority.

## Hard rule

A diagnostic four-values-per-byte toy I2_S kernel cannot satisfy official QK256
proof. Toy parity may support a smoke or fixture claim only when the route row
and receipt say it is not official BitNet QK256 proof.
