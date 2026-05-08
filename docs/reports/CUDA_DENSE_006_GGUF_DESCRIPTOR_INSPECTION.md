# CUDA-DENSE-006 GGUF Descriptor Inspection

`CUDA-DENSE-006` adds a descriptor-only bridge from dense GGUF metadata into the
dense CUDA planning lane. It does not add dense GGUF inference, CUDA execution,
new kernels, or speedup claims.

## What Changed

- Added a dense GGUF descriptor inspector over the existing GGUF reader.
- Classified common dense tensor roles for Qwen/Llama-style GGUF names:
  embeddings, output, attention Q/K/V/O, MLP gate/up/down, and norm tensors.
- Rejected BitNet packed markers and `I2_S` / `IQ2_S` tensors before they can
  become dense descriptor evidence.
- Added a descriptor-only receipt validator for
  `dense_gguf_tensor_descriptor_inspection`.
- Recorded a normalized Qwen-family GGUF reader fixture receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-gguf-descriptor-inspection.json
```

## Descriptor Boundary

The committed fixture is intentionally synthetic and reader-scoped. It proves
that the GGUF reader path can classify required dense tensor roles and record
Q8_0 as descriptor-only evidence.

For Q8_0 dense GGUF tensors, the route status is:

```text
descriptor_only_quant_bridge_required
```

That means a future PR must add a real dense quant bridge before strict dense
CUDA routing or dense GGUF inference can be claimed.

## Claim Boundary

May claim:

- Dense GGUF tensor descriptor inspection exists for Qwen-family tensor roles.
- Q8_0 dense GGUF descriptors are visible but require a future quant bridge.
- Descriptor receipts are rejected as BitNet packed I2_S/QK256 proof.

Must not claim:

- Dense GGUF inference works.
- Dense CUDA execution ran for a GGUF model.
- Dense CUDA speedup exists.
- Dense CUDA proves BitNet packed I2_S or QK256 inference.
- Full dense CUDA residency is proven.
- Tokenizer, loader semantics, transformer behavior, or server behavior changed.

## Next Step

`CUDA-DENSE-007` should inspect real dense GGUF tensor descriptors, then build a
single real dense linear CPU/CUDA parity fixture from those descriptors. It
should still avoid claiming full dense GGUF inference until one-token and short
decode receipts exist.
