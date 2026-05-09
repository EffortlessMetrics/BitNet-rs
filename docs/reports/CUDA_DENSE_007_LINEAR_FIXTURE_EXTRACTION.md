# CUDA-DENSE-007 Linear Fixture Extraction

`CUDA-DENSE-007` adds the next dense GGUF bridge after descriptor inspection:
one recognized dense linear tensor can be selected from a GGUF reader,
materialized as F32, and evaluated through a deterministic CPU reference
matvec. It does not add dense GGUF inference or CUDA execution.

## What Changed

- Added descriptor-driven dense GGUF linear fixture extraction.
- Supports F32, F16, and GGML Q8_0 tensor payloads for CPU-reference fixture
  materialization.
- Preserves the GGUF dense projection convention where source dims are
  interpreted as `[in, out]` and the fixture matrix is `[out, in]`.
- Computes deterministic CPU-reference input and output hashes for future
  CPU/CUDA parity fixture work.
- Added a receipt validator for
  `dense_gguf_linear_fixture_extraction`.
- Recorded a normalized synthetic Qwen-family Q8_0 fixture receipt:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/dense-gguf-linear-fixture-extraction.json
```

## Fixture Boundary

The committed receipt is intentionally synthetic and reader-scoped. It proves
the code path can extract one dense linear tensor and compute the CPU reference
side of a parity fixture.

It does not prove:

- a real Qwen GGUF artifact was loaded locally;
- dense CUDA GEMM consumed a GGUF-derived tensor;
- dense GGUF one-token inference works;
- dense GGUF short decode works.

## Claim Boundary

May claim:

- Dense GGUF linear fixture extraction exists for Qwen-family tensor roles.
- Q8_0 dense GGUF tensors can be materialized as F32 for CPU-reference fixture
  work.
- Linear fixture receipts are rejected as BitNet packed I2_S/QK256 proof.

Must not claim:

- Dense GGUF inference works.
- Dense CUDA execution ran for a GGUF model.
- CPU/CUDA dense GGUF parity is proven.
- Dense CUDA speedup exists.
- Dense CUDA proves BitNet packed I2_S or QK256 inference.
- Full dense CUDA residency is proven.
- Tokenizer, loader semantics, transformer behavior, or server behavior
  changed.

## Next Step

`CUDA-DENSE-008` should use a real dense GGUF artifact when available and route
one extracted dense linear fixture through the dense CUDA GEMM path with
CPU/CUDA parity evidence. That still must remain below full dense GGUF
inference until one-token and short-decode receipts exist.
