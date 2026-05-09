# CUDA-DENSE-009: Extracted Dense GGUF Linear Parity

## Summary

`CUDA-DENSE-009` adds a CLI/operator harness for running one real dense GGUF
linear fixture through the existing dense FP16 GEMM CUDA parity bridge.

The command performs the full diagnostic chain for a single linear tensor:

```text
dense GGUF path
  -> bitnet-models GGUF reader
  -> descriptor-driven dense linear fixture extraction
  -> CUDA-DENSE-008 FP16 GEMM bridge layout
  -> strict RTX 5070 Ti CUDA parity run
  -> dense_gguf_linear_cuda_parity receipt validation
```

This is still a single-linear diagnostic. It does not load a full dense model
graph, run a Qwen token, or claim dense GGUF inference.

## Command

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-linear-parity `
  --model <dense-gguf-path> `
  --role attention_q `
  --device-index 0 `
  --json-out target/bitnet/receipts/dense-gguf-linear-cuda-parity.json
```

The command requires the selected CUDA device to be the RTX 5070 Ti proof lane.
It fails closed when the CUDA probe is unavailable or selects a different
device.

## Claim Boundary

May claim:

- the CLI can extract one dense GGUF linear fixture through `bitnet-models`;
- that fixture can be converted to the existing FP16 GEMM CUDA bridge layout;
- the emitted receipt validates fallback-free dense single-linear CUDA parity
  and preserves the dense `execution_plan` route;
- the receipt remains rejected as BitNet packed I2_S/QK256 proof.

Must not claim:

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA proves BitNet packed inference;
- speedup;
- full CUDA residency;
- tokenizer, prompt-template, transformer, QK256, or server behavior changed.

## Live Receipt Status

This PR adds the operator command and unit-level receipt validation. It does not
commit a live dense GGUF CUDA execution receipt. A future lane can run the
command against a downloaded dense GGUF artifact on the Windows 9950X3D + RTX
5070 Ti host and commit the normalized receipt once the artifact path and model
authority are governed.
