# CUDA-DENSE-001 Receipt Boundary

`CUDA-DENSE-001` starts the dense regular-LLM CUDA lane by adding a receipt
contract, not dense CUDA math.

The dense lane is intentionally separate from the completed RTX 5070 Ti BitNet
packed I2_S/QK256 proof lane. Dense regular-LLM CUDA work may share device
identity, CUDA runtime plumbing, timing fields, transfer accounting, and receipt
validation patterns, but it cannot satisfy BitNet packed-kernel proof gates.

## Receipt Label

Dense regular-LLM CUDA receipts use:

```json
{
  "artifact_kind": "dense_regular_llm_cuda",
  "hardware_lane": "nvidia-rtx-5070-ti-cuda",
  "requested_backend": "nvidia-rtx-5070-ti-cuda",
  "selected_backend": "nvidia-rtx-5070-ti-cuda",
  "runtime_api": "cuda",
  "fallback_used": false,
  "speedup_claim": false,
  "execution_path": {
    "model_class": "dense_regular_llm",
    "bitnet_packed_kernel_proof": false,
    "qk256_proof": false
  },
  "claim_boundary": {
    "dense_regular_llm_cuda_claimed": true,
    "bitnet_packed_i2s_qk256_proof": false,
    "speedup_claim": false,
    "full_cuda_residency_claimed": false
  }
}
```

The validator rejects dense receipts that identify the model family, kernel
family, quantization family, or kernel ID as BitNet packed I2_S/QK256 proof.

## Allowed Claim

- A receipt can identify a dense regular-LLM CUDA lane as
  `dense_regular_llm_cuda`.
- A receipt can preserve RTX 5070 Ti CUDA backend identity, fallback status, and
  measured kernel/transfer fields when present.
- A receipt can prove it is not BitNet packed-kernel evidence.

## Forbidden Claim

- Dense CUDA evidence does not prove BitNet packed I2_S inference.
- Dense CUDA evidence does not prove QK256 inference.
- `CUDA-DENSE-001` does not prove dense CUDA GEMM parity.
- `CUDA-DENSE-001` does not prove dense CUDA speedup.
- `CUDA-DENSE-001` does not prove full dense CUDA residency.

## Next Item

`CUDA-DENSE-002` owns the first dense CUDA FP16/BF16 or cuBLAS-backed GEMM
smoke/parity fixture. It should use the `dense_regular_llm_cuda` receipt
boundary added here, compare against a CPU reference, keep `fallback_used=false`,
and keep `speedup_claim=false`.
