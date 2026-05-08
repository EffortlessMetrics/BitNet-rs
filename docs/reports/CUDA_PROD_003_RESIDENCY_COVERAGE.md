# CUDA-PROD-003 Execution Residency Coverage

## Summary

`CUDA-PROD-003` makes the strict RTX 5070 Ti CUDA answer receipts explicit
about execution residency. The existing proof already records that the BitNet
QK256 linear path routes through `qk256_gemv_cuda` with no BitNet linear CPU
fallback and upload-once CUDA weight handles. This follow-up adds a
`cuda_execution_residency` section so the same receipts also expose the phases
that are CUDA-resident, CPU-resident, or not yet measured.

The change is receipt-only visibility. It does not change CUDA kernels,
QK256 dispatch, transformer math, tokenizer behavior, model loading, or server
behavior.

## Receipt Contract

Strict CUDA `run`, strict CUDA `ask`, and `cuda-warm-session` receipts now carry
the same residency contract:

```text
cuda_execution_residency.full_cuda_residency_claimed = false
cuda_execution_residency.speedup_claim = false
cuda_execution_residency.qk256_bitnet_linears.kernel_id = qk256_gemv_cuda
cuda_execution_residency.weight_residency.scope = qk256_cuda_weight_handles_only
cuda_execution_residency.host_device_transfer_accounting.status = not_measured
```

The section records:

- QK256 BitNet linear CUDA invocation coverage.
- BitNet linear CPU fallback count.
- upload-once QK256 CUDA weight-handle status.
- KV-cache residency and reuse policy.
- explicit non-residency / not-yet-claimed status for embeddings, norms, RoPE,
  attention, ReLU2, LM head, sampling, transfer bytes, and kernel timings.
- claim boundaries for full transformer CUDA residency and speedup.

## Claim Boundary

Allowed claim:

- Strict RTX 5070 Ti CUDA answer receipts expose which decode phases are
  CUDA-resident, CPU-resident, transfer-accounted, or not yet claimed.
- QK256 BitNet linear CUDA coverage and upload-once weight residency remain
  visible in the normal strict answer path.

Not allowed:

- CUDA speedup.
- Broad chat quality beyond committed deterministic prompts.
- Full CUDA residency for every transformer operation.
- Production server readiness.

## Next Work

This PR makes the remaining residency gaps visible. Later work can move hot
phases deliberately, starting with the highest-cost non-resident or unmeasured
items shown by receipts and benchmarks: KV-cache residency, RoPE, norms,
attention/softmax, ReLU2, LM head, and transfer/timing instrumentation.
