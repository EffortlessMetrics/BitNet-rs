# CUDA-DENSE-043 One-Token Runtime Proof Gate

`CUDA-DENSE-043` is the first dense CUDA runtime proof gate after
`CUDA-DENSE-042` added the fail-closed receipt validator. It is deliberately
separate from the validator slice: the next implementation must run one real,
deterministic Qwen token through the dense CUDA route and commit the hardware
receipt only if the validator accepts it.

## Required Runtime Evidence

The future `dense_gguf_qwen_one_token_strict_cuda_proof` receipt must prove:

- the cached `qwen2.5-0.5b-instruct-q8_0` artifact matches the pinned SHA-256;
- the all-layer plan, model-boundary fixture, KV-cache policy, and
  sampling-policy prerequisite receipts are linked by hash or path;
- exactly one deterministic greedy token executed through
  `dense_regular_llm_cuda` on `nvidia-rtx-5070-ti-cuda`;
- `fallback_used=false`, no CPU fallback ops, and no mixed CUDA route;
- CPU and CUDA selected token IDs match;
- CPU and CUDA logits/top-k evidence hashes match or record a governed
  first-divergence diagnostic;
- kernel, residency, transfer, and timing evidence is present.

## Non-Claims

This gate does not claim:

- Qwen short decode or chat works on CUDA;
- dense GGUF inference is generally complete;
- CUDA speedup is benchmark-qualified;
- persistent-session or full dense CUDA residency exists;
- dense regular-LLM CUDA proves BitNet packed I2S/QK256;
- tokenizer, prompt-template, loader, transformer runtime, server, QK256,
  BitNet CUDA, or CUDA kernel math changed.

## Local Preflight

The 5070 Ti machine already has the pinned Qwen artifact cached under
`%LOCALAPPDATA%\bitnet-rs\models`, and `bitnet model verify
qwen2.5-0.5b-instruct-q8_0 --json` passes. That is only artifact authority; it
is not a one-token CUDA runtime proof.

## Next Implementation Slice

The implementation PR should add a narrow CLI/operator command for the strict
one-token proof, emit the hardware receipt under
`ci/hardware/windows-9950x3d-rtx5070ti/`, validate it with
`validate_dense_gguf_qwen_one_token_strict_cuda_proof_receipt_json`, and keep
all broader dense CUDA claims false.
