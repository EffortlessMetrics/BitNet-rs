# CUDA-DENSE-041 One-Token Strict CUDA Proof Contract

`CUDA-DENSE-041` is the next dense CUDA proof gate after
`CUDA-DENSE-040` made logits transfer and deterministic sampling policy
explicit. It defines the governed one-token proof contract only; it does not
implement token generation.

The implementation that satisfies this contract must consume the existing
receipt-backed gates for the verified Qwen2.5 0.5B Q8_0 dense GGUF artifact:

- all-layer `dense_regular_llm_cuda` route plan;
- model-boundary fixtures for embedding, final norm, LM head, logits, and
  top-k diagnostics;
- KV-cache policy receipt;
- logits-transfer and sampling-policy receipt.

## Required Future Receipt

The future one-token proof receipt must record:

- artifact SHA and model contract identity;
- tokenizer and prompt authority used for the deterministic one-token prompt;
- requested and selected backend, with `fallback_used=false`;
- execution route `dense_regular_llm_cuda`;
- prerequisite receipt hashes or paths;
- CPU and CUDA selected token IDs;
- logits/top-k comparison evidence or first-divergence diagnostics;
- kernel, residency, transfer, and timing evidence available for the strict
  one-token path;
- quality gate result for exactly one deterministic greedy token;
- claim boundary showing no short-decode, chat, speedup, full-residency, server,
  BitNet packed proof, or QK256 claim.

## Required Non-Claims

This contract does not claim:

- dense GGUF inference is generally complete;
- Qwen short decode or chat works on CUDA;
- dense regular-LLM CUDA proves BitNet packed I2S/QK256 inference;
- CUDA speedup is benchmark-qualified;
- persistent-session or full dense CUDA residency exists;
- tokenizer, prompt-template, loader, transformer runtime, server, QK256, BitNet
  CUDA, or CUDA kernel math changed.

## Next Gate

After this contract, the implementation slice should add a
`dense_gguf_qwen_one_token_strict_cuda` receipt validator and command, run it
against the verified Qwen artifact on the RTX 5070 Ti machine, and commit the
hardware receipt only if it proves one deterministic greedy token with strict
CUDA fallback rejection.
