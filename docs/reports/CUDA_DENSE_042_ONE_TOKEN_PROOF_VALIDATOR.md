# CUDA-DENSE-042 One-Token Proof Validator

`CUDA-DENSE-042` turns the `CUDA-DENSE-041` one-token proof contract into a
fail-closed receipt validator. It does not implement runtime token generation
or commit a hardware one-token receipt.

The new validator accepts only a `dense_gguf_qwen_one_token_strict_cuda_proof`
receipt that records:

- the verified Qwen dense GGUF model identity;
- contract-authoritative tokenizer and prompt evidence;
- prerequisite receipt hashes for the all-layer CUDA plan, model-boundary
  fixtures, KV-cache policy, and sampling policy;
- selected route `dense_regular_llm_cuda` on `nvidia-rtx-5070-ti-cuda`;
- `fallback_used=false` and zero CPU fallback kernel invocations;
- exactly one deterministic greedy generated token;
- matching CPU and CUDA selected-token IDs;
- logits/top-k evidence hashes;
- kernel coverage, transfer accounting, tensor residency, and timing fields;
- claim boundaries that keep short-decode, chat, speedup, full residency,
  server readiness, BitNet packed proof, QK256 proof, and broad dense GGUF
  inference claims false.

## Validator Rejections

The synthetic tests prove that the one-token gate rejects:

- sampling-policy-only receipts;
- missing prerequisite verification;
- CPU/CUDA selected-token mismatch;
- short-decode claim upgrades;
- speedup and full-residency claim upgrades;
- BitNet packed I2S/QK256 proof claims.

## Non-Claims

This PR does not claim:

- Qwen one-token CUDA works on hardware;
- Qwen short decode or chat works;
- dense GGUF CUDA inference is generally complete;
- CUDA speedup is benchmark-qualified;
- persistent-session or full dense CUDA residency exists;
- tokenizer, prompt-template, loader, transformer runtime, server, QK256,
  BitNet CUDA, or CUDA kernel math changed.

## Next Gate

The next runtime slice can add the one-token command and commit a real hardware
receipt only when the verified Qwen artifact produces one deterministic greedy
token through `dense_regular_llm_cuda` with `fallback_used=false` and passes
this validator.
