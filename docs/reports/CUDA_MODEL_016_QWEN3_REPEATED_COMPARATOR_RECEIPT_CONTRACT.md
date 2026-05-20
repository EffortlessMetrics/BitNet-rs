# CUDA-MODEL-016 Qwen3 Repeated Comparator Receipt Contract

Campaign item: CUDA-MODEL-016

Status: merged in PR #5941

## Scope

CUDA-MODEL-015 defined the Qwen3 repeated comparator baseline. This item adds
the receipt contract and generator required before the hardware lane commits
those repeated source receipts.

The receipt kind is:

```text
qwen3_cuda_repeated_comparator
```

## Required Profiles

The contract covers exactly the Qwen3 product profiles queued by
CUDA-MODEL-015:

- `one_token`;
- `short_decode_8`;
- `short_decode_32`;
- `warm_session_3_turns`;
- `decode_128_from_warm_context`.

Each profile requires at least three same-artifact CPU/CUDA comparator runs.

## Required Proof Fields

The validator requires:

- exact model `qwen3-0.6b-instruct-q8_0`;
- exact SHA-256
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`;
- selected backend `nvidia-rtx-5070-ti-cuda`;
- selected route `dense_regular_llm_cuda`;
- `fallback_used=false`;
- quality and parity pass fields;
- profile-specific generated-token counts;
- model load, tokenizer load, prompt render, tokenize, CUDA context, weight
  upload, prefill, first-token, decode, throughput, kernel, launch,
  H2D/D2H, VRAM, power, and thermal fields from the runtime performance
  contract.

## Claim Boundaries

This receipt is not a promotion. It must preserve:

```text
speedup_claim=false
benchmark_qualified_speedup=false
full_cuda_residency_claimed=false
broad_dense_gguf_ready_claimed=false
qwen25_proof_inherited=false
bitnet_packed_i2s_qk256_proof=false
```

Dense Qwen3 CUDA evidence remains Qwen3-specific dense SLM evidence. It does
not prove Qwen2.5, broad dense GGUF readiness, or BitNet packed I2_S/QK256.

## Tooling

The generator is:

```bash
cargo run --locked -p bitnet-bench-receipts --no-default-features --bin qwen3_cuda_repeated_comparator_receipt -- \
  --one-token-run <PATH> \
  --short-decode-8-run <PATH> \
  --short-decode-32-run <PATH> \
  --warm-session-3-run <PATH> \
  --decode-128-from-warm-context-run <PATH> \
  --receipt-out ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/qwen3-0_6b-repeated-comparator.json
```

Each run flag may be repeated. The generator requires at least three source
receipts per profile and validates the generated aggregate receipt before
writing it.

## Next Proof

The receipt contract and generator are landed. The next hardware step is to
collect the repeated Qwen3 source receipts for all five profiles, generate the
aggregate comparator receipt, and then run a separate benchmark qualification
review. Until that review lands, Qwen3 speed, benchmark-qualified speed, full
residency, and broader readiness remain false.
