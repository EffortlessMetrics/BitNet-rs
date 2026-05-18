# Official Microsoft BitNet 2B Productization Campaign

## Objective

Make `microsoft/BitNet-b1.58-2B-4T` the fully governed official BitNet-rs
reference model family. The current I2_S/QK256 GGUF route remains the bounded
answer-ready and product-CLI-ready CPU/CUDA lane. Every other route or product
claim must be proven independently.

## Current State

- `bitnet_official_2b_i2s_qk256` is the current official I2_S/QK256 coverage
  row and remains `product_cli_ready`.
- That row records `reference_good`, `cpu_answer_ready`,
  `accelerator_answer_ready`, and `bitnet_packed_i2s_qk256_proof`.
- That row still keeps `benchmark_qualified`, `server_ready`,
  `speedup_claim`, and `full_residency_claim` false.
- TL1 ARM, TL2 x86, and BF16/GPU-int2 remain registered candidates with separate
  proof requirements.

## Claim Boundaries

This campaign is route-specific:

- I2_S/QK256 proof is not TL1 proof.
- I2_S/QK256 proof is not TL2 proof.
- I2_S/QK256 proof is not BF16/GPU-int2 proof.
- CUDA proof is not CPU, Apple, A770, TL1, TL2, or BF16/GPU-int2 proof.
- Dense regular-LLM CUDA proof is not BitNet packed-kernel proof.
- Exact-profile server smoke is not broad production server readiness.
- Upload-once QK256 weights are not full device residency.

## Active Work

The active machine-readable work queue is:

- `docs/tracking/campaigns/official-bitnet-2b/active.toml`

The source map and plan are:

- `docs/bitnet/official-2b/README.md`
- `plans/official-bitnet-2b/implementation-plan.md`
