# BITNET-ADR-0005: Proof Families Are Not Interchangeable

- **Status:** Accepted
- **Date:** 2026-05-18
- **Linked proposal/spec:**
  [BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md),
  [BITNET-SPEC-0013](../specs/BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
  [BITNET-SPEC-0014](../specs/BITNET-SPEC-0014-runtime-performance-contract.md)

## Context

BitNet-rs now has useful proof for several different local inference families:
official BitNet I2_S/QK256 CUDA, dense Qwen CUDA, CPU reference routes,
server-smoke profiles, and registered follow-on candidates such as BitNet
TL1/TL2, GPU-int2, Qwen3, SmolLM2, Llama, Gemma, and Phi.

Those proofs are valuable only if status docs, model coverage rows, receipts,
and CLI summaries do not treat one family as evidence for another. A dense
regular-LLM CUDA receipt can prove dense CUDA for the exact artifact and
profile. It does not prove BitNet packed I2_S/QK256 behavior. A BitNet QK256
receipt can prove the official BitNet route for the exact artifact and profile.
It does not prove dense SLM CUDA, TL1/TL2, GPU-int2, or server readiness.

## Decision

BitNet-rs treats these as separate proof families:

```text
BitNet I2_S/QK256
BitNet TL1/TL2
BitNet GPU-int2
dense SLM CUDA
dense small-LLM CUDA
server readiness
```

Each family needs its own artifact, tokenizer, prompt, route, backend,
quality, performance, and server evidence before user-facing claims may promote
that family.

Generic hardware evidence is not enough. A receipt must name the requested
backend, selected backend, selected route, model artifact, profile, and
fallback status before a proof family claim is allowed.

## Consequences

- Model coverage rows must keep proof booleans explicit instead of inferring
  support from adjacent rows.
- Status docs must say which family a proof applies to and which family remains
  forbidden.
- `bitnet model status` and `bitnet receipts explain` must summarize the
  selected route and proof family rather than only saying a command ran.
- Dense Qwen2.5 CUDA proof may accelerate dense CUDA product work, but it does
  not satisfy Qwen3, SmolLM2, Llama, Gemma, Phi, or BitNet proof.
- Official BitNet I2_S/QK256 proof may accelerate BitNet QK256 product work,
  but it does not satisfy TL1/TL2, GPU-int2, dense CUDA, or server readiness.
- Server readiness is endpoint/profile scoped and must not be inferred from
  ask, chat, benchmark, or backend receipts alone.

## Claim Boundary

| Proof family | May claim when gated | Must not claim |
| --- | --- | --- |
| BitNet I2_S/QK256 | Official packed BitNet QK256 route for the exact artifact/backend/profile | dense CUDA, TL1/TL2, GPU-int2, broad server readiness |
| BitNet TL1/TL2 | TL route proof for the exact artifact/backend/profile | inherited I2_S/QK256 or dense CUDA proof |
| BitNet GPU-int2 | GPU-int2 route proof for the exact artifact/backend/profile | inherited QK256, TL, or dense CUDA proof |
| Dense SLM CUDA | Exact dense SLM artifact/backend/profile route | BitNet, I2_S, QK256, TL, GPU-int2, or 1-bit proof |
| Dense small-LLM CUDA | Exact small dense LLM artifact/backend/profile route | dense SLM rows, BitNet proof, or broad small-LLM support |
| Server readiness | Exact endpoint/profile/readiness envelope | speedup, full residency, streaming, concurrency, or broad production readiness |

Speed, full residency, and server readiness remain separate claims even inside
one proof family.

## Alternatives Considered

- **Treat CUDA proof as one family.** Rejected because dense regular-LLM CUDA,
  BitNet QK256 CUDA, TL routes, and GPU-int2 routes have different kernels,
  artifacts, tokenizers, prompt policies, fallback risks, and benchmark
  profiles.
- **Let a product CLI-ready row imply server readiness.** Rejected because
  server readiness depends on endpoint behavior, response metadata, readiness
  envelopes, and serving profiles.
- **Let one dense Qwen row promote other dense rows.** Rejected because each
  model has its own artifact, tokenizer, prompt template, architecture, and
  quality boundary.

## How To Revert

Reverting this ADR would require replacing it with another durable decision
that names how BitNet-rs prevents proof leakage across model families and
server profiles. Existing receipts should remain immutable evidence for what
happened; only the claim rules built on top of those receipts would change.
