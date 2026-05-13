# BITNET-ADR-0004: 9950X3D + RTX 5070 Ti CUDA Product Bench

- **Status:** Accepted
- **Date:** 2026-05-13
- **Linked proposal/spec:**
  [BITNET-PROP-0002](../proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md),
  [BITNET-SPEC-0007](../specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md),
  [RTX 5070 Ti roadmap](../specs/nvidia-rtx-5070-ti-roadmap.md)

## Context

BitNet-rs now has several CUDA proof surfaces for the same physical platform:
hardware receipts, strict backend identity, BitNet QK256 route receipts, dense
regular-LLM planning and validators, benchmark baselines, and answer-readiness
docs. Those surfaces are useful only if product claims name the exact machine,
backend, model family, route, receipt, and benchmark profile they prove.

The 9950X3D + RTX 5070 Ti box is the strongest current CUDA lane because it can
pair a same-box x86 CPU reference with an NVIDIA CUDA target. It should be the
canonical CUDA product bench for strict local inference work, while still
preserving narrow model-family boundaries:

```text
CPU reference: amd-9950x3d-cpu-avx512
CUDA target:  nvidia-rtx-5070-ti-cuda
```

## Decision

Use the 9950X3D + RTX 5070 Ti machine as the canonical CUDA product bench for
x86 CPU reference plus NVIDIA CUDA target productization work.

Strict product claims on this bench must use the explicit backend identities:

```text
requested_backend = nvidia-rtx-5070-ti-cuda
selected_backend = nvidia-rtx-5070-ti-cuda
reference_backend = amd-9950x3d-cpu-avx512
runtime_api = cuda
fallback_used = false
```

Generic `cuda` is acceptable as a user convenience selector only when the
receipt resolves it to the strict selected backend before making a proof claim.
Generic `cuda` by itself is not enough for RTX 5070 Ti proof.

Dense CUDA proof and BitNet CUDA proof remain separate:

- dense regular-LLM CUDA proof does not satisfy BitNet I2_S, QK256, 1-bit, or
  packed-kernel proof;
- BitNet QK256 CUDA proof does not satisfy dense SLM or small dense LLM proof;
- WGPU, Vulkan, D3D12, and generic GPU receipts do not satisfy CUDA proof;
- CPU AVX-512 receipts provide same-box reference evidence, not CUDA execution.

Speed claims are profile-specific and receipt-gated. A speedup accepted for
`one_token` does not imply speedup for `short_decode_32`, warm sessions, server
paths, dense SLMs, or other model families.

## Consequences

- The platform has one canonical CUDA product bench identity instead of several
  ambiguous labels.
- Future `ask`, `chat`, `bench`, and receipt explanation work can share one
  product contract while keeping route-specific evidence separate.
- Documentation, model coverage rows, campaign items, and receipts must name
  the exact route and model family before promoting a claim.
- Candidate models such as Qwen3, SmolLM2, Llama 3.2, Gemma, and Phi must
  graduate through their own artifact, tokenizer, CPU, CUDA, and benchmark
  ladders.
- Cross-platform GPU work remains useful, but it is reference evidence unless a
  CUDA receipt proves the CUDA route.

## Claim Boundary

This ADR does not prove:

- new CUDA runtime behavior;
- new CUDA kernel correctness;
- model answer readiness;
- dense SLM support;
- BitNet coherent local answers beyond already committed receipts;
- server readiness;
- any CUDA speedup;
- any claim for GPUs, CPUs, operating systems, or model artifacts outside the
  named proof bench.

This ADR only records the durable product-bench decision and the claim
boundaries future PRs must preserve.

## Alternatives Considered

- **Use generic `cuda` as the product label.** Rejected because it hides the
  actual hardware and lets receipts from different NVIDIA systems look
  interchangeable.
- **Treat dense Qwen CUDA as the CUDA product proof.** Rejected because dense
  regular-LLM CUDA is useful but does not prove BitNet I2_S, QK256, or 1-bit
  behavior.
- **Treat BitNet QK256 CUDA as proof for all local CUDA models.** Rejected
  because dense SLMs and small dense LLMs have different artifacts, tokenizers,
  prompt templates, route plans, kernels, and benchmark profiles.
- **Keep CUDA productization hardware-agnostic.** Rejected for strict claims:
  users need receipt-backed hardware identity before trusting fallback,
  residency, quality, or speed statements.

## How To Revert

Revert this ADR and update BITNET-PROP-0002, BITNET-SPEC-0007, the NVIDIA
5070 Ti campaign, model coverage rows, and CUDA user docs to remove the
canonical-bench assumption. Receipts already committed for this machine should
remain evidence for what happened, but future claims would need a replacement
bench decision before promotion.
