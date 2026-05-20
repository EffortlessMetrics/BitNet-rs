# CUDA Capability Matrix

This page is the user-facing status map for the 9950X3D + RTX 5070 Ti CUDA
product lane. It summarizes the current model rows from
`ci/model-artifacts/model-coverage-matrix.toml` without changing any claim.

The operational source of truth remains the model coverage matrix, the NVIDIA
campaign tracker, and the committed receipts. This page exists so users can
start from one place before running:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda
bitnet model coverage <row-id>
bitnet receipts explain --latest
```

## Source Of Truth

| Surface | Source |
| --- | --- |
| Model tier and claim booleans | `ci/model-artifacts/model-coverage-matrix.toml` |
| Human-readable model coverage | `docs/model-artifacts/MODEL_COVERAGE_MATRIX.md` |
| RTX 5070 Ti campaign state | `docs/tracking/campaigns/nvidia-5070ti/CAMPAIGN.md` |
| Strict CUDA product contract | `docs/specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md` |
| CUDA route/proof-family contract | `docs/specs/BITNET-SPEC-CUDA-ROUTE-CONTRACT.md` |
| Server readiness boundary | `docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md` |
| User quickstart | `docs/tutorials/9950x3d-5070ti-cuda-quickstart.md` |
| BitNet CUDA guide | `docs/tutorials/rtx5070ti-bitnet-cuda.md` |
| Dense Qwen CUDA guide | `docs/tutorials/rtx5070ti-dense-qwen-cuda.md` |
| Receipt triage guide | `docs/tutorials/cuda-receipt-triage.md` |

## Current CUDA Rows

| Model | Class | Tier | Route | Ask | Warm/session | Bench | Speed | Server | Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Microsoft BitNet 2B I2_S/QK256 | BitNet | `product_cli_ready` | `bitnet_qk256_cuda` | yes | yes | reviewed | no | no | Official packed I2_S/QK256 only. |
| Qwen2.5 0.5B Q8_0 | dense SLM | `product_cli_ready` | `dense_regular_llm_cuda` | yes | yes | reviewed | no | exact profile only | Dense regular-LLM CUDA only; not BitNet proof. |
| Qwen3 0.6B Q8_0 | dense SLM | `product_cli_ready` | `dense_regular_llm_cuda` | yes | yes | reviewed | no | exact profile only | Own Qwen3 receipts only; not Qwen2.5 or BitNet proof. |
| SmolLM2 360M Q8_0 | dense SLM candidate | `structurally_valid` | none | no | no | no | no | no | CPU quality is blocked pending same-prompt reference comparator capture. |
| Llama 3.2 1B | dense SLM candidate | `registered` | none | no | no | no | no | no | Artifact, tokenizer, prompt, CPU sanity, and route plan still required. |
| Llama 3.2 3B | dense SLM candidate | `registered` | none | no | no | no | no | no | Memory envelope plus full proof ladder still required. |
| Gemma/Phi small | dense SLM candidates | `registered` | none | no | no | no | no | no | Architecture-specific proof is required before route claims. |

## Claim Boundaries

- `product_cli_ready` means normal CLI surfaces exist for the scoped proof lane.
  It does not imply speedup, full residency, or server readiness.
- `accelerator_answer_ready` means strict accelerator receipts exist for the
  scoped model and route. It does not imply broad product UX readiness.
- Qwen3 product CLI and server-readiness claims are scoped to Qwen3 receipts
  only; they do not inherit Qwen2.5 or BitNet proof.
- Dense SLM CUDA proof is first-class CUDA product evidence, but it never proves
  BitNet packed I2_S/QK256 behavior.
- BitNet QK256 CUDA proof never proves dense regular-LLM CUDA behavior.
- CUDA receipts must name a route from the CUDA route contract and include an
  execution plan before they can promote a CUDA claim.
- Generic `cuda`, WGPU, Vulkan, CPU fallback, or hardware visibility is not
  strict RTX 5070 Ti CUDA proof.
- `speedup_claim=false` remains correct until a governed benchmark
  qualification receipt accepts an exact model/profile.
- `server_ready=true` is exact-profile only. Dense Qwen2.5 and Qwen3 are ready
  only for their own refreshed non-streaming RTX 5070 Ti shared-engine
  `/v1/chat/completions` receipts; this does not imply broad dense serving,
  concurrency, deployment readiness, speedup, full residency, or BitNet proof.

## Next Proofs

| Row | Next proof |
| --- | --- |
| `bitnet_official_2b_i2s_qk256` | Profile-specific speedup qualification and deeper residency/transfer timing. |
| `dense_qwen25_05b_q8_cuda` | Post-OPS requalification keeps speedup, benchmark-qualified speed, and full residency false. The next reduced-D2H proof must use a CUDA device top-k or greedy sampler receipt, then a refreshed exact-profile comparator with pure H2D timing and phase residency; exact-profile server readiness remains limited to `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json`. |
| `dense_qwen3_06b_q8_candidate` | Exact-profile server readiness is promoted by `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`; the `qwen3_cuda_repeated_comparator` contract and generator exist, but hardware aggregate comparator evidence and a separate review remain required before speed or benchmark-qualified claims. |
| `dense_smollm2_360m_candidate` | Same-prompt SmolLM2 first-token/top-k or checkpoint comparator capture using the SLM-CPU-022 contract. |
| Later dense SLM / small-LLM candidates | Artifact contract, tokenizer/prompt authority, CPU answer sanity, all-layer plan, boundary fixtures, strict CUDA proof, warm session, and benchmark review. |

## Validation

Run these checks after editing this page or the underlying status surfaces:

```powershell
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Qwen3.6 Candidate Boundary

Qwen3.6 rows are registered candidates only in the model coverage matrix. They do not currently prove dense CUDA execution, product CLI, server readiness, speedup, or full residency.
