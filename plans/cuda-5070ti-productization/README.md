# 9950X3D + RTX 5070 Ti CUDA Productization Plan

This plan turns the existing NVIDIA RTX 5070 Ti proof lane into a normal,
receipt-backed CUDA product bench without broadening any model, hardware, or
speed claim.

The product bench identities are fixed by
[BITNET-ADR-0004](../../docs/adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md):

```text
CPU reference: amd-9950x3d-cpu-avx512
CUDA target:  nvidia-rtx-5070-ti-cuda
```

## Source-Of-Truth Links

- Proposal:
  [`BITNET-PROP-0002`](../../docs/proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md)
- Contract spec:
  [`BITNET-SPEC-0007`](../../docs/specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- CUDA route contract:
  [`BITNET-SPEC-CUDA-ROUTE-CONTRACT`](../../docs/specs/BITNET-SPEC-CUDA-ROUTE-CONTRACT.md)
- Server readiness spec:
  [`BITNET-SPEC-0010`](../../docs/specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md)
- Bench ADR:
  [`BITNET-ADR-0004`](../../docs/adr/BITNET-ADR-0004-9950x3d-5070ti-cuda-product-bench.md)
- CUDA campaign:
  [`docs/tracking/campaigns/nvidia-5070ti/CAMPAIGN.md`](../../docs/tracking/campaigns/nvidia-5070ti/CAMPAIGN.md)
- Live campaign state:
  `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- Model coverage:
  `ci/model-artifacts/model-coverage-matrix.toml`
- Receipt root:
  `ci/hardware/windows-9950x3d-rtx5070ti/**`

## Files

- [`implementation-plan.md`](implementation-plan.md) lists the PR-sized queue.
- [`bitnet-official-i2s.md`](bitnet-official-i2s.md) owns the official BitNet
  2B I2_S/QK256 product path.
- [`dense-qwen.md`](dense-qwen.md) owns the Qwen2.5 0.5B Q8_0 dense CUDA lane.
- [`small-llm-candidates.md`](small-llm-candidates.md) owns the candidate
  onboarding ladder.
- [`benchmark-qualification.md`](benchmark-qualification.md) owns
  profile-specific speed decisions.
- [`server-readiness.md`](server-readiness.md) owns exact-profile server
  readiness promotion rules.

## Claim Boundary

This plan does not claim new runtime behavior. It only defines the order,
acceptance, receipts, and rollback paths for future PRs. CUDA-route proof now
flows through the narrower route contract so future receipts can distinguish
BitNet QK256 CUDA, dense regular-LLM CUDA, diagnostics, layer planning, and
server shared-engine profiles without proof-family conflation.

Do not use this plan to claim:

- dense Qwen proof as BitNet proof;
- BitNet QK256 proof as dense SLM proof;
- generic `cuda` as RTX 5070 Ti proof;
- CUDA receipts without a selected route, execution plan, and proof-family
  booleans as promotable product proof;
- hardware execution as answer quality;
- benchmark baselines as accepted speedup;
- server readiness from CLI receipts.

## Validation For This Plan PR

```bash
git diff --check
```
