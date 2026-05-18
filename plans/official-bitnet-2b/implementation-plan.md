# Official BitNet 2B Implementation Plan

## Objective

Make `microsoft/BitNet-b1.58-2B-4T` the fully governed official BitNet-rs
reference model family without widening current claims. The existing
I2_S/QK256 GGUF lane remains the product-CLI-ready CPU/CUDA answer lane; TL1,
TL2, BF16/GPU-int2, Apple, A770/OpenCL, speedup, full residency, and broad
server readiness remain route-specific future proof families.

## Hard Constraints

- Do not commit model binaries.
- Do not weaken or demote the current I2_S/QK256 `product_cli_ready` row.
- Do not promote speedup without same-artifact, same-tokenizer, same-prompt,
  same-route, fallback-free exact-profile benchmark review.
- Do not promote full residency without per-phase residency proof.
- Do not promote broad server readiness from exact-profile smoke.
- Do not let TL1/TL2 inherit I2_S/QK256 proof.
- Do not let dense SLM proof satisfy BitNet packed proof.
- Do not let CUDA proof satisfy Apple, A770, CPU, TL1, TL2, or BF16/GPU-int2
  proof.
- Do not claim no-scale F32 diagnostic QK256 as production I2_S.
- Keep receipts fallback-explicit and selected-route-explicit.

## Phase 0: Source-of-Truth Docs

### OFFICIAL-2B-000: Add official 2B source map

Status: ready

Goal: add the official 2B source map, campaign manifest, and source-of-truth
plan without runtime changes or claim promotion.

Files:

- `docs/bitnet/official-2b/README.md`
- `plans/official-bitnet-2b/README.md`
- `plans/official-bitnet-2b/implementation-plan.md`
- `docs/tracking/campaigns/official-bitnet-2b/active.toml`
- `docs/tracking/campaigns/official-bitnet-2b/CAMPAIGN.md`
- `docs/specs/INDEX.md`
- `docs/status/BITNET_CAPABILITY_MATRIX.md`
- generated campaign dashboards, if required by `xtask campaign generate`

Acceptance:

- docs only;
- no model binaries;
- no runtime changes;
- no claim promotion;
- current I2_S/QK256 state summarized;
- TL1/TL2/BF16 candidate boundaries explicit;
- validation passes or unavailable proof is documented.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check official-bitnet-2b
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Rollback: revert the source-map docs and generated campaign dashboard changes.

## Phase 1: Proposal and Specs

Future PRs add the proposal and behavior contracts as separate work items:

1. `docs/proposals/BITNET-PROP-0014-official-bitnet-2b-productization.md`
2. `docs/specs/BITNET-SPEC-OFFICIAL-2B-ARTIFACT-CONTRACT.md`
3. `docs/specs/BITNET-SPEC-OFFICIAL-2B-TOKENIZER-PROMPT.md`
4. `docs/specs/BITNET-SPEC-OFFICIAL-2B-I2S-QK256.md`
5. `docs/specs/BITNET-SPEC-OFFICIAL-2B-TL1-TL2.md`
6. `docs/specs/BITNET-SPEC-OFFICIAL-2B-CPU.md`
7. `docs/specs/BITNET-SPEC-OFFICIAL-2B-CUDA.md`
8. `docs/specs/BITNET-SPEC-OFFICIAL-2B-APPLE.md`
9. `docs/specs/BITNET-SPEC-OFFICIAL-2B-A770-OPENCL.md`
10. `docs/specs/BITNET-SPEC-OFFICIAL-2B-QUALITY.md`
11. `docs/specs/BITNET-SPEC-OFFICIAL-2B-PERFORMANCE.md`
12. `docs/specs/BITNET-SPEC-OFFICIAL-2B-RESIDENCY.md`
13. `docs/specs/BITNET-SPEC-OFFICIAL-2B-SERVER.md`
14. `docs/specs/BITNET-SPEC-OFFICIAL-2B-STATUS-SURFACE.md`

## Later Runtime Phases

Later phases remain blocked on their specs and route-specific proof:

- CPU counter audit, scaled AVX2/AVX512 proof, and hot-path cleanup.
- CUDA repeated benchmark, exact-profile speed review, transfer/residency
  closure, and bounded server exact-profile readiness.
- Apple/ARM I2_S and TL1 proof.
- Intel Arc A770 OpenCL I2_S/QK256 proof.
- x86 TL2 layout/scalar/answer proof.
- BF16 master to GPU int2/W2A8 contract and packer parity.
- CLI status and receipt explanation polish.
