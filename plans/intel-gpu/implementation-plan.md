# Intel GPU implementation plan

## Purpose

This plan lays down the PR sequence for making Intel GPU support a first-class,
receipt-backed inference family without conflating A770 native OpenCL, Arc 140V
native OpenCL, OpenVINO GPU, Intel NPU, CPU, CUDA, BitNet QK256, or dense SLM
proof families.

## Authorities

- Source-of-truth map: `docs/intel-gpu/README.md`
- A770 hardware/runtime lane: `docs/specs/intel-arc-a770-gpu-roadmap.md`
- A770 BitNet claim boundary: `docs/specs/a770-bitnet-claim-boundary.md`
- Lunar Lake platform campaign: `docs/tracking/campaigns/intel-258v-platform/`
- A770 campaign: `docs/tracking/campaigns/intel-a770/`

## Phase 0: source-of-truth alignment

### INTEL-GPU-DOCS-000: add Intel GPU source-of-truth map

Scope:

- Add `docs/intel-gpu/README.md`.
- Add `plans/intel-gpu/README.md`.
- Add this implementation plan.
- Update `docs/specs/INDEX.md`.
- Update the Intel A770 and Intel 258V active goals to point at the shared map
  without changing runtime claims.

Acceptance:

- Documentation only.
- No route promotion.
- No receipt changes.
- No QK256, OpenCL, OpenVINO, model-coverage, or server-code changes.
- Campaign checks and generation checks pass, or any environment limitation is
  recorded.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Phase 1: shared Intel GPU specifications

Add one semantic spec PR at a time:

1. Productization proposal: why Intel GPU exists as a product family.
2. Route contract and device identity specs.
3. Native Intel GPU BitNet QK256 contract.
4. Dense SLM OpenVINO GPU contract.
5. Quality, performance, and residency contracts.
6. Status-surface contract.

Each spec PR must preserve the non-conflation rules from
`docs/intel-gpu/README.md` and must not promote runtime routes.

## Phase 2: A770 route truth and proof ledger

Reconcile `ci/hardware/device-kernel-routing.toml`, committed A770 receipts,
A770 verification tooling, and the A770 campaign. If claim-grade receipts are
not committed, diagnostic rows remain diagnostic and the missing artifacts are
listed explicitly.

## Phase 3: A770 native OpenCL productization

Proceed from selected-device identity through QK256 scalar-oracle fixtures,
claim-grade OpenCL receipts, deterministic answer behavior, quality-gated
profile timings, and only then trusted partial acceleration promotion for named
operations.

Allowed end claim after all gates pass:

```text
Official BitNet b1.58 I2_S/QK256 has trusted partial A770 OpenCL acceleration
for named operations.
```

This is not a full-device-residency, generic Intel GPU, dense SLM, OpenVINO GPU,
CUDA, NPU, or all-OpenCL-device claim.

## Phase 4: Lunar Lake Arc 140V OpenVINO GPU route

Classify OpenVINO GPU corpus-v2 failures, fix profile-timing applicability, add
profile comparisons, and promote only exact profiles that pass quality,
fallback, comparator, telemetry, generated-token-boundary, and accepted
advantage gates.

OpenVINO GPU proof is dense graph/runtime proof; it is not native OpenCL proof,
NPU proof, or BitNet QK256 proof.

## Phase 5: Arc 140V native OpenCL BitNet-adjacent lane

Refresh selected-device native OpenCL smoke/parity, add QK256 candidate fixtures
only after layout and scaled I2_S-I8_S parity are proven, and decide whether Arc
140V should pursue native BitNet before or after A770. Until evidence says
otherwise, A770 owns native BitNet OpenCL first and Arc 140V remains
smoke/parity plus dense OpenVINO GPU candidate.

## Phase 6: shared Intel GPU UX

Add user-facing route-family explanations after the proof contracts exist:

- `receipts explain` route-family boundaries.
- Intel GPU capability matrix.
- `bitnet gpu doctor --vendor intel --format json`.

These surfaces must show route ID, proof family, claim level, selected backend,
runtime API, quality status, performance status, residency status, server
status, not-claims, and next required proof.
