# Intel GPU implementation plan

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-STATUS-SURFACE.md`
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Documentation only until a later runtime PR provides receipts.
Policy impact: Adds no exceptions.

## Goal

Productize Intel GPU as a receipt-backed family of exact routes without
conflating A770 native OpenCL, Arc 140V native OpenCL, OpenVINO GPU, Intel NPU,
CPU, CUDA, BitNet QK256, and dense SLM proof.

## Phase 0: source-of-truth alignment

### INTELGPU-DOCS-000: source map and contracts

Add:

- `docs/intel-gpu/README.md`
- `plans/intel-gpu/README.md`
- `plans/intel-gpu/implementation-plan.md`
- `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
- Intel GPU route, identity, BitNet QK256, dense SLM, quality, performance,
  residency, and status-surface specs.

Update:

- `docs/specs/INDEX.md`
- `docs/tracking/campaigns/intel-a770/active.toml`
- `docs/tracking/campaigns/intel-258v-platform/active.toml`
- generated campaign dashboards if required by `xtask`.

Acceptance:

- Documentation only.
- No route promotion.
- No receipt changes.
- No QK256/OpenCL kernel or model coverage changes.
- Campaign checks and `git diff --check` pass.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Rollback:

Revert the Intel GPU docs, specs, plan files, index update, campaign active
manifest additions, and any generated dashboard changes from this plan item.

## Phase 1: specs

1. `docs(proposal): add Intel GPU productization proposal`
2. `docs(spec): add Intel GPU route contract`
3. `docs(spec): add Intel GPU device identity contract`
4. `docs(spec): add Intel GPU BitNet QK256 contract`
5. `docs(spec): add Intel GPU dense SLM contract`
6. `docs(spec): add Intel GPU quality/performance/residency contracts`
7. `docs(spec): add Intel GPU status surface contract`

These PRs must not promote runtime claims. They define the proof rules that
later runtime and receipt PRs must satisfy.

## Phase 2: A770 route truth and proof ledger

- Reconcile committed A770 proof receipts with `ci/hardware/device-kernel-routing.toml`.
- Keep diagnostic rows diagnostic unless claim-grade receipts are committed.
- Add validators that prevent promoted A770 rows without receipt paths,
  `fallback_used=false`, selected backend `intel-arc-a770-opencl`, and explicit
  not-claims.

## Phase 3: A770 native OpenCL productization

- Refresh selected-device OpenCL/Level Zero identity.
- Lock QK256 scaled I2S-I8S scalar oracle and OpenCL parity fixtures.
- Record claim-grade QK256 OpenCL receipts.
- Add deterministic BitNet answer behavior gates.
- Add quality-gated profile timings.
- Promote only named trusted partial acceleration after all gates pass.

## Phase 4: Lunar Lake Arc 140V OpenVINO GPU route

- Classify OpenVINO GPU corpus-v2 failures.
- Fix profile-specific timing applicability gaps.
- Promote only exact OpenVINO GPU dense SLM profiles where quality,
  fallback-free execution, comparator advantage, and telemetry rules pass.

## Phase 5: Arc 140V native OpenCL BitNet-adjacent lane

- Refresh selected-device native OpenCL parity.
- Add BitNet QK256 candidate fixtures without claiming full BitNet support.
- Decide whether Arc 140V native OpenCL should pursue BitNet QK256 before A770.

## Phase 6: shared Intel GPU UX

- Teach `receipts explain` to state proof families and not-claims.
- Add an Intel GPU capability matrix.
- Add `bitnet gpu doctor --vendor intel --format json` for route readiness and
  driver/runtime identity.

## Non-goals for docs/spec PRs

- Do not add or modify QK256 kernels.
- Do not add or modify OpenCL kernels.
- Do not promote route-matrix rows.
- Do not change model coverage.
- Do not claim generic Intel GPU support.
- Do not claim speedup, full residency, or cross-family proof inheritance.
