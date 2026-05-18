# Apple Silicon Source-of-Truth Implementation Plan

## Purpose

Lay down Apple Silicon docs/specs/rails so future Codex agents can continue M4
Mac mini, MacBook, dense SLM, BitNet CPU/NEON, and Metal phase work without
mixing proof families or broadening support claims.

This plan is a documentation rollout. It does not promote runtime support and it
does not authorize touching model binaries, QK256 kernels, Metal kernels, server
runtime code, or live hardware timing paths.

## Current Authority Inputs

- `docs/specs/apple-m4-mac-mini-roadmap.md` defines the M4 lane as Metal-first,
  with MPSGraph as a graph/reference lane and CPU/NEON as fallback/parity.
- `docs/slm/apple-m4-dense-slm-model-support-matrix.md` defines supported dense
  M4 SLM models and promotion gates.
- `docs/slm/apple-m4-inference-excellence.md` is the operator-facing narrative
  for the active M4 inference-excellence campaign.
- `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml` is the
  machine-readable current M4 excellence campaign state.
- `ci/hardware/apple-m4-mac-mini/**` contains committed machine evidence; those
  artifacts are receipts, not broad Apple Silicon claims.

## Global Rails

Every PR in this rollout must preserve these boundaries:

- Dense Qwen SLM evidence is not BitNet evidence.
- BitNet CPU/NEON evidence is not Metal evidence.
- Metal visibility is not Metal execution.
- Metal subgraph parity is not full Metal inference.
- MPSGraph smoke is not native Metal proof.
- MPSGraph smoke is not Neural Engine proof unless the resolved target is
  receipt-backed.
- CPU fallback cannot count as Metal execution.
- MacBook evidence is not M4 Mac mini runtime proof.
- M4 Mac mini evidence is not broad Apple Silicon proof.
- QK256-on-x86/CUDA/A770 evidence is not QK256-on-Metal evidence.
- Supported dense SLMs must be artifact-pinned and tokenizer-authoritative.
- No model binaries are committed.
- Live hardware/model timing is never required in ordinary generic PR CI.

## PR Rollout

### PR 0 — Apple Silicon source-of-truth map

Title: `docs(apple): add Apple Silicon source-of-truth map`

Add:

- `docs/apple-silicon/README.md`
- `plans/apple-silicon/README.md`
- `plans/apple-silicon/implementation-plan.md`

Update:

- `docs/specs/INDEX.md`
- `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml`
- generated campaign docs if required by `xtask`

Acceptance:

- lists Apple proof families;
- identifies current source of truth for M4 dense SLM, BitNet, Metal, and
  MacBook work;
- explains which older campaign/docs surfaces are historical audit records;
- defines that future Apple Silicon specs are contractual rails, not runtime
  promotions.

Validation:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### PR 1 — Apple Silicon proposal

Title: `docs(proposal): add Apple Silicon productization proposal`

Add `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md`.

Acceptance:

- explains dense SLM-first product path;
- explains BitNet CPU/NEON path;
- explains phase-scoped Metal;
- explains MacBook auxiliary lane;
- makes no model/status/claim changes.

### PR 2 — Route contract

Title: `docs(spec): add Apple Silicon route contract`

Add `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md`.

Acceptance:

- defines `apple-m4-cpu-neon`, `apple-m4-metal`, `apple-m4-mpsgraph`, and
  MacBook labels;
- defines strict fallback failure rules and receipt fields;
- keeps proof families separated.

### PR 3 — Dense SLM appliance spec

Title: `docs(spec): define Apple M4 dense SLM appliance path`

Add `docs/specs/BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md`.

Acceptance:

- codifies default, supported, candidate, diagnostic-only, and rejected states;
- imports support matrix gates contractually;
- keeps Qwen support separate from BitNet.

### PR 4 — BitNet CPU/NEON spec

Title: `docs(spec): define Apple M4 BitNet CPU/NEON path`

Add `docs/specs/BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md`.

Acceptance:

- codifies the accepted Microsoft I2_S artifact path;
- requires strict tokenizer/loader proof;
- defines one-shot, warm, chat, and serve gates;
- prevents Metal, QK256, Neural Engine, MPSGraph, and dense SLM claim leakage.

### PR 5 — Metal phase-scoped acceleration spec

Title: `docs(spec): define phase-scoped Apple Metal acceleration`

Add `docs/specs/BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md`.

Acceptance:

- defines tiny smoke to parity to phase-contribution to generation-contribution
  ladder;
- requires CPU reference parity;
- blocks full Metal inference claims until a full route is proven separately.

### PR 6 — Quality and benchmark envelope specs

Title: `docs(spec): add Apple quality and benchmark envelope contracts`

Add:

- `docs/specs/BITNET-SPEC-APPLE-QUALITY-CORPUS.md`
- `docs/specs/BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE.md`

Acceptance:

- keeps dense and BitNet corpora separate;
- documents benchmark fields and profile list;
- requires p50/p90/p99/min/max and repeatability rules where profiles claim
  comparable timing.

### PR 7 — Reproducible identity and MacBook auxiliary specs

Title: `docs(spec): add Apple reproducibility and MacBook auxiliary lane specs`

Add:

- `docs/specs/BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md`
- `docs/specs/BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md`

Acceptance:

- makes Apple run identity fields contractual;
- states MacBook proof cannot promote M4 Mac mini proof;
- explains the storage/larger-artifact auxiliary role.

### PR 8 — Service surface spec

Title: `docs(spec): define Apple ask chat serve readiness`

Add `docs/specs/BITNET-SPEC-APPLE-SERVICE-SURFACE.md`.

Acceptance:

- defines doctor, evidence, ask, chat, serve, receipts-check, regression,
  report-refresh, and benchmark semantics;
- requires per-request receipts and exact-profile serve scope;
- avoids production-hosting overclaims.

## Proof Commands

For documentation PRs in this rollout, run:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If a command cannot run in the environment, record the command, the reason, any
substitute evidence, and whether it blocks merge.
