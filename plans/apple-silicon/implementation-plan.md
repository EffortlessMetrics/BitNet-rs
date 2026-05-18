# Apple Silicon docs and rails implementation plan

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md` (planned)
Linked specs: `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` (planned)
Linked ADRs: n/a
Linked plan: `plans/apple-silicon/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No support-tier promotion; this plan only sequences docs/spec rails.
Policy impact: No policy exception.

## Goal

Lay down Apple Silicon docs/spec rails so Codex agents can continue M4 Mac Mini,
MacBook, dense SLM, BitNet CPU/NEON, and Metal phase work without conflating
proof families or promoting unsupported runtime claims.

## Non-goals

- Do not promote full `apple-m4-metal` inference.
- Do not claim Neural Engine execution.
- Do not claim MPSGraph as native Metal proof.
- Do not claim dense SLM evidence proves BitNet.
- Do not claim BitNet CPU/NEON evidence proves Metal.
- Do not claim MacBook evidence proves M4 Mac Mini behavior.
- Do not touch QK256, Metal kernels, server runtime, or model binaries in these
  docs/spec PRs unless a later work item explicitly allows it.
- Do not add live hardware/model timing to ordinary generic PR CI.

## PR sequence

### PR 0 — Apple Silicon source-of-truth map

Title: `docs(apple): add Apple Silicon source-of-truth map`

Add:

- `docs/apple-silicon/README.md`
- `plans/apple-silicon/README.md`
- `plans/apple-silicon/implementation-plan.md`

Update:

- `docs/specs/INDEX.md`
- `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml`
- generated campaign status via `cargo run --locked -p xtask --no-default-features -- campaign generate`

Acceptance:

- Lists Apple proof families.
- Identifies current source of truth for M4 dense SLM, BitNet, Metal, MacBook,
  machine artifacts, operator narrative, and historical campaigns.
- Explains that old campaigns remain historical evidence.
- States that new specs are contractual rails, not support promotions.

Validation:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### PR 1 — Apple Silicon proposal

Title: `docs(proposal): add Apple Silicon productization proposal`

Add `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md`.

Acceptance: explain dense SLM-first product path, BitNet CPU/NEON path,
phase-scoped Metal, MacBook auxiliary role, no Neural Engine claim without
receipts, and no broad Apple Silicon claim from one Mac.

### PR 2 — Route contract

Title: `docs(spec): add Apple Silicon route contract`

Add `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md`.

Acceptance: define Apple route IDs, backend labels, receipt fields, fallback
failure rules, and proof-family separation.

### PR 3 — Dense SLM appliance spec

Title: `docs(spec): define Apple M4 dense SLM appliance path`

Add `docs/specs/BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md`.

Acceptance: codify supported/default/candidate/diagnostic-only/rejected states,
import support matrix gates, and keep Qwen dense evidence separate from BitNet.

### PR 4 — BitNet CPU/NEON spec

Title: `docs(spec): define Apple M4 BitNet CPU/NEON path`

Add `docs/specs/BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md`.

Acceptance: codify accepted Microsoft I2_S artifact proof ladder, tokenizer and
loader proof, one-shot/warm/chat/serve gates, and not-claims for Metal, QK256,
Neural Engine, MPSGraph, broad Apple Silicon, and dense SLM evidence.

### PR 5 — Metal phase-scoped acceleration spec

Title: `docs(spec): define phase-scoped Apple Metal acceleration`

Add `docs/specs/BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md`.

Acceptance: define Metal visibility, tiny smoke, CPU parity, I2_S and dense
phase fixtures, generation-contribution candidate, full-route candidate, CPU
reference parity requirements, and full Metal inference claim blockers.

### PR 6 — Quality and benchmark envelope specs

Title: `docs(spec): add Apple quality and benchmark envelope contracts`

Add:

- `docs/specs/BITNET-SPEC-APPLE-QUALITY-CORPUS.md`
- `docs/specs/BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE.md`

Acceptance: keep dense and BitNet corpora separate; document mechanical scoring,
failure taxonomy, p50/p90/p99/min/max, repeat counts, profile list, outlier
policy, and no broad Apple Silicon benchmark claim from one M4 Mac Mini profile.

### PR 7 — Reproducible identity and MacBook auxiliary specs

Title: `docs(spec): add Apple reproducibility and MacBook auxiliary lane specs`

Add:

- `docs/specs/BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md`
- `docs/specs/BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md`

Acceptance: make run identity fields contractual and require MacBook receipts to
remain separate from M4 Mac Mini proof with `counts_as_m4_mac_mini_proof=false`.

### PR 8 — Service surface spec

Title: `docs(spec): define Apple ask chat serve readiness`

Add `docs/specs/BITNET-SPEC-APPLE-SERVICE-SURFACE.md`.

Acceptance: define `bitnet mac` doctor/evidence/ask/chat/serve/receipts-check/
regression/report-refresh/benchmark semantics, per-request receipts, exact
profile serve scope, fallback boundaries, and no production hosting overclaim.

## Shared validation for docs/spec PRs

Each PR should run at least:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If a PR touches a generated campaign file, run the generator rather than editing
the generated file by hand.
