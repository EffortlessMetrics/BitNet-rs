# OpenVINO Lunar Lake Implementation Plan

Status: active
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion from docs/spec PRs
Policy impact: none

## Goal

Lay down the docs, specs, and rails for OpenVINO productization on the Lunar Lake
258V platform. This is a docs/specs/receipt-governance campaign first, not a
runtime promotion campaign.

## Current State

- The Intel 258V platform campaign is active and already has CPU, Arc 140V GPU,
  Intel AI Boost NPU, and dense SLM OpenVINO receipts.
- Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric export exists as a dense SLM
  operating path.
- OpenVINO CPU/GPU/NPU bounded smokes, operator asks, corpus-v2 execution, phase
  comparison, and route-profile comparison receipts exist.
- OpenVINO routes remain candidate-only where quality, timing, direct-token,
  cache/resident, or benchmark-qualified advantage gaps remain.
- CPU remains the promoted dense SLM default route.
- OpenVINO dense SLM proof must not become BitNet QK256 proof.
- OpenVINO GPU proof must not become native OpenCL proof.
- OpenVINO NPU proof must not become generic accelerator proof.

## Phase A - Encode Docs And Proof Boundaries

### Work item: LNL258V-OV-DOCS-001

Status: ready
Linked proposal: BITNET-PROP-0004
Linked specs: BITNET-SPEC-OPENVINO-ROUTE-CONTRACT
Linked ADRs: n/a
Campaign item: `LNL258V-OV-DOCS-001`
Blocked by: existing Lunar Lake 258V campaign state
Blocks: LNL258V-OV-DOCS-002

#### Goal

Add the OpenVINO Lunar Lake productization proposal, the route identity contract,
and this implementation plan.

#### Production delta

Docs only. No runtime code, script, CI hardware receipt, model artifact, or
support-tier promotion changes.

#### Non-goals

No OpenVINO GPU/NPU promotion, speedup claim, broad dense SLM quality claim,
BitNet QK256 claim, native OpenCL claim, cold one-off NPU usability claim, or
model-binary commit.

#### Acceptance

- Proposal explains why OpenVINO exists as BitNet-rs's governed Intel-runtime
  dense SLM lane and separate BitNet-shaped reference lane.
- Route contract defines CPU/GPU/NPU identities, proof families, required
  receipt fields, strict fallback rules, and claim boundaries.
- Implementation plan lists PR-sized next steps.
- Campaign tracker receives docs/spec work items only.
- No runtime claims are promoted.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

#### Allowed paths

```text
docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md
docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md
plans/openvino-lunar-lake/README.md
plans/openvino-lunar-lake/implementation-plan.md
docs/tracking/campaigns/intel-258v-platform/active.toml
docs/tracking/campaigns/intel-258v-platform/generated/**
docs/tracking/generated/**
```

#### Forbidden paths

```text
crates/**
scripts/**
ci/hardware/**
ci/model-artifacts/**
README.md
```

#### Claim boundary

No runtime, quality, timing, speed, power, residency, server, BitNet QK256,
native OpenCL, or NPU cold-route claim changes.

#### Rollback

Revert the docs-only proposal, route contract, plan files, and campaign work
item. Existing receipts remain unchanged.

### Work item: LNL258V-OV-DOCS-002

Status: planned
Linked proposal: BITNET-PROP-0004
Linked specs: BITNET-SPEC-OPENVINO-DENSE-SLM
Linked ADRs: n/a
Campaign item: `LNL258V-OV-DOCS-002`
Blocked by: LNL258V-OV-DOCS-001
Blocks: LNL258V-OV-DOCS-003

#### Goal

Add the dense SLM model/export contract for OpenVINO GenAI.

#### Acceptance

Define the Qwen2.5 0.5B Instruct OpenVINO IR contract, required export manifest
fields, tokenizer/prompt-template provenance, proof ladder, and profile-scoped
promotion preconditions. No route is promoted.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-003

Status: planned
Linked proposal: BITNET-PROP-0004
Linked specs: BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE
Linked ADRs: n/a
Campaign item: `LNL258V-OV-DOCS-003`
Blocked by: LNL258V-OV-DOCS-002
Blocks: LNL258V-OV-DOCS-004

#### Goal

Add the NPU cold/cache/warm/resident timing contract.

#### Acceptance

Specify first-ever compile/infer, cached compile or pipeline construction,
first streamed chunk, time to first token, steady decode, warm second ask,
resident session, `CACHE_DIR`, `CACHE_MODE`, `PREFILL_HINT`, `GENERATE_HINT`,
`MAX_PROMPT_LEN`, and `MIN_RESPONSE_LEN` evidence. No cold one-off route is
promoted.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-004

Status: planned
Linked proposal: BITNET-PROP-0004
Linked specs: BITNET-SPEC-OPENVINO-QUALITY-CORPUS, BITNET-SPEC-OPENVINO-PHASE-TIMING
Linked ADRs: n/a
Campaign item: `LNL258V-OV-DOCS-004`
Blocked by: LNL258V-OV-DOCS-003
Blocks: LNL258V-OV-DOCS-005

#### Goal

Add OpenVINO quality corpus and phase-timing specs.

#### Acceptance

Define corpus-v2 case output, failure taxonomy, profile/category summaries,
prompt evidence, generation config, retokenized-token labeling, profile-specific
timing fields, OpenVINO performance metrics, and forbidden timing comparisons.
No route is promoted.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-005

Status: planned
Linked proposal: BITNET-PROP-0004
Linked specs: BITNET-SPEC-OPENVINO-ROUTE-PROMOTION, BITNET-SPEC-OPENVINO-BITNET-BOUNDARY
Linked ADRs: n/a
Campaign item: `LNL258V-OV-DOCS-005`
Blocked by: LNL258V-OV-DOCS-004
Blocks: LNL258V-OV-DOCS-006

#### Goal

Add route-promotion and BitNet-boundary specs.

#### Acceptance

Define route states, exact-profile promotion preconditions, blockers, no-claim
boundaries, and the future static BitNet-shaped OpenVINO subgraph ladder. No
OpenVINO dense SLM success may imply BitNet QK256 proof.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-006

Status: planned
Linked proposal: BITNET-PROP-0004
Linked specs: BITNET-SPEC-OPENVINO-RUST-BRIDGE, BITNET-SPEC-OPENVINO-SERVER
Linked ADRs: n/a
Campaign item: `LNL258V-OV-DOCS-006`
Blocked by: LNL258V-OV-DOCS-005
Blocks: Phase B receipt validation/status work

#### Goal

Add Rust bridge and exact-profile server specs.

#### Acceptance

Define the staged path from Python proof harnesses to Rust wrappers/readers and
possible runtime bindings, require receipt compatibility, forbid deleting Python
proof harnesses until Rust emits equivalent receipts, and define server readiness
only after ask/chat route readiness.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Phase B - Improve Receipt Validation And Status

Planned after Phase A specs land:

1. OpenVINO receipt validator for selected device consistency, strict fallback,
   retokenized token labeling, no BitNet QK256 leakage, no native OpenCL leakage,
   and NPU cache/warm fields when promotion is attempted.
2. OpenVINO capability matrix for Qwen2.5 OpenVINO CPU/GPU/NPU and BitNet
   subgraph research rows.
3. `receipts explain` OpenVINO route summaries that show route ID, selected
   backend, OpenVINO device, proof family, quality status, timing scope,
   promotion status, blockers, and what the receipt does not prove.

## Phase C - Close Quality Gaps

Planned after validators/status explainers:

1. OpenVINO corpus-v2 failure diagnosis report by route.
2. Generation-budget sensitivity receipts and policy decisions.
3. Corpus-v2 rerun after prompt/template/generation-policy fixes.

No profile is promotion-eligible until every case for that profile passes or an
explicit spec marks the case diagnostic-only and excluded.

## Phase D - Close Performance Evidence Gaps

Planned after quality gaps:

1. Profile-specific OpenVINO phase runner with prompt/output token counts,
   pipeline construction, tokenize, first chunk, TTFT, decode, throughput,
   perf metrics, and cache config.
2. GPU profile benchmark for regression, ask, prefill, decode, and structured
   profiles.
3. NPU cold/cache/warm/resident benchmark.
4. Power/thermal telemetry upgrade.

No speed, power, or full-residency claim is allowed without exact-profile
evidence.

## Phase E - Route Promotion

Promotion reviews are separate PRs after quality and timing evidence exists:

1. GPU route promotion review for exact profiles only.
2. NPU warm/resident/low-power route promotion review for exact profiles only.
3. Model status OpenVINO surface.

No docs/spec PR promotes a route.

## Phase F - Rust-Native Product Surface

After receipt governance is stable:

1. Rust CLI wrapper around existing OpenVINO Python harnesses.
2. Rust receipt readers and model status integration.
3. Rust OpenVINO runtime binding strategy ADR.

Python proof harnesses remain until Rust surfaces emit equivalent receipts and
pass equivalent validators.
