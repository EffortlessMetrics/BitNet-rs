# OpenVINO Lunar Lake Implementation Plan

Status: proposed
Owner: intel-runtime/product
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs:
- [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: self
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No support-tier promotion until later proof receipts and promotion reviews satisfy the specs.
Policy impact: No policy exception.

## Campaign Goal

Lay down docs, specs, and rails for OpenVINO productization on the Lunar Lake
258V platform. This is a docs/specs/receipt-governance campaign first, not a
runtime promotion campaign.

OpenVINO must become BitNet-rs's governed Intel-runtime lane for dense SLMs and
selected small LLMs on Lunar Lake CPU/GPU/NPU, plus a clearly separate reference
lane for future BitNet-shaped graph/subgraph experiments.

## Global Hard Rules

- Do not promote OpenVINO GPU/NPU routes from docs PRs.
- Do not claim speedup without exact-profile benchmark qualification.
- Do not claim broad dense SLM quality from partial corpus evidence.
- Do not claim BitNet QK256/I2_S from OpenVINO dense SLM receipts.
- Do not claim native OpenCL from OpenVINO GPU receipts.
- Do not claim cold one-off NPU usability from hot-path numbers.
- Do not treat retokenized generated text as direct pipeline-internal generated
  token IDs.
- Keep model binaries uncommitted.
- Keep Python proof harnesses until Rust surfaces emit equivalent receipts and
  pass the same validators.

## Phase A: Encode Docs and Proof Boundaries

### Work item: LNL258V-OV-DOCS-001

Status: ready
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-DOCS-002
Blocked by: n/a

#### Goal

Add the Lunar Lake OpenVINO productization proposal, the OpenVINO route identity
contract, and this implementation plan.

#### Production delta

Docs only:

- `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
- `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`
- `plans/openvino-lunar-lake/README.md`
- `plans/openvino-lunar-lake/implementation-plan.md`
- `docs/tracking/campaigns/intel-258v-platform/active.toml`

#### Non-goals

No runtime code, no scripts, no model artifacts, no route promotion, no speedup
claim, no power claim, no server readiness claim, and no BitNet QK256 claim.

#### Acceptance

- Proposal defines why OpenVINO exists as a governed Intel-runtime lane.
- Route contract defines CPU/GPU/NPU identities and claim boundaries.
- Implementation plan lists PR-sized next steps.
- No runtime claims are promoted.
- Campaign tracker gets only docs/spec work items.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

#### Rollback

Revert the docs/spec/plan files and the active campaign work item.

### Work item: LNL258V-OV-DOCS-002

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-DOCS-003
Blocked by: LNL258V-OV-DOCS-001

#### Goal

Add `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md` to define the dense SLM
model/export contract and proof ladder.

#### Acceptance

The spec captures the Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric IR contract,
artifact/export manifest requirements, OpenVINO CPU/GPU/NPU proof ladder,
promotion prerequisites, and model-binary uncommitted rule.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-003

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-DOCS-004
Blocked by: LNL258V-OV-DOCS-002

#### Goal

Add `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md` to make NPU
first-ever compile, cached compile/load, warm second ask, resident session, and
low-power proof measurable.

#### Acceptance

The spec defines `npu_cold_one_off`, `npu_cached_one_off`,
`npu_warm_second_ask`, `npu_resident_10x_ask_short`,
`npu_resident_warm_chat`, and `npu_low_power_ask_short`; it requires cache and
resident timing fields and forbids cold one-off promotion from hot-path numbers.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-004

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs:
- `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md`
- `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-DOCS-005
Blocked by: LNL258V-OV-DOCS-003

#### Goal

Add quality corpus and phase timing specs.

#### Acceptance

Quality corpus spec defines direct outputs, profile/category summaries,
failure taxonomy, prompt evidence, generation config, fallback requirements,
and retokenized-token caveats. Phase timing spec defines model/pipeline load,
tokenizer load, prompt render, tokenization, pipeline construct, first chunk,
TTFT, prefill, decode, generated-token counts, throughput, and OpenVINO perf
metric fields.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-005

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs:
- `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md`
- `docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-DOCS-006
Blocked by: LNL258V-OV-DOCS-004

#### Goal

Add route promotion and BitNet boundary specs.

#### Acceptance

Route promotion spec defines route states, profile-scoped promotion gates,
blocker format, and benchmark/telemetry conditions. BitNet boundary spec
prevents dense SLM OpenVINO evidence from becoming BitNet QK256/I2_S, native
OpenCL, or full BitNet inference proof and links to the NPU subgraph ladder.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Work item: LNL258V-OV-DOCS-006

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs:
- `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md`
- `docs/specs/BITNET-SPEC-OPENVINO-SERVER.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-VALIDATE-001
Blocked by: LNL258V-OV-DOCS-005

#### Goal

Add Rust bridge and server specs.

#### Acceptance

Rust bridge spec preserves the Python proof harness until Rust surfaces emit
equivalent receipts. Server spec limits OpenVINO server readiness to exact
endpoint/profile receipts after ask/chat route readiness.

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Phase B: Receipt Validation and Status Without Runtime Changes

### Work item: LNL258V-OV-VALIDATE-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-STATUS-001
Blocked by: LNL258V-OV-DOCS-006

#### Goal

Add OpenVINO receipt validators for selected backend/runtime device consistency,
strict fallback rejection, retokenized token marking, dense SLM claim boundaries,
native OpenCL boundary, and NPU cache/warm fields when promotion is attempted.

#### Non-goals

No route promotion.

### Work item: LNL258V-OV-STATUS-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-STATUS-002
Blocked by: LNL258V-OV-VALIDATE-001

#### Goal

Add `docs/status/OPENVINO_CAPABILITY_MATRIX.md` with candidate/control rows for
Qwen2.5 OpenVINO CPU/GPU/NPU and a research row for BitNet OpenVINO subgraphs.

#### Non-goals

No speed, power, server, or BitNet QK256 promotion.

### Work item: LNL258V-OV-STATUS-002

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-QUAL-REPORT-001
Blocked by: LNL258V-OV-STATUS-001

#### Goal

Teach `receipts explain` to summarize OpenVINO route ID, selected backend,
OpenVINO device, proof family, quality status, timing scope, promotion status,
blockers, and what the receipt does not prove.

#### Non-goals

No route promotion.

## Phase C: Close Quality Gaps

### Work item: LNL258V-OV-QUAL-REPORT-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-QUAL-FIX-001
Blocked by: LNL258V-OV-STATUS-002

#### Goal

Add `docs/reports/OPENVINO_LUNAR_LAKE_CORPUS_V2_FAILURES.md` classifying CPU,
GPU, and NPU corpus-v2 failures.

### Work item: LNL258V-OV-QUAL-FIX-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-QUAL-RERUN-001
Blocked by: LNL258V-OV-QUAL-REPORT-001

#### Goal

Codify generation-budget sensitivity and fix or document corpus fixture policy
before any OpenVINO route promotion.

### Work item: LNL258V-OV-QUAL-RERUN-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-PERF-001
Blocked by: LNL258V-OV-QUAL-FIX-001

#### Goal

Rerun OpenVINO CPU/GPU/NPU corpus-v2 with the same export, prompt template,
generation policy, all 12 cases, and `fallback_used=false`.

## Phase D: Close Performance Evidence Gaps

### Work item: LNL258V-OV-PERF-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-PERF-GPU-001
Blocked by: LNL258V-OV-QUAL-RERUN-001

#### Goal

Add profile-specific OpenVINO phase runner output for prompt/output token count,
pipeline construct, tokenization, first text chunk, TTFT, decode, per-token time,
throughput, perf metrics, and cache config.

### Work item: LNL258V-OV-PERF-GPU-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-PERF-NPU-001
Blocked by: LNL258V-OV-PERF-001

#### Goal

Run GPU profile benchmarks for regression, short ask, normal ask, prefill-heavy,
decode-heavy, and structured profiles.

### Work item: LNL258V-OV-PERF-NPU-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs:
- `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md`
- `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-PERF-TELEMETRY-001
Blocked by: LNL258V-OV-PERF-GPU-001

#### Goal

Run NPU cold/cache/warm/resident benchmarks with cache, prompt/response shape,
prefill, and generate hints recorded.

### Work item: LNL258V-OV-PERF-TELEMETRY-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-PROMOTE-GPU-001
Blocked by: LNL258V-OV-PERF-NPU-001

#### Goal

Upgrade power/thermal telemetry requirements and receipts. No power claim unless
measured or accepted by explicit policy.

## Phase E: Route Promotion Reviews

### Work item: LNL258V-OV-PROMOTE-GPU-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-PROMOTE-NPU-001
Blocked by: LNL258V-OV-PERF-TELEMETRY-001

#### Goal

Review exact-profile GPU promotion only after quality passes, fallback is false,
Arc 140V is selected, profile timing is present, CPU comparator or accepted UX
advantage exists, and no BitNet/OpenCL claim leaks.

### Work item: LNL258V-OV-PROMOTE-NPU-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-MODEL-STATUS-001
Blocked by: LNL258V-OV-PROMOTE-GPU-001

#### Goal

Review exact warm/resident/low-power NPU promotion only after quality, selected
NPU, fallback, cache/resident, cold-start caveat, and power/thermal or
power-proxy evidence pass.

### Work item: LNL258V-OV-MODEL-STATUS-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-RUST-001
Blocked by: LNL258V-OV-PROMOTE-NPU-001

#### Goal

Add `bitnet model status --device intel-258v-openvino` once route states are
validated.

## Phase F: Rust-Native Product Surface

### Work item: LNL258V-OV-RUST-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-RUST-002
Blocked by: LNL258V-OV-MODEL-STATUS-001

#### Goal

Add Rust CLI wrappers around existing OpenVINO Python proof harnesses for ask,
corpus, and bench while preserving receipt compatibility.

### Work item: LNL258V-OV-RUST-002

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-ADR-001
Blocked by: LNL258V-OV-RUST-001

#### Goal

Add Rust receipt readers and integrate OpenVINO receipt summaries into model
status and receipt explainers independently of whether a receipt came from
Python or Rust.

### Work item: LNL258V-OV-ADR-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md`
Linked ADRs: `docs/adr/BITNET-ADR-OPENVINO-RUST-BINDING-STRATEGY.md`
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: LNL258V-OV-SERVER-001
Blocked by: LNL258V-OV-RUST-002

#### Goal

Add an ADR choosing the OpenVINO Rust binding, C/C++ FFI, or subprocess bridge
strategy without blocking proof on native bindings unless required.

### Work item: LNL258V-OV-SERVER-001

Status: blocked
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-OPENVINO-SERVER.md`
Linked ADRs: n/a
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: n/a
Blocked by: LNL258V-OV-ADR-001

#### Goal

Add exact-profile OpenVINO server proof only after ask/chat route readiness is
stable.

## Final Priority Order

1. Docs/rails: proposal, route contract, dense SLM contract.
2. Quality: diagnose and fix corpus-v2 failures.
3. Timing: profile-specific OpenVINO phase runner.
4. NPU: cold/cache/warm/resident proof.
5. GPU: exact-profile promotion review.
6. NPU: warm/low-power promotion review.
7. Status UX: OpenVINO capability matrix and `receipts explain`.
8. Rust bridge: wrap existing Python proof without losing receipts.
9. Server: exact-profile OpenVINO server proof only after ask/chat is stable.
10. BitNet OpenVINO: selected static subgraph research only, not full QK256 proof.
