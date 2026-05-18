# OpenVINO Lunar Lake Implementation Plan

Status: active
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../../docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](../../docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](../../docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](../../docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../../docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../../docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-BITNET-BOUNDARY](../../docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md), [BITNET-SPEC-OPENVINO-RUST-BRIDGE](../../docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md), [BITNET-SPEC-OPENVINO-SERVER](../../docs/specs/BITNET-SPEC-OPENVINO-SERVER.md)
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; PR sequencing only
Policy impact: no policy exception

## Scope

This plan sequences OpenVINO Lunar Lake work in PR-sized increments. Phase A is
source-of-truth documentation and proof-boundary work. Later phases improve
validators, quality diagnosis, timing evidence, route promotion reviews, status
UX, Rust bridge surfaces, server readiness, and BitNet subgraph research.

The campaign must keep OpenVINO dense SLM, OpenVINO GPU, OpenVINO NPU, native
OpenCL, BitNet QK256, and server proof families separate.

## Phase A: Encode Docs and Proof Boundaries

### Work item: LNL258V-OPENVINO-DOCS-001

Status: merged
Campaign item: `LNL258V-OPENVINO-DOCS-001`
Linked proposal: `BITNET-PROP-0004`
Linked specs: `BITNET-SPEC-OPENVINO-ROUTE-CONTRACT`
Blocked by: none
Blocks: `LNL258V-OPENVINO-DOCS-002`

#### Goal

Add the OpenVINO Lunar Lake productization proposal, route contract, and
implementation plan.

#### Production delta

Docs/specs only. No runtime code, scripts, model artifacts, generated receipts,
or route promotion.

#### Acceptance

- Proposal defines why OpenVINO exists as a governed Intel-runtime lane.
- Route contract defines CPU/GPU/NPU identities, proof families, required
  receipt fields, fallback rules, and claim boundaries.
- Implementation plan lists PR-sized next steps.
- Campaign tracker adds docs/spec work items only.
- No runtime claims are promoted.

#### Allowed paths

```text
docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md
docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md
plans/openvino-lunar-lake/README.md
plans/openvino-lunar-lake/implementation-plan.md
docs/tracking/campaigns/intel-258v-platform/active.toml
```

#### Forbidden paths

```text
crates/**
scripts/**
ci/hardware/**
ci/model-artifacts/**
README.md
```

#### Proof commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

#### Claim boundary

No OpenVINO GPU/NPU route promotion, speedup, power advantage, broad dense SLM
quality, BitNet QK256, native OpenCL, cold one-off NPU usability, model-binary,
or server-readiness claim.

### Work item: LNL258V-OPENVINO-DOCS-002

Status: merged
Linked proposal: `BITNET-PROP-0004`
Linked specs: `BITNET-SPEC-OPENVINO-ROUTE-CONTRACT`, `BITNET-SPEC-OPENVINO-DENSE-SLM`
Blocked by: `LNL258V-OPENVINO-DOCS-001`

Add `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md` defining dense SLM support
through OpenVINO GenAI, exact model/export contract fields, the proof ladder,
and profile-scoped promotion prerequisites. Do not promote any route.

Acceptance additions:

- Qwen2.5 0.5B Instruct has a precise OpenVINO artifact/export manifest
  contract.
- Future small LLM candidates must enter through the same manifest, smoke,
  answer, phase, route-profile, and promotion-review ladder.
- Dense SLM OpenVINO receipts remain separate from BitNet QK256/I2_S proof.

### Work item: LNL258V-OPENVINO-DOCS-003

Status: merged
Linked proposal: `BITNET-PROP-0004`
Linked specs: `BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE`
Blocked by: `LNL258V-OPENVINO-DOCS-002`

Add `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md` defining NPU
first-ever compile, cached startup, warm second ask, resident session, cache,
`PREFILL_HINT`, `GENERATE_HINT`, `MAX_PROMPT_LEN`, and `MIN_RESPONSE_LEN`
receipt requirements. Do not claim cold one-off NPU usability.

Acceptance additions:

- First-ever cold, cached cold-process, warm same-process, and resident-session
  timing modes are separate receipt modes.
- NPU cache identity, cache hit/miss evidence, GenAI configuration, phase
  timing, answer quality, fallback, and route-promotion fields are required.
- Hot first-token/decode evidence alone cannot support cold one-off NPU
  usability, speedup, power-advantage, or route-promotion claims.

### Work item: LNL258V-OPENVINO-DOCS-004

Status: merged
Linked proposal: `BITNET-PROP-0004`
Linked specs: `BITNET-SPEC-OPENVINO-QUALITY-CORPUS`, `BITNET-SPEC-OPENVINO-PHASE-TIMING`
Blocked by: `LNL258V-OPENVINO-DOCS-003`

Add:

```text
docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md
docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md
```

Define corpus-v2 profile gates, failure taxonomy, retokenized token-ID marking,
prompt evidence, generation config, and profile-specific timing fields.

Acceptance additions:

- Quality corpus receipts define required profiles/categories, prompt/template
  evidence, stop/EOS policy, generation config, and direct versus retokenized
  token accounting.
- Phase timing receipts define profile token-bound applicability, cold/cache/
  warm/resident split, OpenVINO metric gaps, telemetry context, and comparison
  requirements.
- Quality and timing evidence are inputs to route-promotion review but do not
  promote OpenVINO GPU/NPU routes by themselves.

### Work item: LNL258V-OPENVINO-DOCS-005

Status: merged
Linked proposal: `BITNET-PROP-0004`
Linked specs: `BITNET-SPEC-OPENVINO-ROUTE-PROMOTION`, `BITNET-SPEC-OPENVINO-BITNET-BOUNDARY`
Blocked by: `LNL258V-OPENVINO-DOCS-004`

Add:

```text
docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md
docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md
```

Define route states, exact-profile promotion gates, and the separation between
OpenVINO dense SLM proof and BitNet QK256/I2_S proof.

Acceptance additions:

- Route-promotion spec defines candidate/promoted/blocked states,
  exact-profile promotion packages, invalidation, auto-route behavior, and
  CPU/GPU/NPU promotion gates.
- BitNet-boundary spec separates OpenVINO dense SLM, OpenVINO BitNet subgraph,
  native OpenCL, NPU, server, and CPU BitNet reference proof families.
- Dense SLM OpenVINO success cannot count as BitNet QK256/I2_S proof, and
  accelerator BitNet claims require CPU-reference parity plus exact
  kernel/subgraph timing evidence.

### Work item: LNL258V-OPENVINO-DOCS-006

Status: merged
Linked proposal: `BITNET-PROP-0004`
Linked specs: `BITNET-SPEC-OPENVINO-RUST-BRIDGE`, `BITNET-SPEC-OPENVINO-SERVER`
Blocked by: `LNL258V-OPENVINO-DOCS-005`

Add:

```text
docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md
docs/specs/BITNET-SPEC-OPENVINO-SERVER.md
```

Define the Python-to-Rust bridge stages and exact-profile server readiness only
after ask/chat route readiness.

Acceptance additions:

- Rust bridge spec defines staged proof from Python harness through Rust wrapper,
  validator, subprocess bridge, binding, and product surfaces.
- Server spec defines exact-profile server receipts, underlying route linkage,
  cold/warm timing, fallback behavior, exposure fields, and streaming/
  concurrency boundaries.
- Neither spec claims route promotion, broad server readiness, speedup, power
  advantage, or BitNet QK256/I2_S behavior.

## Phase B: Improve Receipt Validation and Status Without Runtime Promotion

### Work item: LNL258V-OPENVINO-VALIDATE-001

Status: merged
Blocked by: Phase A specs

Add validators for selected backend/device consistency, `fallback_used=false` on
strict routes, retokenized token-ID marking, no dense-SLM-to-BitNet claim leak,
no OpenVINO-GPU-to-native-OpenCL claim leak, and NPU cache/warm fields when NPU
promotion is attempted.

Production delta: receipt validation only. No inference, no route promotion, no
runtime execution change, and no committed hardware artifact refresh.

Acceptance additions:

- `bitnet-receipts-core` exposes a Lunar Lake OpenVINO receipt validator and
  synthetic rejection tests for fallback, device/backend mismatch, token-ID
  source ambiguity, claim leakage, and premature NPU promotion.
- `bitnet validate open-vino-lunar-lake --receipt <path>` runs the validator and
  can emit a validation summary without changing the source receipt.
- Existing committed OpenVINO corpus, phase, route-profile, route-promotion,
  operator-ask, and NPU cold-start diagnosis receipts pass the new validator.

### Work item: LNL258V-OPENVINO-STATUS-001

Status: merged
Blocked by: `LNL258V-OPENVINO-VALIDATE-001`

Add `docs/status/OPENVINO_CAPABILITY_MATRIX.md` with claim-neutral rows for
Qwen2.5 OpenVINO CPU/GPU/NPU and BitNet OpenVINO subgraph research.

Production delta: status documentation only. The matrix indexes current
candidate, promoted, diagnostic, and planned OpenVINO rows, the source receipts,
the validator command, and the claim boundaries without running inference or
promoting GPU/NPU routes.

Acceptance additions:

- OpenVINO CPU/GPU/NPU dense SLM rows link to the route ledger, route-profile
  comparison, corpus-v2, phase, NPU cold-start, and operator-ask evidence.
- BitNet OpenVINO rows remain diagnostic/planned subgraph research and do not
  imply BitNet QK256/I2_S, full accelerator inference, or QK256 decode proof.
- The status surface names the validation command and the blockers required
  before route promotion, speedup, power, or server claims can be made.

### Work item: LNL258V-OPENVINO-UX-001

Status: proposed
Blocked by: `LNL258V-OPENVINO-STATUS-001`

Teach `receipts explain` to summarize OpenVINO route ID, selected backend,
device, proof family, quality status, timing scope, promotion status, blockers,
and what the receipt does not prove.

## Phase C: Close Quality Gaps

1. Add `docs/reports/OPENVINO_LUNAR_LAKE_CORPUS_V2_FAILURES.md` and classify
   failures by route and failure class.
2. Codify generation-budget sensitivity evidence for normalized-match failures.
3. Rerun corpus-v2 only after fixture/generation policy fixes or documentation.

No route can promote until profile cases pass or an explicit spec marks a case
diagnostic-only.

## Phase D: Close Performance Evidence Gaps

1. Add a profile-specific OpenVINO phase runner with prompt/output token counts,
   pipeline construction, tokenization, first chunk, TTFT, decode, throughput,
   perf metrics, and cache config.
2. Run GPU profile benchmarks for regression, ask, prefill, decode, and
   structured profiles.
3. Run NPU cold/cache/warm/resident benchmarks with cache and GenAI NPU config.
4. Upgrade power/thermal telemetry or record explicit unavailable reasons.

No speed or power claim is allowed without exact-profile benchmark
qualification.

## Phase E: Route Promotion Reviews

- GPU route promotion review may promote only exact profiles that pass quality,
  select GPU.0 / Arc 140V without fallback, include profile timing, compare
  against same-profile CPU evidence, and avoid OpenCL/BitNet claim leakage.
- NPU route promotion review may promote only exact warm/resident/low-power
  profiles that pass quality, select NPU without fallback, include cache and
  resident proof, expose cold-start caveats, and include power/thermal or
  accepted power-proxy evidence.

## Phase F: Rust-Native Product Surface

Wrap existing Python OpenVINO proof harnesses before replacing them:

```text
stage 0: Python proof harness, committed receipts
stage 1: Rust CLI wrapper invokes Python script with strict args
stage 2: Rust receipt validator consumes Python receipt schema
stage 3: Rust OpenVINO runtime binding / subprocess bridge
stage 4: Rust-native OpenVINO GenAI wrapper if feasible
stage 5: user-facing ask/chat/bench/server surfaces
```

Do not delete Python proof harnesses until Rust surfaces emit equivalent
receipts and pass the same validators.

## Phase G: Server and BitNet OpenVINO Research

Server readiness is exact-profile only and follows ask/chat readiness. BitNet
OpenVINO starts with static subgraph parity and does not become full BitNet
QK256 decode or speedup proof without a later spec, ADR, plan, and receipts.
