# OpenVINO Lunar Lake Implementation Plan

This file sequences the docs, specs, validation, promotion, status, Rust bridge,
server, and BitNet-subgraph work needed to make OpenVINO an honest Intel-runtime
lane for Lunar Lake 258V.

The first phase is docs/spec/receipt governance. It must not promote GPU/NPU
routes, claim speedup, claim broad quality, claim native OpenCL proof, claim
BitNet QK256 proof, or claim cold one-off NPU usability.

## Sequence

| Order | Item | Expected files | Runtime delta |
| --- | --- | --- | --- |
| 1 | Proposal and route contract | `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`, `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`, this plan | none |
| 2 | Dense SLM model/export contract | `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md` | none |
| 3 | NPU cold/warm/cache contract | `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md` | none |
| 4 | Quality corpus and phase timing contracts | `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md`, `docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md` | none |
| 5 | Route promotion and BitNet boundary contracts | `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md`, `docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md` | none |
| 6 | Rust bridge and server contracts | `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md`, `docs/specs/BITNET-SPEC-OPENVINO-SERVER.md` | none |
| 7 | Receipt validator hardening | validator code and focused tests | validation only |
| 8 | OpenVINO capability matrix | `docs/status/OPENVINO_CAPABILITY_MATRIX.md` | none |
| 9 | `receipts explain` OpenVINO summaries | CLI/status reader code and tests | explanation only |
| 10 | Corpus-v2 failure diagnosis | `docs/reports/OPENVINO_LUNAR_LAKE_CORPUS_V2_FAILURES.md` | none |
| 11 | Generation budget sensitivity receipts | diagnostic receipts | diagnostic only |
| 12 | Corpus-v2 rerun after fixes | updated corpus receipts | candidate rerun only |
| 13 | Profile-specific OpenVINO phase runner | runner updates and receipts | measurement only |
| 14 | GPU profile benchmark | GPU profile receipts and review | no promotion unless review passes |
| 15 | NPU cold/cache/warm benchmark | NPU cache/warm/resident receipts | no cold promotion |
| 16 | Power/thermal telemetry upgrade | telemetry fields and receipts | measurement only |
| 17 | GPU route promotion review | promotion review artifact | exact-profile promotion only if gates pass |
| 18 | NPU warm route promotion review | promotion review artifact | warm/resident/low-power promotion only if gates pass |
| 19 | Model status OpenVINO surface | `bitnet model status --device intel-258v-openvino` | status UX |
| 20 | Rust CLI wrapper around Python harness | `bitnet openvino ask/corpus/bench` | wrapper only |
| 21 | Rust receipt readers and model status integration | receipt readers/status integration | schema-compatible readers |
| 22 | Rust OpenVINO binding strategy ADR | `docs/adr/BITNET-ADR-OPENVINO-RUST-BINDING-STRATEGY.md` | decision only |

## Work Item: LNL258V-OV-PROD-001

Status: in_progress
Linked proposal: `docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`
Linked spec: `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md`
Linked ADRs: none
Campaign: `docs/tracking/campaigns/intel-258v-platform/active.toml`
Blocks: all OpenVINO dense SLM, NPU cache/warm, route promotion, status, Rust
bridge, server, and BitNet subgraph follow-on work
Blocked by: none

### Goal

Add the OpenVINO Lunar Lake productization proposal, route identity contract,
plan README, implementation plan, and campaign work items for the docs/spec
campaign.

### Production Delta

No runtime delta. This is a docs/spec/campaign-governance PR.

### Non-Goals

Do not touch runtime code, Python proof harnesses, CI hardware receipts, model
artifact ledgers, model binaries, or route-promotion receipts. Do not promote
OpenVINO GPU/NPU, claim speedup, claim broad quality, claim native OpenCL proof,
claim BitNet QK256 proof, or claim cold one-off NPU usability.

### Acceptance

- Proposal defines why OpenVINO exists as the governed Intel-runtime lane.
- Route contract defines CPU/GPU/NPU identities, required receipt fields, proof
  families, fallback rules, AUTO/HETERO diagnostics, retokenized token evidence,
  and claim boundaries.
- Implementation plan lists PR-sized next steps.
- Campaign tracker gains docs/spec work items only.
- No runtime claims are promoted.

### Proof Commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Rollback

Revert the proposal, route contract, `plans/openvino-lunar-lake/`, and the
OpenVINO docs/spec campaign work items from the Intel 258V active manifest.
Generated campaign docs, if updated by `xtask`, should be regenerated after the
revert.

## Follow-On Docs/Spec Work Items

### LNL258V-OV-PROD-002: Dense SLM Model/Export Contract

Add `docs/specs/BITNET-SPEC-OPENVINO-DENSE-SLM.md`. Define Qwen2.5 0.5B
Instruct OpenVINO IR INT4 symmetric export requirements, model/export manifest
fields, tokenizer and prompt-template authority, proof ladder, promotion rule,
and non-goals. Do not promote routes or edit runtime code.

### LNL258V-OV-PROD-003: NPU Cold/Warm/Cache Contract

Add `docs/specs/BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md`. Require fields
for first-ever compile/infer, cached construction, cache mode, cache hit, TTFT,
decode, warm second ask, resident sessions, and low-power profiles. Keep cold
one-off NPU promotion blocked until cold load/compile is acceptable.

### LNL258V-OV-PROD-004: Quality Corpus And Phase Timing Contracts

Add `docs/specs/BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md` and
`docs/specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md`. Define corpus-v2 pass/fail
rules, failure taxonomy, direct output evidence, retokenized token marking,
profile summaries, timing fields, OpenVINO perf metrics, and speedup guardrails.

### LNL258V-OV-PROD-005: Route Promotion And BitNet Boundary Contracts

Add `docs/specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md` and
`docs/specs/BITNET-SPEC-OPENVINO-BITNET-BOUNDARY.md`. Define route states,
promotion blockers, exact-profile promotion gates, and the static BitNet-shaped
subgraph reference ladder without full BitNet/QK256 claims.

### LNL258V-OV-PROD-006: Rust Bridge And Server Contracts

Add `docs/specs/BITNET-SPEC-OPENVINO-RUST-BRIDGE.md` and
`docs/specs/BITNET-SPEC-OPENVINO-SERVER.md`. Define the Python-proof-harness to
Rust-surface bridge, receipt compatibility, exact-profile server profiles, and
the hard rail that Python harnesses stay until Rust receipts are equivalent.

## Follow-On Validation And Product Work

After Phase A docs/spec work lands, later PRs should harden receipt validators,
add an OpenVINO capability matrix, improve `receipts explain`, diagnose and fix
corpus-v2 failures, add profile-specific phase timing, run GPU/NPU benchmark
reviews, promote only exact profiles with passing receipts, add status UX, wrap
existing Python proof harnesses, and defer server and BitNet OpenVINO research
until ask/chat evidence is stable.
