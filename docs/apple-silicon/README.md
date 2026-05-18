# Apple Silicon source-of-truth map

Status: draft
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md` (planned)
Linked specs: `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` (planned)
Linked ADRs: n/a
Linked plan: `plans/apple-silicon/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No support-tier promotion; this map records claim boundaries and authorities only.
Policy impact: No policy exception.

This directory is the navigation point for Apple Silicon documentation. It does
not replace existing M4 campaign pages, hardware receipts, support matrices, or
future specs. It points each proof family at its current authority so future M4
Mac Mini, MacBook, dense SLM, BitNet CPU/NEON, MPSGraph, Neural Engine, and Metal
phase work cannot accidentally borrow evidence from another lane.

## Product target

The current Apple Silicon product target is an M4 Mac Mini local inference
appliance, not full Metal BitNet inference:

```text
M4 Mac Mini local appliance:
  dense SLMs: product path on apple-m4-cpu-neon first
  BitNet: accepted artifact on apple-m4-cpu-neon first
  Metal: phase-scoped acceleration proof only
  MPSGraph: reference/graph lane only
  Neural Engine: not claimed unless explicitly receipt-proven
```

## Current source-of-truth hierarchy

| Layer | Current authority | Notes |
| --- | --- | --- |
| Apple Silicon route semantics | `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` (planned) | Will define route IDs, backend labels, fallback rules, and proof-family separation. Until it lands, this map and the active M4 campaign constraints are navigation rails only. |
| M4 Mac Mini hardware facts | `docs/specs/apple-m4-mac-mini-roadmap.md` | Defines the M4 Mac Mini lane as Metal-first, MPSGraph reference, and CPU/NEON fallback/parity, with explicit not-claims. |
| Supported dense SLM models | `docs/slm/apple-m4-dense-slm-model-support-matrix.md` | Owns supported/default/candidate/rejected dense model identity, artifact pinning, tokenizer authority, prompt templates, and promotion gates. |
| Current M4 excellence state | `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml` | Owns the active M4 inference-excellence queue, proof commands, allowed paths, forbidden paths, and claim boundaries. |
| Operator-facing narrative | `docs/slm/apple-m4-inference-excellence.md` | Explains current M4 operator evidence, dashboards, quality, benchmark, regression, and envelope status. |
| Historical proof campaigns | Existing `docs/tracking/campaigns/apple-m4-*` folders | Preserve historical PR sequencing and proof receipts; they should not be deleted or rewritten into a new current-truth page. |
| MacBook auxiliary lane | `docs/apple-silicon/macbook-lane.md` and `docs/apple-silicon/m3-macbook-air-roadmap.md` | Separate machine and proof family; MacBook evidence can inform exploration but cannot prove M4 Mac Mini runtime behavior. |
| Machine artifacts | `ci/hardware/apple-m4-mac-mini/**` and future MacBook artifact paths | Own per-run receipts, hardware/model identity, timing evidence, and replayable proof material. |
| General Apple Metal/MPS/NEON policy | Planned Apple Silicon proposal/spec set | Will define durable policy for route labels, quality corpora, benchmark envelopes, run identity, service readiness, and MacBook auxiliary work. |

## Apple proof families

Apple receipts and docs must identify one proof family at a time:

| Proof family | Machine | Runtime/API lane | Model family | May prove | Must not prove |
| --- | --- | --- | --- | --- | --- |
| `apple_m4_cpu_neon_dense_slm` | `apple-m4-mac-mini` | CPU/NEON | Dense SLM | Supported Qwen-class dense local answers, quality corpora, matching-history receipts, and operator envelope for exact model/tokenizer/backend identity. | BitNet behavior, Metal execution, MPSGraph execution, Neural Engine execution, MacBook behavior, or broad Apple Silicon behavior. |
| `apple_m4_cpu_neon_bitnet` | `apple-m4-mac-mini` | CPU/NEON | BitNet | Accepted BitNet artifact local answers and BitNet-specific eval/benchmark/chat/serve gates when each gate has receipts. | Dense SLM behavior, Metal execution, QK256 acceleration, MPSGraph execution, Neural Engine execution, MacBook behavior, or broad Apple Silicon behavior. |
| `apple_m4_metal_phase` | `apple-m4-mac-mini` | Metal | Phase-scoped dense or BitNet subgraph | Named kernels/subgraphs with CPU reference parity, fallback-free phase receipts, and phase-local timing. | Full autoregressive Metal inference, end-to-end speedup, QK256-on-Metal, MPSGraph, Neural Engine, or CPU fallback execution. |
| `apple_m4_mpsgraph_reference` | `apple-m4-mac-mini` | MPSGraph | Reference/graph lane | Explicit graph/reference experiments with receipts identifying MPSGraph. | Native Metal proof or Neural Engine proof unless a resolved target receipt explicitly proves it. |
| `apple_m4_neural_engine_research` | `apple-m4-mac-mini` | Neural Engine research | Research only | Nothing product-facing until a future receipt proves target resolution and execution. | Metal, MPSGraph, CPU/NEON, dense, BitNet, or broad Apple Silicon claims. |
| `apple_macbook_cpu_neon_bitnet` | `apple-silicon-macbook` / current M3 Air lane | CPU/NEON | BitNet or larger artifact exploration | MacBook-specific storage, longer soaks, and external reference comparisons with `counts_as_m4_mac_mini_proof=false`. | M4 Mac Mini runtime behavior or broad Apple Silicon support. |
| `apple_macbook_metal_phase` | `apple-silicon-macbook` / current M3 Air lane | Metal phase | Phase-scoped experiments | MacBook-specific Metal/CPU parity experiments with separate machine identity. | M4 Mac Mini Metal behavior, full Metal inference, or broad Apple Silicon support. |

## Hard claim rails

These rails apply to Apple docs, campaign manifests, plans, status summaries, and
receipt interpretation:

```text
Dense Qwen SLM evidence is not BitNet evidence.
BitNet CPU/NEON evidence is not Metal evidence.
Metal visibility is not Metal execution.
Metal subgraph parity is not full Metal inference.
MPSGraph smoke is not native Metal proof.
MPSGraph smoke is not Neural Engine proof unless the resolved target is receipt-backed.
CPU fallback cannot count as Metal execution.
MacBook evidence is not M4 Mac Mini runtime proof.
M4 Mac Mini evidence is not broad Apple Silicon proof.
QK256-on-x86/CUDA/A770 evidence is not QK256-on-Metal evidence.
Supported dense SLMs must be artifact-pinned and tokenizer-authoritative.
No model binaries are committed.
Live hardware/model timing is never required in ordinary generic PR CI.
```

## Historical campaign handling

Do not delete historical Apple campaign documents. They remain audit evidence for
how a proof family reached its current state. New Apple Silicon specs are
contractual rails for future work; they are not retroactive support promotions,
not new runtime evidence, and not replacements for per-machine receipts.

## Planned contract documents

The rollout plan in `plans/apple-silicon/implementation-plan.md` sequences the
planned Apple proposal and specs. Those documents should be added as separate PRs
unless a future active work item explicitly combines them.
