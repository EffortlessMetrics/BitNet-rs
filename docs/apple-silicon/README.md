# Apple Silicon source-of-truth map

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](../specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md), [Apple M4 dense SLM appliance](../specs/BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md), [Apple M4 BitNet CPU/NEON](../specs/BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md), [Apple Metal phased acceleration](../specs/BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md), [Apple quality corpus](../specs/BITNET-SPEC-APPLE-QUALITY-CORPUS.md), [Apple benchmark envelope](../specs/BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE.md), [Apple reproducible run identity](../specs/BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md), [Apple MacBook auxiliary lane](../specs/BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md), [Apple service surface](../specs/BITNET-SPEC-APPLE-SERVICE-SURFACE.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; this map only routes proof families to existing authorities and new contracts
Policy impact: no policy exception

Apple Silicon work is split into proof families so one successful Mac run cannot
become an unsupported claim about every Apple backend, model family, or
acceleration path. This page is a navigation contract, not a new runtime claim
or a replacement for campaign receipts.

## Current product target

The first Apple Silicon product target is the M4 Mac Mini local appliance:

- supported dense SLMs run first on `apple-m4-cpu-neon`;
- the accepted BitNet artifact runs first on `apple-m4-cpu-neon`;
- Metal remains a phase-scoped acceleration proof lane;
- MPSGraph remains a graph/reference lane;
- Neural Engine execution is not claimed unless a future receipt proves it.

## Source-of-truth hierarchy

| Layer | Source of truth | Notes |
| --- | --- | --- |
| Apple Silicon route semantics | [`BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md`](../specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md) | Backend labels, proof families, receipt fields, and fallback rules. |
| M4 Mac Mini hardware facts | [`docs/specs/apple-m4-mac-mini-roadmap.md`](../specs/apple-m4-mac-mini-roadmap.md) | Existing M4 roadmap and hardware/backend lane framing. |
| Supported dense SLM models | [`docs/slm/apple-m4-dense-slm-model-support-matrix.md`](../slm/apple-m4-dense-slm-model-support-matrix.md) | Artifact-pinned model states and promotion gates. |
| Current M4 excellence state | [`docs/tracking/campaigns/apple-m4-inference-excellence/active.toml`](../tracking/campaigns/apple-m4-inference-excellence/active.toml) | Active campaign work items and proof commands. |
| Operator-facing narrative | [`docs/slm/apple-m4-inference-excellence.md`](../slm/apple-m4-inference-excellence.md) | Human-readable current M4 evidence narrative. |
| Historical proof campaigns | Existing `docs/tracking/campaigns/apple-m4-*` folders | Keep as historical evidence; do not delete or rewrite as current truth. |
| MacBook auxiliary lane | Apple Silicon MacBook campaign/docs and [`BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md`](../specs/BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md) | Separate machine and proof family; never substitutes for M4 Mac Mini proof. |
| Machine artifacts | `ci/hardware/apple-m4-mac-mini/**` and future MacBook paths | Receipt inputs/outputs, not broad claims by themselves. |
| General Apple Metal/MPS/NEON policy | New Apple Silicon specs linked above | Contractual rails for future work. |

## Proof families

| Proof family | Route label | Counts as | Does not count as |
| --- | --- | --- | --- |
| `apple_m4_cpu_neon_dense_slm` | `apple-m4-cpu-neon` | Supported dense Qwen-class M4 CPU/NEON evidence. | BitNet, Metal, MPSGraph, Neural Engine, MacBook, or broad Apple Silicon proof. |
| `apple_m4_cpu_neon_bitnet` | `apple-m4-cpu-neon` | Accepted BitNet artifact on M4 CPU/NEON with strict receipts. | Dense SLM, Metal, QK256 acceleration, Neural Engine, MPSGraph, or broad Apple Silicon proof. |
| `apple_m4_metal_phase` | `apple-m4-metal` | Named Metal kernel/subgraph phase with CPU parity and fallback-free receipts. | Full autoregressive Metal inference until a full route is proven. |
| `apple_m4_mpsgraph_reference` | `apple-m4-mpsgraph` | Reference/graph-lane experiments with explicit target receipts. | Native Metal or Neural Engine proof. |
| `apple_m4_neural_engine_research` | future explicit route | Research only until receipt-proven. | Any current product claim. |
| `apple_macbook_cpu_neon_bitnet` | MacBook CPU/NEON route | MacBook-specific larger-artifact or longer-soak evidence. | M4 Mac Mini runtime proof. |
| `apple_macbook_metal_phase` | MacBook Metal route | MacBook-specific Metal phase evidence. | M4 Mac Mini Metal proof or broad Apple Silicon proof. |

## Hard claim rails

- Dense Qwen SLM evidence is not BitNet evidence.
- BitNet CPU/NEON evidence is not Metal evidence.
- Metal visibility is not Metal execution.
- Metal subgraph parity is not full Metal inference.
- MPSGraph smoke is not native Metal proof.
- MPSGraph smoke is not Neural Engine proof unless the resolved target is receipt-backed.
- CPU fallback cannot count as Metal execution.
- MacBook evidence is not M4 Mac Mini runtime proof.
- M4 Mac Mini evidence is not broad Apple Silicon proof.
- QK256-on-x86/CUDA/A770 evidence is not QK256-on-Metal evidence.
- Supported dense SLMs must be artifact-pinned and tokenizer-authoritative.
- No model binaries are committed.
- Live hardware/model timing is never required in ordinary generic PR CI.

## Historical campaign rule

Do not delete old Apple campaign docs and do not create a competing "current
truth" page. New specs are contractual rails; current execution remains owned
by the active campaign manifest and proof receipts.
