# Apple Silicon Source-Of-Truth Map

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: planned `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md`
Linked specs: planned Apple Silicon contract specs under `docs/specs/`
Linked ADRs: n/a
Linked plan: `plans/apple-silicon/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support-tier promotion; docs-only authority map
Policy impact: none

Apple Silicon work in BitNet-rs is a Mac-native, receipt-backed inference
appliance lane. The first product target is an M4 Mac Mini local appliance, not
full Metal BitNet inference: supported dense SLMs run on
`apple-m4-cpu-neon`, the accepted BitNet artifact is productized on
`apple-m4-cpu-neon`, Metal remains phase-scoped proof work, MPSGraph remains a
reference/graph lane, and Neural Engine execution is not claimed without a
separate receipt-backed proof.

This page is a map, not a new competing status dashboard. It points each proof
family to its current authority and to the Apple Silicon contracts that will make
claim boundaries explicit for future work.

## Proof Families

| Proof family | Current authority | Contract direction | Claim boundary |
| --- | --- | --- | --- |
| M4 dense SLM CPU/NEON | `docs/slm/apple-m4-dense-slm-model-support-matrix.md` and `docs/slm/apple-m4-inference-excellence.md` | `BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md` | Dense Qwen-class evidence is not BitNet evidence, Metal evidence, MPSGraph evidence, Neural Engine evidence, MacBook proof, or broad Apple Silicon proof. |
| M4 BitNet CPU/NEON | `docs/specs/apple-m4-mac-mini-roadmap.md`, active M4 campaign receipts, and accepted artifact proof under `ci/hardware/apple-m4-mac-mini/**` | `BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md` | BitNet CPU/NEON evidence is not Metal, QK256 acceleration, Neural Engine, MPSGraph, dense SLM, MacBook, or broad Apple Silicon proof. |
| M4 Metal phase proof | `docs/specs/apple-m4-mac-mini-roadmap.md` and Metal kernel/fixture receipts | `BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md` | Metal visibility, smoke, and subgraph parity are not full Metal inference or speedup claims. CPU fallback cannot count as Metal execution. |
| M4 MPSGraph reference | `docs/specs/apple-m4-mac-mini-roadmap.md` | `BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` | MPSGraph smoke is not native Metal proof and is not Neural Engine proof unless the resolved target is receipt-backed. |
| M4 Neural Engine research | no product authority yet | `BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` | Neural Engine execution is not claimed by default and must have explicit target receipts. |
| MacBook auxiliary lane | `apple-silicon-macbook` campaign/docs when present | `BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md` | MacBook receipts use a distinct machine ID and proof family and do not prove M4 Mac Mini runtime behavior. |
| Apple quality corpora | `docs/slm/apple-m4-inference-excellence.md` and active M4 campaign receipts | `BITNET-SPEC-APPLE-QUALITY-CORPUS.md` | Dense and BitNet corpora stay separate; mechanical pass rates are exact-profile evidence, not broad model quality claims. |
| Apple benchmark envelope | `docs/slm/apple-m4-inference-excellence.md` and active M4 campaign receipts | `BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE.md` | M4 Mac Mini timing envelopes are not broad Apple Silicon, Metal, MacBook, or Neural Engine benchmark claims. |
| Reproducible run identity | active M4 campaign receipts and generated reports | `BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md` | Comparisons require matching machine, OS, build, model, tokenizer, prompt, backend, fallback, corpus/profile, seed, and timing identity. |
| Mac service surface | active M4 campaign and `bitnet mac ...` operator surfaces | `BITNET-SPEC-APPLE-SERVICE-SURFACE.md` | Ask, chat, and serve readiness are exact-profile states with per-request receipts; they are not generic production-hosting claims. |

## Source-Of-Truth Hierarchy

| Layer | Source of truth |
| --- | --- |
| Apple Silicon route semantics | planned `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` |
| M4 Mac Mini hardware facts | `docs/specs/apple-m4-mac-mini-roadmap.md` |
| Supported dense SLM models | `docs/slm/apple-m4-dense-slm-model-support-matrix.md` |
| Current M4 excellence state | `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml` |
| Operator-facing narrative | `docs/slm/apple-m4-inference-excellence.md` |
| Historical proof campaigns | existing `docs/tracking/campaigns/apple-m4-*` folders |
| MacBook auxiliary lane | `apple-silicon-macbook` campaign/docs when present |
| Machine artifacts | `ci/hardware/apple-m4-mac-mini/**` and future MacBook receipt paths |
| General Apple Metal/MPS/NEON policy | planned Apple Silicon specs under `docs/specs/` |

## Hard Claim Rails

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

## Historical Campaign Rule

Do not delete old Apple campaign docs. They preserve proof history. Future Apple
Silicon specs are contractual boundaries for new work; they do not retroactively
promote old receipts or create new runtime claims.
