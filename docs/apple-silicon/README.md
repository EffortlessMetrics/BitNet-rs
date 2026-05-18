# Apple Silicon Source-of-Truth Map

This page is the navigation contract for Apple Silicon work in BitNet-rs. It
prevents one successful Mac run from becoming an unsupported claim about every
Apple machine, backend, model family, or acceleration path.

It is intentionally a map, not a new proof source. The linked roadmap, matrix,
campaign, receipts, and future specs remain the authorities for their own
layers. Historical campaign pages stay in place for auditability.

## Product Target

The first Apple Silicon product target is a Mac-native local inference appliance
for the M4 Mac mini:

```text
M4 Mac mini local appliance:
  dense SLMs: product path on apple-m4-cpu-neon first
  BitNet: accepted artifact on apple-m4-cpu-neon first
  Metal: phase-scoped acceleration proof only
  MPSGraph: reference/graph lane only
  Neural Engine: not claimed unless explicitly receipt-proven
```

This target does not claim full `apple-m4-metal` inference, Neural Engine
execution, MPSGraph model inference, broad Apple Silicon support, or MacBook
proof for M4 Mac mini runtime behavior.

## Proof Families

Apple proof families are separated by machine, model family, backend, runtime
API, and fallback state:

| Proof family | Machine | Model family | Backend/runtime scope | Current role |
| --- | --- | --- | --- | --- |
| `apple_m4_cpu_neon_dense_slm` | `apple-m4-mac-mini` | dense SLM | `apple-m4-cpu-neon` / CPU | Product appliance path for supported Qwen-class dense SLMs. |
| `apple_m4_cpu_neon_bitnet` | `apple-m4-mac-mini` | BitNet | `apple-m4-cpu-neon` / CPU | Productization path for the accepted BitNet artifact. |
| `apple_m4_metal_phase` | `apple-m4-mac-mini` | dense SLM or BitNet phase fixtures | `apple-m4-metal` / Metal | Phase-scoped acceleration research and parity proof only. |
| `apple_m4_mpsgraph_reference` | `apple-m4-mac-mini` | graph/reference fixtures | MPSGraph | Graph/reference lane; not native Metal proof. |
| `apple_m4_neural_engine_research` | `apple-m4-mac-mini` | receipt-proven only | resolved Neural Engine target only if proven | Research lane with no current product claim. |
| `apple_macbook_cpu_neon_bitnet` | `apple-silicon-macbook` | BitNet | CPU/NEON | Auxiliary larger-artifact and longer-soak lane; not M4 proof. |
| `apple_macbook_metal_phase` | `apple-silicon-macbook` | phase fixtures | Metal | Auxiliary parity experiments; not M4 proof. |

Every receipt that contributes to one family must identify its `machine_id`,
`model_family`, `requested_backend`, `selected_backend`, `runtime_api`,
`fallback_used`, and `proof_family` so dashboard and operator claims cannot mix
families.

## Authority Table

| Layer | Source of truth | Notes |
| --- | --- | --- |
| Apple Silicon route semantics | Future `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` | Contractual labels, fallback rules, and proof-family receipt fields. Until it lands, this map is only routing guidance. |
| M4 Mac mini hardware facts | [`docs/specs/apple-m4-mac-mini-roadmap.md`](../specs/apple-m4-mac-mini-roadmap.md) | Defines the M4 lane as Metal-first, MPSGraph reference, and CPU/NEON fallback/parity. |
| Supported dense SLM models | [`docs/slm/apple-m4-dense-slm-model-support-matrix.md`](../slm/apple-m4-dense-slm-model-support-matrix.md) | Defines the supported/default/candidate dense SLM matrix and promotion gates. |
| Current M4 excellence state | [`docs/tracking/campaigns/apple-m4-inference-excellence/active.toml`](../tracking/campaigns/apple-m4-inference-excellence/active.toml) | Machine-readable active campaign queue and proof commands. |
| Operator-facing narrative | [`docs/slm/apple-m4-inference-excellence.md`](../slm/apple-m4-inference-excellence.md) | Human-readable M4 appliance status, dashboards, and claim boundaries. |
| Historical proof campaigns | Existing `docs/tracking/campaigns/apple-m4-*` folders | Historical proof surfaces remain audit records and must not be deleted during cleanup. |
| MacBook auxiliary lane | Future Apple MacBook campaign/docs | MacBook can aid larger artifacts and longer soaks, but must not replace M4 Mac mini proof. |
| Machine artifacts | `ci/hardware/apple-m4-mac-mini/**` and future MacBook paths | Receipt evidence, not source-of-truth prose. No model binaries. |
| General Apple Metal/MPS/NEON policy | Future Apple Silicon specs | Contractual docs added by the Apple Silicon docs rollout. |

## Current Mature Lane: Dense SLM on M4 CPU/NEON

The dense SLM lane is the most mature Apple product path. The support matrix
keeps it separate from BitNet, QK256, full Metal inference, MPSGraph, and Neural
Engine work.

Current supported dense M4 models are:

```text
qwen2.5-0.5b-instruct-q8_0       default
qwen2.5-0.5b-instruct-q4_k_m     supported, storage-conscious
qwen2.5-1.5b-instruct-q4_k_m     supported, non-default
```

These rows are M4 Mac mini, Apple CPU/NEON, supported dense model identity
claims only. They are not broad Apple Silicon, BitNet, Metal, MPSGraph, Neural
Engine, or MacBook claims.

## BitNet Lane: M4 CPU/NEON First

BitNet Apple work is productized on `apple-m4-cpu-neon` before any full Metal
route claim. Accepted BitNet evidence must stay BitNet-specific and must not use
dense Qwen receipts as substitute proof. Metal, MPSGraph, Neural Engine, and
QK256 acceleration claims require separate receipt-backed proof families.

## Metal Lane: Phase-Scoped Proof Only

Metal is a useful proof lane when each kernel or subgraph has explicit CPU
reference parity and fallback-free receipts. The current source-of-truth cleanup
does not promote full Metal inference. In particular:

- Metal visibility is not Metal execution.
- CPU fallback is not Metal execution.
- Metal subgraph parity is not full autoregressive inference.
- MPSGraph smoke is not native Metal proof.
- Phase-local timing is not a broad speedup claim.

## Claim Rails

These rails apply across Apple specs, plans, active manifests, status docs, and
operator-facing pages:

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

## Cleanup Rules

- Do not delete old campaign docs.
- Do not create another competing “current truth” page.
- Add Apple Silicon specs as contractual rails, not as runtime promotions.
- Keep source-of-truth roles separated: proposals explain why, specs define
  behavior, plans sequence work, active goals describe current execution, and
  receipts prove exact runs.
