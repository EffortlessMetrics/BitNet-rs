# BitNet-rs NPU source of truth

## Purpose

This directory is the source-of-truth map for BitNet-rs NPU productization.
NPU support is a governed low-power and warm/resident inference lane, not a
catch-all accelerator bucket and not a replacement for the CPU, CUDA, OpenCL, or
AVX lanes.

The current implementation target is Intel AI Boost NPU on Lunar Lake 258V
through OpenVINO NPU. Future NPU families such as Apple Neural Engine,
Qualcomm Hexagon, and AMD Ryzen AI require their own proof families; they do not
inherit Intel Lunar Lake proof.

## Authority stack

| Role | Artifact | Scope |
| --- | --- | --- |
| Source-of-truth rules | [`docs/reference/SPEC_SYSTEM.md`](../reference/SPEC_SYSTEM.md) | Repository-wide artifact authority and generated-status rules. |
| Campaign state | [`docs/tracking/campaigns/intel-npu/active.toml`](../tracking/campaigns/intel-npu/active.toml) | Current Intel NPU work items, allowed paths, proof commands, and claim boundaries. |
| Current roadmap/spec | [`docs/specs/intel-lunar-lake-npu-roadmap.md`](../specs/intel-lunar-lake-npu-roadmap.md) | Intel Lunar Lake/OpenVINO proof levels and current not-claims. |
| Rollout plan | [`plans/npu/implementation-plan.md`](../../plans/npu/implementation-plan.md) | PR-by-PR NPU productization sequence from governance to route promotion. |
| Cross-lane product context | [`docs/proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md`](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md) | OpenVINO Lunar Lake dense-SLM context; not itself NPU proof. |

## Current target and future families

| NPU family | Current role | Proof inheritance |
| --- | --- | --- |
| Intel AI Boost NPU on Lunar Lake 258V via OpenVINO NPU | First real implementation lane. | Own Intel Lunar Lake/OpenVINO receipts only. |
| Apple Neural Engine | Future research family. | Does not inherit Intel proof. |
| Qualcomm Hexagon | Future research family. | Does not inherit Intel proof. |
| AMD Ryzen AI | Future research family. | Does not inherit Intel proof. |

A receipt or passing test from one NPU family must not be described as generic
NPU support for another family.

## Merged Intel NPU evidence

| Work item | Status | Evidence boundary |
| --- | --- | --- |
| `NPU-002` | merged | Intel NPU requested/selected backend identity is preserved and is not mapped to Metal, CUDA, generic GPU, or CPU fallback. |
| `NPU-003` | merged | OpenVINO NPU runtime detection fields exist. |
| `NPU-004` | merged | CLI can emit OpenVINO NPU runtime visibility receipts. |
| `NPU-005` | merged | Tiny static OpenVINO NPU graph smoke path exists. |
| `NPU-006` | merged | NPU receipts include structured backend, runtime, shape, and fallback fields. |
| `NPU-007` | merged | Static BitNet RMSNorm subgraph parity through OpenVINO NPU. |
| `NPU-008` | merged | Static BitNet linear-projection subgraph parity through OpenVINO NPU. |
| `NPU-009` | merged | OpenVINO llama.cpp GGUF reference lane is tracked as external evidence only. |
| `NPU-010` | merged | Live 258V OpenVINO 2026.1 NPU visibility, tiny graph smoke, RMSNorm parity, and linear parity receipts are recorded. |
| `NPU-011` | merged | Static BitNet-shaped FFN/ReLU2 subgraph parity through OpenVINO NPU. |

This evidence currently proves runtime visibility, one tiny static graph smoke
path, and selected static BitNet-shaped subgraph parity only.

## Not claimed

Current NPU evidence does not prove:

- native bitnet-rs NPU inference;
- full BitNet inference on NPU;
- packed QK256 decode on NPU;
- NPU acceleration for BitNet;
- NPU server readiness;
- NPU speedup;
- NPU full residency;
- broad dense-SLM quality on NPU;
- CPU fallback as NPU execution.

These not-claims must remain visible in NPU status surfaces until profile-scoped
receipts prove otherwise.

## Product direction

The first useful NPU product route is a dense small-language-model OpenVINO
GenAI route for warm/resident, low-power, short-answer profiles. BitNet-specific
NPU work remains a static subgraph and graph-lowering research lane until there
is receipt-backed evidence for broader decode, quality, performance, and
residency claims.

The preferred ordering is:

```text
NPU identity
→ OpenVINO NPU visibility
→ static graph smoke
→ static BitNet-shaped subgraph parity
→ dense SLM OpenVINO GenAI route
→ cold/cache/warm/resident benchmark
→ exact-profile route promotion
→ selected BitNet subgraph expansion
→ hybrid CPU/GPU/NPU routing
→ full inference research
```

## Hard rules

- NPU detection is not NPU execution.
- OpenVINO NPU smoke is not full inference.
- Static BitNet-shaped subgraph parity is not full BitNet inference.
- Dense SLM NPU proof is not BitNet QK256 proof.
- OpenVINO GPU proof is not NPU proof.
- Arc 140V OpenCL proof is not NPU proof.
- CPU fallback cannot count as NPU execution.
- AUTO/HETERO proof is not selected NPU proof unless execution devices are recorded.
- Cold one-off usability must not be claimed from warm or cached measurements.
- Speedup must not be claimed without quality-gated exact-profile benchmark receipts.
- Full residency must not be claimed without per-phase residency evidence.
- Live OpenVINO NPU execution must remain opt-in and must not be added to ordinary generic PR CI.
