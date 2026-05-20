# BITNET-SPEC-B158-LARGE-CONVERSION

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [tokenizer/prompt](BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; conversion contract only
Policy impact: no policy exception

## Purpose

Define conversion lanes for `1bitLLM/bitnet_b1_58-large` without pretending that
packed BitNet conversion is already solved in BitNet-rs.

The existing `bitnet-st2gguf` converter currently writes F16 GGUF metadata and
is not sufficient for `I2_S`, `TL1`, or `TL2` packed BitNet proof. It is useful
as a structural/reference bridge only.

## Allowed conversion lanes

| Lane | Meaning | Claim |
| --- | --- | --- |
| `hf_safetensors_structural` | Inspect HF files and config. | No inference claim. |
| `st2gguf_f16_reference` | BitNet-rs F16 GGUF structural/reference conversion. | Structural/reference only. |
| `bitnetcpp_i2s_conversion` | Reproducible upstream-compatible `I2_S` path. | Candidate packed BitNet artifact. |
| `bitnetcpp_tl1_conversion` | Reproducible upstream-compatible `TL1` path. | ARM candidate after proof. |
| `bitnetcpp_tl2_conversion` | Reproducible upstream-compatible `TL2` path. | x86 candidate after proof. |
| `third_party_gguf` | External GGUF with independent hash/authority. | Diagnostic unless approved. |

## `st2gguf_f16_reference` acceptance

A F16 structural/reference receipt must prove that the conversion:

- converts safetensors to GGUF F16;
- preserves LayerNorm tensors;
- emits sidecar metadata;
- loads structurally;
- records input and output SHA256 hashes;
- marks `packed_bitnet_claim = false`;
- does not claim `I2_S`, `TL1`, `TL2`, QK256, or packed BitNet acceleration.

## `I2_S`/`TL1`/`TL2` acceptance

A packed conversion receipt must record:

- exact conversion command;
- upstream commit or tool version;
- input artifact SHA256;
- output file path and SHA256;
- quantization/layout family;
- tensor layout verification;
- reference runner command and result;
- answer corpus pass/fail summary;
- BitNet-rs parser recognition result;
- model/kernel compatibility ledger update;
- fallback and claim booleans.

`TL1` and `TL2` routes require their own tensor/layout specs before they can be
routed through runtime kernels. Do not route TL artifacts through QK256 `I2_S`
code.

## Blocked outcome

If an authoritative conversion route cannot be reproduced, commit a blocked
receipt that records:

- command attempted;
- source revision;
- tool version or commit;
- failure mode;
- whether the failure blocks answer, backend, or benchmark work;
- why no workaround was used.
